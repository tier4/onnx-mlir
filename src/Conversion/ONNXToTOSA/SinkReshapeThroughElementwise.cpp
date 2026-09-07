/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ SinkReshapeThroughElementwise.cpp ----------------------------===//
//
// Sinks tosa.reshape ops past their elementwise consumers.
//
// tosa.matmul takes rank-3 operands with a matching batch dimension, so for an
// ONNX MatMul with a rank-2 (unbatched) weight and a leading dim != 1 the
// lowering in Math/MatMul.cpp folds all leading dims of the LHS into M and
// uses a batch of 1.  Folding is the right choice -- the alternative is tiling
// the weight, which would replicate it once per batch element.  But the ONNX
// result type is still the unfolded shape, so step 3 of that lowering reshapes
// the result back, leaving a reshape wedged between the matmul and its
// epilogue (bias add, activation).
//
// That reshape can block fusion downstream: backends that form fusion groups
// from producer-consumer chains often do not fuse across reshapes, so the
// epilogue ends up separated from the matmul.
//
// Sinking the reshape below the epilogue restores a direct matmul -> epilogue
// edge.  Each rewrite step is individually semantics-preserving, so a chain
// that fails to sink all the way just fuses less -- it never miscompiles.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

/// tosa.mul and tosa.cast are elementwise but are not built on
/// Tosa_ElementwiseOp, so they do not carry the trait and must be named.
static bool isTosaElementwise(Operation *op) {
  return op->hasTrait<mlir::OpTrait::tosa::TosaElementwiseOperator>() ||
         isa<mlir::tosa::MulOp, mlir::tosa::CastOp>(op);
}

static RankedTensorType staticRankedType(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape())
    return nullptr;
  return t;
}

/// An operand whose only non-unit dims sit where `from` and `into` already
/// agree broadcasts correctly against both shapes, so it can be reused as is.
static bool broadcastsAgainstBoth(
    ArrayRef<int64_t> operand, ArrayRef<int64_t> from, ArrayRef<int64_t> into) {
  if (operand.size() != from.size() || from.size() != into.size())
    return false;
  for (auto [o, f, i] : llvm::zip(operand, from, into))
    if (o != 1 && f != i)
      return false;
  return true;
}

/// Is `v` the result of a matmul, reached only through elementwise ops that
/// already produce `shape`?  This fixes the direction of the rewrite: chains
/// are pulled into the shape their matmul produces and never pushed back out,
/// so the pattern cannot ping-pong between two shapes.
static bool isMatmulRooted(Value v, ArrayRef<int64_t> shape, int depth = 0) {
  if (depth > 8)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<mlir::tosa::MatMulOp>(def))
    return true;
  if (!isTosaElementwise(def))
    return false;
  auto defTy = staticRankedType(def->getResult(0));
  if (!defTy || defTy.getShape() != shape)
    return false;
  return llvm::any_of(def->getOperands(), [&](Value o) {
    auto ot = staticRankedType(o);
    return ot && ot.getShape() == shape && isMatmulRooted(o, shape, depth + 1);
  });
}

/// Rewrites `ew(reshape(x), ...)` into `reshape(ew(x, ...))`.
struct SinkReshapePastElementwise : public RewritePattern {
  SinkReshapePastElementwise(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isTosaElementwise(op) || op->getNumResults() != 1)
      return failure();
    RankedTensorType resTy = staticRankedType(op->getResult(0));
    if (!resTy)
      return failure();
    ArrayRef<int64_t> into = resTy.getShape();

    // Pick the reshape that defines the shape we sink into.  It must cover the
    // whole result shape and only re-lay-out the dims (same rank, same count).
    mlir::tosa::ReshapeOp driver;
    for (Value v : op->getOperands()) {
      auto rs = v.getDefiningOp<mlir::tosa::ReshapeOp>();
      if (!rs)
        continue;
      RankedTensorType srcTy = staticRankedType(rs.getInput1());
      RankedTensorType dstTy = staticRankedType(rs.getResult());
      if (!srcTy || !dstTy)
        continue;
      if (dstTy.getShape() != into || srcTy.getRank() != dstTy.getRank())
        continue;
      if (srcTy.getShape() == dstTy.getShape())
        continue;
      if (!isMatmulRooted(rs.getInput1(), srcTy.getShape()))
        continue;
      driver = rs;
      break;
    }
    if (!driver)
      return failure();
    ArrayRef<int64_t> from =
        staticRankedType(driver.getInput1()).getShape();

    // Bring every operand into the `from` shape, bailing if any cannot follow.
    SmallVector<Value> operands;
    SmallVector<std::pair<unsigned, Value>> needReshape;
    for (auto [idx, v] : llvm::enumerate(op->getOperands())) {
      if (v == driver.getResult()) {
        operands.push_back(driver.getInput1());
        continue;
      }
      RankedTensorType vt = staticRankedType(v);
      // Non-tensor and differently-ranked operands (tosa.mul's shift, shape
      // operands) are shape-agnostic and pass through untouched.
      if (!vt || vt.getRank() != resTy.getRank()) {
        operands.push_back(v);
        continue;
      }
      if (broadcastsAgainstBoth(vt.getShape(), from, into)) {
        operands.push_back(v);
        continue;
      }
      if (vt.getShape() != into)
        return failure();
      // A full-shape operand has to be re-laid-out too.  If it is itself a
      // reshape out of `from`, reuse its source so the two cancel.
      if (auto rs = v.getDefiningOp<mlir::tosa::ReshapeOp>()) {
        RankedTensorType s = staticRankedType(rs.getInput1());
        if (s && s.getShape() == from) {
          operands.push_back(rs.getInput1());
          continue;
        }
      }
      operands.push_back(nullptr);
      needReshape.emplace_back(idx, v);
    }

    for (auto &[idx, v] : needReshape) {
      auto vt = cast<RankedTensorType>(v.getType());
      operands[idx] =
          onnx_mlir::tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter,
              op->getLoc(), RankedTensorType::get(from, vt.getElementType()), v,
              mlir::tosa::getTosaConstShape(rewriter, op->getLoc(), from));
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(RankedTensorType::get(from, resTy.getElementType()));
    state.addAttributes(op->getAttrs());
    Operation *sunk = rewriter.create(state);

    Value restored = onnx_mlir::tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(
        rewriter, op->getLoc(), resTy, sunk->getResult(0),
        mlir::tosa::getTosaConstShape(rewriter, op->getLoc(), into));
    rewriter.replaceOp(op, restored);
    return success();
  }
};

struct SinkReshapeThroughElementwisePass
    : public PassWrapper<SinkReshapeThroughElementwisePass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SinkReshapeThroughElementwisePass)

  StringRef getArgument() const override {
    return "tosa-sink-reshape-through-elementwise";
  }
  StringRef getDescription() const override {
    return "Sink tosa.reshape past elementwise consumers so a matmul and its "
           "epilogue stay adjacent and can be fused by the backend.";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SinkReshapePastElementwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSinkReshapeThroughElementwisePass() {
  return std::make_unique<SinkReshapeThroughElementwisePass>();
}

} // namespace onnx_mlir
