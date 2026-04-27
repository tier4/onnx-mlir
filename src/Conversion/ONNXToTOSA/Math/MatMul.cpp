/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- MatMul.cpp - Lowering MatMul Op --------------------===//
//
// Copyright 2026
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMatMulOpLoweringToTOSA : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value A(adaptor.getA());
    Value B(adaptor.getB());
    auto aType = mlir::dyn_cast<RankedTensorType>(A.getType());
    auto bType = mlir::dyn_cast<RankedTensorType>(B.getType());

    if (!aType || !bType)
      return rewriter.notifyMatchFailure(op, "inputs must be ranked tensors");

    auto resultType = mlir::dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    if (aType.getRank() < 2 || bType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "TOSA MatMul lowering requires rank >= 2 inputs");

    // Step 1: normalize both inputs to 3-D, using the result type's
    // leading dims as the broadcast target.
    auto normalized = normalizeInputsTo3D(
        op, A, B, aType, bType, resultType.getShape(), rewriter, loc);
    if (failed(normalized))
      return failure();
    auto [a3d, b3d] = *normalized;

    // Step 2: tosa.matmul on 3-D operands.
    Value mm =
        tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter, loc,
            RankedTensorType::get(kDynamic3D, resultType.getElementType()),
            a3d, b3d)
            .getResult();

    // Step 3: reshape back to the expected result shape if needed.  The
    // reshape's const_shape allows at most one '-1' for a dynamic dim;
    // result types with multiple dynamic dims (other than batch) cannot be
    // recovered from the 3-D matmul output alone.
    auto mmType = mlir::cast<RankedTensorType>(mm.getType());
    Value result;
    if (shapesEqual(mmType.getShape(), resultType.getShape())) {
      result = mm;
    } else {
      if (countDynamic(resultType.getShape()) > 1)
        return rewriter.notifyMatchFailure(op,
            "MatMul: result type has multiple dynamic dims, cannot unfold "
            "matmul output");
      result = tosaBuilder.reshape(mm, resultType.getShape());
    }
    rewriter.replaceOp(op, {result});
    return success();
  }

private:
  static constexpr int64_t kDyn = ShapedType::kDynamic;
  static constexpr std::array<int64_t, 3> kDynamic3D = {kDyn, kDyn, kDyn};

  /// Reshape \p v to a tensor with the given target shape.
  static Value reshapeTo(Value v, Type elemType, ArrayRef<int64_t> shape,
      Location loc, ConversionPatternRewriter &rewriter) {
    return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, loc,
               RankedTensorType::get(
                   SmallVector<int64_t>(shape.size(), kDyn), elemType),
               v, mlir::tosa::getTosaConstShape(rewriter, loc, shape))
        .getResult();
  }

  /// Tile \p v with the given multiples.  The result rank equals \p v's rank.
  static Value tileWithMultiples(Value v, Type elemType,
      ArrayRef<int64_t> multiples, Location loc,
      ConversionPatternRewriter &rewriter) {
    int64_t rank = multiples.size();
    return tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(SmallVector<int64_t>(rank, kDyn), elemType), v,
        mlir::tosa::getTosaConstShape(rewriter, loc, multiples))
        .getResult();
  }

  /// Product of \p shape's dims, returning kDynamic if any dim is dynamic.
  /// In a TOSA const_shape this kDynamic is encoded as -1, which tosa.reshape
  /// treats as "infer from total elements".
  static int64_t dimProduct(ArrayRef<int64_t> shape) {
    int64_t product = 1;
    for (int64_t d : shape) {
      if (ShapedType::isDynamic(d))
        return kDyn;
      product *= d;
    }
    return product;
  }

  static int64_t countDynamic(ArrayRef<int64_t> shape) {
    return llvm::count_if(shape, ShapedType::isDynamic);
  }

  /// Equal dim-by-dim, treating dynamic-vs-dynamic as a match.
  static bool shapesEqual(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (ShapedType::isDynamic(a[i]) && ShapedType::isDynamic(b[i]))
        continue;
      if (a[i] != b[i])
        return false;
    }
    return true;
  }

  /// Broadcast \p x's leading (batch) dims to \p batchShape and fold them
  /// into one, producing a 3-D tensor [batchProduct, innerR, innerC].
  /// Returns nullptr (with notifyMatchFailure already called) on unsupported
  /// shapes.
  Value broadcastAndFold(ONNXMatMulOp op, Value x, RankedTensorType xType,
      ArrayRef<int64_t> batchShape, int64_t innerR, int64_t innerC,
      ConversionPatternRewriter &rewriter, Location loc) const {
    int64_t batchRank = batchShape.size();
    int64_t rank = xType.getRank();
    int64_t leadRank = rank - 2;
    Type elem = xType.getElementType();
    auto origShape = xType.getShape();

    // Aligned shape: prepend 1s so the operand has rank batchRank + 2.
    SmallVector<int64_t> aligned;
    aligned.reserve(batchRank + 2);
    for (int64_t i = 0; i < batchRank - leadRank; ++i)
      aligned.push_back(1);
    for (int64_t i = 0; i < leadRank; ++i)
      aligned.push_back(origShape[i]);
    aligned.push_back(innerR);
    aligned.push_back(innerC);

    // Per-axis tile multiples for the batch portion.  TOSA tile takes
    // a constant multiples shape, so any required tile factor must be static.
    bool needTile = false;
    bool allBatchOnes = batchRank > 0;
    SmallVector<int64_t> multiples(batchRank + 2, 1);
    for (int64_t i = 0; i < batchRank; ++i) {
      int64_t a = aligned[i];
      int64_t b = batchShape[i];
      if (a != 1)
        allBatchOnes = false;
      if (a == b)
        continue; // exact match; covers static==static and dyn==dyn
      if (ShapedType::isDynamic(b)) {
        // Target is dynamic. If the operand contributes 1, we'd have to
        // tile by a runtime value — TOSA tile requires static multiples.
        if (a == 1) {
          (void)rewriter.notifyMatchFailure(op,
              "MatMul: cannot tile a unit batch dim by a dynamic broadcast "
              "target");
          return nullptr;
        }
        // Operand dim is static and non-1, target is dynamic: shape inference
        // implies they're equal at runtime.
        continue;
      }
      if (ShapedType::isDynamic(a)) {
        // Operand dim is dynamic, target is static. Tile-vs-no-tile depends
        // on the runtime value, which we don't know.
        if (b == 1)
          continue; // target is 1, no tile needed regardless of a.
        (void)rewriter.notifyMatchFailure(op,
            "MatMul: cannot disambiguate dynamic operand batch vs static "
            "broadcast target");
        return nullptr;
      }
      if (a == 1) {
        multiples[i] = b;
        needTile = true;
        continue;
      }
      (void)rewriter.notifyMatchFailure(
          op, "MatMul batch dims are not broadcast-compatible");
      return nullptr;
    }

    int64_t batchProduct = dimProduct(batchShape);
    SmallVector<int64_t, 3> folded3D = {batchProduct, innerR, innerC};

    // No tiling needed: collapse straight to [batchProduct, R, C].
    if (!needTile) {
      // Skip the reshape if the operand already has the target shape
      // (dim-by-dim, treating dynamic-vs-dynamic as a match).
      if (rank == 3 && shapesEqual(origShape, folded3D))
        return x;
      // tosa.reshape's const_shape allows at most one '-1'.  If two of
      // [batchProduct, innerR, innerC] are dynamic we cannot encode the
      // fold reshape.
      if (countDynamic(folded3D) > 1) {
        (void)rewriter.notifyMatchFailure(op,
            "MatMul: too many dynamic dims in folded operand shape");
        return nullptr;
      }
      return reshapeTo(x, elem, folded3D, loc, rewriter);
    }

    // From here on, a per-axis tile is needed.  All the tile multiples we
    // collected are static (we returned early if any were dynamic), but the
    // optimizations and final fold need a static batchProduct too.
    if (ShapedType::isDynamic(batchProduct)) {
      (void)rewriter.notifyMatchFailure(op,
          "MatMul: dynamic batch product after per-axis tile");
      return nullptr;
    }

    // Optimization: if the operand carries no real batch dims (all 1s after
    // alignment), we can fold to [1, R, C] and tile by [batchProduct, 1, 1]
    // instead of materializing a batchRank+2 intermediate.
    if (allBatchOnes) {
      Value flat = reshapeTo(x, elem, {1, innerR, innerC}, loc, rewriter);
      Value tiled = tileWithMultiples(
          flat, elem, {batchProduct, 1LL, 1LL}, loc, rewriter);
      return tiled;
    }

    // General case: align rank, tile per-axis, fold leading dims.
    Value alignedV =
        (rank == batchRank + 2) ? x : reshapeTo(x, elem, aligned, loc, rewriter);
    Value tiled = tileWithMultiples(alignedV, elem, multiples, loc, rewriter);
    return reshapeTo(tiled, elem, folded3D, loc, rewriter);
  }

  /// Transform both inputs into 3-D tensors with matching batch dimensions,
  /// ready for tosa.matmul.  Calls notifyMatchFailure on unsupported cases.
  FailureOr<std::pair<Value, Value>> normalizeInputsTo3D(ONNXMatMulOp op,
      Value A, Value B, RankedTensorType aType, RankedTensorType bType,
      ArrayRef<int64_t> resultShape, ConversionPatternRewriter &rewriter,
      Location loc) const {
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    Type aElem = aType.getElementType();
    Type bElem = bType.getElementType();
    int64_t M = aShape[aRank - 2];
    int64_t Ka = aShape[aRank - 1];
    int64_t Kb = bShape[bRank - 2];
    int64_t N = bShape[bRank - 1];

    // Fast path: B is a 2-D weight.  Fold all of A's non-K dims into one
    // flat row dim so both operands stay at batch = 1 and we avoid tiling
    // the weight.  A's leading dims may include dynamic dims — they fold
    // into a single dynamic dim, encoded as -1 in the const_shape.  Each
    // reshape's target must have at most one dynamic dim.
    if (bRank == 2 && aRank > 2) {
      int64_t flatM = dimProduct(aShape.drop_back(1));
      SmallVector<int64_t, 3> aFold = {1, flatM, Ka};
      SmallVector<int64_t, 3> bFold = {1, Kb, N};
      if (countDynamic(aFold) <= 1 && countDynamic(bFold) <= 1) {
        Value a3d = reshapeTo(A, aElem, aFold, loc, rewriter);
        Value b3d = reshapeTo(B, bElem, bFold, loc, rewriter);
        return std::make_pair(a3d, b3d);
      }
      // Fall through to the generic path.
    }

    // The result's leading dims are the numpy-broadcasted batch shape.
    ArrayRef<int64_t> batchShape = resultShape.drop_back(2);

    Value a3d =
        broadcastAndFold(op, A, aType, batchShape, M, Ka, rewriter, loc);
    if (!a3d)
      return failure();
    Value b3d =
        broadcastAndFold(op, B, bType, batchShape, Kb, N, rewriter, loc);
    if (!b3d)
      return failure();
    return std::make_pair(a3d, b3d);
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
