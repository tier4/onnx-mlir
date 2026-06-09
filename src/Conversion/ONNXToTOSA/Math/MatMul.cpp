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

    // Run the matmul shape helper to obtain rank-aligned per-axis dim info
    // for both operands.  The helper pads each operand's leading dims with
    // 1s up to a common rank and propagates literal values from one operand
    // to the other when shape inference can disambiguate them, sparing us
    // from doing the alignment by hand here.
    //
    // Use the analysis-mode IndexExprBuilder + dim-analysis mode so the
    // helper never tries to materialize runtime shape ops (which our target
    // dialect set wouldn't legalize) or rewrite operand types.
    IndexExprBuilderForAnalysis createIE(loc);
    ONNXMatMulOpShapeHelper shapeHelper(op, {}, &createIE);
    shapeHelper.setDimAnalysisMode();
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(op, "MatMul shape helper failed");
    SmallVector<int64_t> aPadded = toShape(shapeHelper.aDims);
    SmallVector<int64_t> bPadded = toShape(shapeHelper.bDims);

    // Step 1: normalize both inputs to 3-D, using the result type's
    // leading dims as the broadcast target.
    auto normalized = normalizeInputsTo3D(op, A, B, aType, bType, aPadded,
        bPadded, resultType.getShape(), rewriter, loc);
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

  /// Convert a shape-helper IndexExpr dim list to int64_t shape values:
  /// literal dims keep their value, runtime dims become kDynamic.
  static SmallVector<int64_t> toShape(ArrayRef<IndexExpr> dims) {
    SmallVector<int64_t> r;
    r.reserve(dims.size());
    for (const IndexExpr &d : dims)
      r.push_back(d.isLiteral() ? d.getLiteral() : kDyn);
    return r;
  }

  /// Broadcast \p x's leading (batch) dims to \p batchShape and fold them
  /// into one, producing a 3-D tensor [batchProduct, innerR, innerC].
  /// \p paddedShape is the operand's shape after rank-padding by the matmul
  /// shape helper (rank == batchShape.size() + 2; the trailing two entries
  /// are innerR and innerC).  Returns nullptr (with notifyMatchFailure
  /// already called) on unsupported shapes.
  Value broadcastAndFold(ONNXMatMulOp op, Value x, RankedTensorType xType,
      ArrayRef<int64_t> paddedShape, ArrayRef<int64_t> batchShape,
      ConversionPatternRewriter &rewriter, Location loc) const {
    int64_t batchRank = batchShape.size();
    int64_t paddedRank = paddedShape.size(); // = batchRank + 2
    int64_t innerR = paddedShape[batchRank];
    int64_t innerC = paddedShape[batchRank + 1];
    int64_t rank = xType.getRank();
    Type elem = xType.getElementType();
    auto origShape = xType.getShape();

    // Per-axis tile multiples for the batch portion.  TOSA tile takes
    // a constant multiples shape, so any required tile factor must be static.
    bool needTile = false;
    bool allBatchOnes = batchRank > 0;
    SmallVector<int64_t> multiples(paddedRank, 1);
    for (int64_t i = 0; i < batchRank; ++i) {
      int64_t a = paddedShape[i];
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
        (rank == paddedRank) ? x : reshapeTo(x, elem, paddedShape, loc, rewriter);
    Value tiled = tileWithMultiples(alignedV, elem, multiples, loc, rewriter);
    return reshapeTo(tiled, elem, folded3D, loc, rewriter);
  }

  /// Transform both inputs into 3-D tensors with matching batch dimensions,
  /// ready for tosa.matmul.  \p aPadded and \p bPadded come from the matmul
  /// shape helper (each has rank == max(aRank, bRank, 2), with leading dims
  /// padded by 1).  Calls notifyMatchFailure on unsupported cases.
  FailureOr<std::pair<Value, Value>> normalizeInputsTo3D(ONNXMatMulOp op,
      Value A, Value B, RankedTensorType aType, RankedTensorType bType,
      ArrayRef<int64_t> aPadded, ArrayRef<int64_t> bPadded,
      ArrayRef<int64_t> resultShape, ConversionPatternRewriter &rewriter,
      Location loc) const {
    Type aElem = aType.getElementType();
    Type bElem = bType.getElementType();
    int64_t paddedRank = aPadded.size();
    int64_t Ka = aPadded[paddedRank - 1];
    int64_t Kb = bPadded[paddedRank - 2];
    int64_t N = bPadded[paddedRank - 1];

    // Fast path: B is a 2-D weight.  Fold all of A's non-K dims into one
    // flat row dim so both operands stay at batch = 1 and we avoid tiling
    // the weight.  A's leading dims may include dynamic dims — they fold
    // into a single dynamic dim, encoded as -1 in the const_shape.  Each
    // reshape's target must have at most one dynamic dim.
    if (bType.getRank() == 2 && aType.getRank() > 2) {
      int64_t flatM = dimProduct(aPadded.drop_back(1));
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
        broadcastAndFold(op, A, aType, aPadded, batchShape, rewriter, loc);
    if (!a3d)
      return failure();
    Value b3d =
        broadcastAndFold(op, B, bType, bPadded, batchShape, rewriter, loc);
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
