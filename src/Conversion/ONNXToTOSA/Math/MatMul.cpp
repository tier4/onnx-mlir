/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MatMul.cpp - MatMul Op ------------------------------===//
//
// Copyright 2026
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to TOSA dialect.
//
// TOSA MatMul requires exactly 3-D inputs [batch, rows, cols]. Every
// supported case is therefore reduced to the same three-step pattern:
//
//   1. normalizeInputsTo3D  – reshape / tile inputs into 3-D tensors whose
//      batch dimensions agree.
//   2. tosa.matmul          – the single 3-D matmul.
//   3. tosa.reshape         – restore the original output shape.
//
// Supported rank combinations:
//   2D x 2D   – prepend batch = 1, strip it afterward.
//   3D x 3D   – pass through directly.
//   ND x ND   – fold static leading batch dims into one.
//   ND x 2D   – fold A's dims (except K) into one flat row dim.
//   2D x 3D   – tile A along B's static batch dim.
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

    Value A = adaptor.getA();
    Value B = adaptor.getB();
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

    // Step 1: normalize both inputs to 3-D.
    auto normalized =
        normalizeInputsTo3D(op, A, B, aType, bType, rewriter, loc);
    if (failed(normalized))
      return failure();
    auto [a3d, b3d] = *normalized;

    // Step 2: tosa.matmul on 3-D operands.
    Value mm =
        tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter, loc,
            RankedTensorType::get(kDynamic3D, resultType.getElementType()),
            a3d, b3d)
            .getResult();

    // Step 3: reshape back to the expected result shape.
    Value result = tosaBuilder.reshape(mm, resultType.getShape());
    rewriter.replaceOp(op, {result});
    return success();
  }

private:
  static constexpr int64_t kDyn = ShapedType::kDynamic;
  static constexpr std::array<int64_t, 3> kDynamic3D = {kDyn, kDyn, kDyn};

  /// Reshape \p v to a 3-D tensor with the given target shape.
  static Value reshapeTo3D(Value v, Type elemType, ArrayRef<int64_t> shape3D,
      Location loc, ConversionPatternRewriter &rewriter) {
    return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, loc,
               RankedTensorType::get(kDynamic3D, elemType), v,
               mlir::tosa::getTosaConstShape(rewriter, loc, shape3D))
        .getResult();
  }

  /// Return the product of \p shape[start..end). Fail if any dim is dynamic.
  static FailureOr<int64_t> staticDimProduct(
      ArrayRef<int64_t> shape, int64_t start, int64_t end) {
    int64_t product = 1;
    for (int64_t i = start; i < end; ++i) {
      if (ShapedType::isDynamic(shape[i]))
        return failure();
      product *= shape[i];
    }
    return product;
  }

  /// Transform both inputs into 3-D tensors with matching batch dimensions,
  /// ready for tosa.matmul.  Calls notifyMatchFailure on unsupported cases.
  FailureOr<std::pair<Value, Value>> normalizeInputsTo3D(ONNXMatMulOp op,
      Value A, Value B, RankedTensorType aType, RankedTensorType bType,
      ConversionPatternRewriter &rewriter, Location loc) const {
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    Type aElem = aType.getElementType();
    Type bElem = bType.getElementType();

    // 3D x 3D – already in the right shape.
    if (aRank == 3 && bRank == 3)
      return std::make_pair(A, B);

    // 2D x 2D – prepend a unit batch dim.
    if (aRank == 2 && bRank == 2)
      return std::make_pair(
          reshapeTo3D(A, aElem, {1, aShape[0], aShape[1]}, loc, rewriter),
          reshapeTo3D(B, bElem, {1, bShape[0], bShape[1]}, loc, rewriter));

    // Equal rank > 3 – fold leading batch dims into one.
    if (aRank == bRank && aRank > 3) {
      auto batch = staticDimProduct(aShape, 0, aRank - 2);
      if (failed(batch)) {
        rewriter.notifyMatchFailure(
            op, "dynamic batch dims for rank > 3 MatMul not yet supported");
        return failure();
      }
      int64_t M = aShape[aRank - 2], K = aShape[aRank - 1];
      int64_t N = bShape[bRank - 1];
      return std::make_pair(
          reshapeTo3D(A, aElem, {*batch, M, K}, loc, rewriter),
          reshapeTo3D(B, bElem, {*batch, K, N}, loc, rewriter));
    }

    // ND x 2D – fold all of A's dims except K into one flat row dim.
    //   A: [..., M, K] x B: [K, N] -> [1, flatM, K] x [1, K, N]
    if (bRank == 2) {
      auto flatM = staticDimProduct(aShape, 0, aRank - 1);
      if (failed(flatM)) {
        rewriter.notifyMatchFailure(op,
            "dynamic A dims for rank-mismatched MatMul (bRank==2) "
            "not yet supported");
        return failure();
      }
      int64_t K = aShape[aRank - 1], N = bShape[1];
      return std::make_pair(
          reshapeTo3D(A, aElem, {1, *flatM, K}, loc, rewriter),
          reshapeTo3D(B, bElem, {1, K, N}, loc, rewriter));
    }

    // 2D x 3D – tile A to match B's batch dimension.
    //   A: [M, K] x B: [batch, K, N] -> tile A to [batch, M, K]
    if (aRank == 2 && bRank == 3) {
      int64_t batch = bShape[0];
      if (ShapedType::isDynamic(batch)) {
        rewriter.notifyMatchFailure(op,
            "dynamic batch dim in B for MatMul (aRank==2, bRank==3) "
            "not yet supported");
        return failure();
      }
      int64_t M = aShape[0], K = aShape[1];
      Value a3d = reshapeTo3D(A, aElem, {1, M, K}, loc, rewriter);
      Value multiples =
          mlir::tosa::getTosaConstShape(rewriter, loc, {batch, 1LL, 1LL});
      a3d = tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
                RankedTensorType::get(kDynamic3D, aElem), a3d, multiples)
                .getResult();
      return std::make_pair(a3d, B);
    }

    rewriter.notifyMatchFailure(
        op, "unsupported rank combination for TOSA MatMul lowering");
    return failure();
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
