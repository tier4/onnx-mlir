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

  /// Tile a 3-D value along its batch axis so that axis becomes \p targetBatch.
  static Value tileBatch3D(Value v, Type elemType, int64_t targetBatch,
      Location loc, ConversionPatternRewriter &rewriter) {
    Value multiples = mlir::tosa::getTosaConstShape(
        rewriter, loc, {targetBatch, 1LL, 1LL});
    return tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, elemType), v, multiples)
        .getResult();
  }

  /// Canonicalize one MatMul operand to a 3-D tensor [batch, R, C] without
  /// consulting the other operand.  Rank-2 gets a unit batch prepended,
  /// rank-3 is returned as-is, rank>3 has its leading dims folded (they
  /// must all be static).
  FailureOr<Value> canonicalizeToBatch3D(ONNXMatMulOp op, Value x,
      RankedTensorType xType, StringRef operandName,
      ConversionPatternRewriter &rewriter, Location loc) const {
    int64_t rank = xType.getRank();
    auto shape = xType.getShape();
    Type elem = xType.getElementType();
    int64_t R = shape[rank - 2];
    int64_t C = shape[rank - 1];

    if (rank == 3)
      return x;
    if (rank == 2)
      return reshapeTo3D(x, elem, {1, R, C}, loc, rewriter);

    // rank > 3: fold leading dims into a single batch dim.
    auto batch = staticDimProduct(shape, 0, rank - 2);
    if (failed(batch)) {
      rewriter.notifyMatchFailure(op,
          Twine("dynamic leading dims on operand ") + operandName +
              " (rank > 3) not yet supported");
      return failure();
    }
    return reshapeTo3D(x, elem, {*batch, R, C}, loc, rewriter);
  }

  /// Given two 3-D operands, make their batch dims agree.  Equal batches
  /// pass through; a static batch of 1 on one side is tiled to match the
  /// other.  Anything else (mismatched non-1 batches, or batch=1 facing a
  /// dynamic batch we cannot tile to) is an unsupported broadcast.
  FailureOr<std::pair<Value, Value>> reconcileBatchDims(ONNXMatMulOp op,
      Value a3d, Value b3d, ConversionPatternRewriter &rewriter,
      Location loc) const {
    auto aType = mlir::cast<RankedTensorType>(a3d.getType());
    auto bType = mlir::cast<RankedTensorType>(b3d.getType());
    int64_t batchA = aType.getShape()[0];
    int64_t batchB = bType.getShape()[0];
    bool dynA = ShapedType::isDynamic(batchA);
    bool dynB = ShapedType::isDynamic(batchB);

    if (!dynA && !dynB && batchA == batchB)
      return std::make_pair(a3d, b3d);

    if (batchA == 1 && !dynB)
      return std::make_pair(
          tileBatch3D(a3d, aType.getElementType(), batchB, loc, rewriter),
          b3d);
    if (batchB == 1 && !dynA)
      return std::make_pair(a3d,
          tileBatch3D(b3d, bType.getElementType(), batchA, loc, rewriter));

    rewriter.notifyMatchFailure(op,
        "MatMul batch dims are not broadcast-compatible for TOSA lowering");
    return failure();
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

    // Fast path: B is a 2-D weight.  Fold all of A's non-K dims into one
    // flat row dim so both operands stay at batch = 1 and we avoid tiling
    // the weight.  Requires A's non-K dims to all be static.
    if (bRank == 2 && aRank > 2) {
      auto flatM = staticDimProduct(aShape, 0, aRank - 1);
      if (succeeded(flatM)) {
        int64_t K = aShape[aRank - 1], N = bShape[1];
        Value a3d = reshapeTo3D(A, aElem, {1, *flatM, K}, loc, rewriter);
        Value b3d = reshapeTo3D(B, bElem, {1, K, N}, loc, rewriter);
        return std::make_pair(a3d, b3d);
      }
      // Fall through to the generic path (which will tile B instead).
    }

    // Generic path: canonicalize each operand to 3-D independently, then
    // reconcile batch dims.
    auto a3d = canonicalizeToBatch3D(op, A, aType, "A", rewriter, loc);
    if (failed(a3d))
      return failure();
    auto b3d = canonicalizeToBatch3D(op, B, bType, "B", rewriter, loc);
    if (failed(b3d))
      return failure();

    return reconcileBatchDims(op, *a3d, *b3d, rewriter, loc);
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
