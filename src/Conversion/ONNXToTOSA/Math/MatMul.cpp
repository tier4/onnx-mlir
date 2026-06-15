/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- MatMul.cpp - Lowering MatMul Op --------------------===//
//
// Copyright 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to TOSA dialect.
//
// Only fully static shapes are supported: any dynamic dimension on an input
// makes the pattern bail out via notifyMatchFailure. This keeps the lowering
// simple, as every batch/row/column size is known at compile time.
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

    if (!aType.hasStaticShape() || !bType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "TOSA MatMul lowering only supports static shapes");

    if (aType.getRank() < 2 || bType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "TOSA MatMul lowering requires rank >= 2 inputs");

    auto resultType = mlir::dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    // Step 1: normalize both inputs to 3-D operands with matching batch dims.
    auto normalized = normalizeInputsTo3D(op, A, B, aType, bType, rewriter, loc);
    if (failed(normalized))
      return failure();
    auto [a3d, b3d] = *normalized;

    // Step 2: tosa.matmul on the 3-D operands.
    Value mm = tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, resultType.getElementType()), a3d,
        b3d)
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

  /// Return the product of \p shape[start..end). All dims must be static.
  static int64_t dimProduct(
      ArrayRef<int64_t> shape, int64_t start, int64_t end) {
    int64_t product = 1;
    for (int64_t i = start; i < end; ++i)
      product *= shape[i];
    return product;
  }

  /// Tile a 3-D value along its batch axis so that axis becomes \p targetBatch.
  static Value tileBatch3D(Value v, Type elemType, int64_t targetBatch,
      Location loc, ConversionPatternRewriter &rewriter) {
    Value multiples =
        mlir::tosa::getTosaConstShape(rewriter, loc, {targetBatch, 1LL, 1LL});
    return tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, elemType), v, multiples)
        .getResult();
  }

  /// Canonicalize one MatMul operand to a 3-D tensor [batch, R, C] without
  /// consulting the other operand.  Rank-2 gets a unit batch prepended,
  /// rank-3 is returned as-is, rank>3 has its leading dims folded into a
  /// single batch dim.
  static Value canonicalizeToBatch3D(Value x, RankedTensorType xType,
      ConversionPatternRewriter &rewriter, Location loc) {
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
    int64_t batch = dimProduct(shape, 0, rank - 2);
    return reshapeTo3D(x, elem, {batch, R, C}, loc, rewriter);
  }

  /// Given two 3-D operands, make their batch dims agree.  Equal batches
  /// pass through; a batch of 1 on one side is tiled to match the other.
  /// Mismatched non-1 batches are an unsupported broadcast.
  FailureOr<std::pair<Value, Value>> reconcileBatchDims(ONNXMatMulOp op,
      Value a3d, Value b3d, ConversionPatternRewriter &rewriter,
      Location loc) const {
    auto aType = mlir::cast<RankedTensorType>(a3d.getType());
    auto bType = mlir::cast<RankedTensorType>(b3d.getType());
    int64_t batchA = aType.getShape()[0];
    int64_t batchB = bType.getShape()[0];

    if (batchA == batchB)
      return std::make_pair(a3d, b3d);
    if (batchA == 1)
      return std::make_pair(
          tileBatch3D(a3d, aType.getElementType(), batchB, loc, rewriter), b3d);
    if (batchB == 1)
      return std::make_pair(
          a3d, tileBatch3D(b3d, bType.getElementType(), batchA, loc, rewriter));

    return rewriter.notifyMatchFailure(
        op, "MatMul batch dims are not broadcast-compatible for TOSA lowering");
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

    // Fast path: B is a 2-D weight.  Fold all of A's non-K dims into one
    // flat row dim so both operands stay at batch = 1 and we avoid tiling
    // the weight.
    if (bRank == 2 && aRank > 2) {
      int64_t flatM = dimProduct(aShape, 0, aRank - 1);
      int64_t K = aShape[aRank - 1], N = bShape[1];
      Value a3d = reshapeTo3D(A, aType.getElementType(), {1, flatM, K}, loc,
          rewriter);
      Value b3d =
          reshapeTo3D(B, bType.getElementType(), {1, K, N}, loc, rewriter);
      return std::make_pair(a3d, b3d);
    }

    // Generic path: canonicalize each operand to 3-D independently, then
    // reconcile batch dims.
    Value a3d = canonicalizeToBatch3D(A, aType, rewriter, loc);
    Value b3d = canonicalizeToBatch3D(B, bType, rewriter, loc);
    return reconcileBatchDims(op, a3d, b3d, rewriter, loc);
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
