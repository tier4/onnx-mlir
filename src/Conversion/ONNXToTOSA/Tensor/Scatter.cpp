/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Scatter.cpp - Scatter Ops ---------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX ScatterElements and ScatterND operators to TOSA
// dialect.
//
// Both are expressed in terms of tosa.scatter, the inverse of tosa.gather,
// whose semantics are a batched scatter over a 3-D tensor:
//   values_in : tensor<N x K x C>
//   indices   : tensor<N x W>      (i32)
//   input     : tensor<N x W x C>
//   output    : tensor<N x K x C>
// where:
//   output[n, indices[n, w], :] = input[n, w, :]
// and every other position is copied from values_in.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {

// Cast \p value to a signless 32-bit integer tensor, which is the index type
// required by tosa.scatter. Returns \p value unchanged if it is already i32.
Value castToI32(PatternRewriter &rewriter, Location loc, Value value) {
  auto type = mlir::cast<RankedTensorType>(value.getType());
  if (type.getElementType().isInteger(32))
    return value;
  return tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc,
      RankedTensorType::get(type.getShape(), rewriter.getI32Type()), value);
}

int64_t product(llvm::ArrayRef<int64_t> dims) {
  return std::accumulate(
      dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<>());
}

class ONNXScatterElementsLoweringToTOSA
    : public OpConversionPattern<ONNXScatterElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXScatterElementsOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXScatterElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    if (adaptor.getReduction() != "none")
      return rewriter.notifyMatchFailure(
          op, "only reduction 'none' supported.");

    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    Value updates = adaptor.getUpdates();

    auto dataType = mlir::dyn_cast<RankedTensorType>(data.getType());
    auto indicesType = mlir::dyn_cast<RankedTensorType>(indices.getType());
    auto resultType = mlir::dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    if (!dataType || !indicesType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types.");
    if (!dataType.hasStaticShape() || !indicesType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes.");

    llvm::ArrayRef<int64_t> dataShape = dataType.getShape();
    llvm::ArrayRef<int64_t> indicesShape = indicesType.getShape();
    int64_t rank = dataType.getRank();
    Type elementType = dataType.getElementType();

    int64_t axis = adaptor.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range.");

    // Flatten the whole scatter onto a single batch (N = 1) over the row-major
    // linearization of data:
    //   K = total number of data elements
    //   C = 1
    //   W = total number of index entries
    int64_t K = product(dataShape);
    int64_t W = product(indicesShape);

    // Row-major strides of data.
    llvm::SmallVector<int64_t, 4> strides(rank);
    int64_t stride = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
      strides[d] = stride;
      stride *= dataShape[d];
    }

    // For ScatterElements the target coordinate of an update equals its own
    // position in the index tensor, except along `axis` where it is the index
    // value. The non-axis contribution to the flattened offset is therefore a
    // compile-time constant. Precompute it in row-major order over
    // indicesShape.
    llvm::SmallVector<int32_t> base(W);
    llvm::SmallVector<int64_t, 4> coord(rank, 0);
    for (int64_t lin = 0; lin < W; ++lin) {
      int64_t offset = 0;
      for (int64_t d = 0; d < rank; ++d)
        if (d != axis)
          offset += coord[d] * strides[d];
      base[lin] = static_cast<int32_t>(offset);
      for (int64_t d = rank - 1; d >= 0; --d) {
        if (++coord[d] < indicesShape[d])
          break;
        coord[d] = 0;
      }
    }

    // values_in: collapse data to [1, K, 1].
    Value values = tosaBuilder.reshape(data, {1, K, 1});

    // Normalize negative indices (range [-dataShape[axis], dataShape[axis]-1])
    // along the scattered axis.
    Value idx = castToI32(rewriter, loc, indices);
    llvm::SmallVector<int64_t, 4> oneShape(rank, 1);
    Value zero = tosaBuilder.getConst(ArrayRef<int32_t>{0}, oneShape);
    Value axisSizeConst = tosaBuilder.getConst(
        ArrayRef<int32_t>{static_cast<int32_t>(dataShape[axis])}, oneShape);
    Value isNegative = tosa::CreateOpAndInfer<mlir::tosa::GreaterOp>(rewriter,
        loc, RankedTensorType::get(indicesShape, rewriter.getI1Type()), zero,
        idx);
    Value shifted = tosaBuilder.binaryOp<mlir::tosa::AddOp>(idx, axisSizeConst);
    idx = tosa::CreateOpAndInfer<mlir::tosa::SelectOp>(rewriter, loc,
        RankedTensorType::get(indicesShape, rewriter.getI32Type()), isNegative,
        shifted, idx);

    // flat[i] = idx[i] * stride[axis] + base[i].
    Value axisStrideConst = tosaBuilder.getConst(
        ArrayRef<int32_t>{static_cast<int32_t>(strides[axis])}, oneShape);
    Value weighted = tosaBuilder.mul(idx, axisStrideConst);
    Value baseConst =
        tosaBuilder.getConst(llvm::ArrayRef<int32_t>(base), indicesShape);
    Value flat = tosaBuilder.binaryOp<mlir::tosa::AddOp>(weighted, baseConst);
    flat = tosaBuilder.reshape(flat, {1, W});

    // input: collapse updates to [1, W, 1].
    Value updatesInput = tosaBuilder.reshape(updates, {1, W, 1});

    // tosa.scatter: [1, K, 1] x [1, W] x [1, W, 1] -> [1, K, 1].
    Value scattered = tosa::CreateOpAndInfer<mlir::tosa::ScatterOp>(rewriter,
        loc, RankedTensorType::get({1, K, 1}, elementType), values, flat,
        updatesInput);

    // Reshape back to the data (== output) shape.
    Value result = tosaBuilder.reshape(scattered, resultType.getShape());

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ONNXScatterNDLoweringToTOSA
    : public OpConversionPattern<ONNXScatterNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXScatterNDOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXScatterNDOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    if (adaptor.getReduction() != "none")
      return rewriter.notifyMatchFailure(
          op, "only reduction 'none' supported.");

    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    Value updates = adaptor.getUpdates();

    auto dataType = mlir::dyn_cast<RankedTensorType>(data.getType());
    auto indicesType = mlir::dyn_cast<RankedTensorType>(indices.getType());
    auto resultType = mlir::dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    if (!dataType || !indicesType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types.");
    if (!dataType.hasStaticShape() || !indicesType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes.");

    llvm::ArrayRef<int64_t> dataShape = dataType.getShape();
    llvm::ArrayRef<int64_t> indicesShape = indicesType.getShape();
    int64_t rank = dataType.getRank();
    int64_t indicesRank = indicesType.getRank();
    Type elementType = dataType.getElementType();

    // ScatterND has no batch dimensions; the last dimension of indices selects
    // which leading dimensions of data are indexed.
    int64_t numIndexed = indicesShape[indicesRank - 1];
    if (numIndexed > rank)
      return rewriter.notifyMatchFailure(op, "indices[-1] out of range.");

    // Map onto a single-batch tosa.scatter (N = 1):
    //   W = number of index tuples
    //   K = flattened size of the indexed dimensions
    //   C = number of elements in each scattered slice
    int64_t W = product(indicesShape.take_front(indicesRank - 1));
    int64_t K = product(dataShape.take_front(numIndexed));
    int64_t C = product(dataShape.drop_front(numIndexed));

    // values_in: collapse data to [1, K, C].
    Value values = tosaBuilder.reshape(data, {1, K, C});

    // indices: reshape to [1, W, ND], cast to i32 and fix up negative indices.
    Value idx = castToI32(rewriter, loc, indices);
    idx = tosaBuilder.reshape(idx, {1, W, numIndexed});

    llvm::SmallVector<int64_t, 3> idxShape = {1, W, numIndexed};
    llvm::SmallVector<int32_t, 4> dimSizes, coeffs;
    for (int64_t j = 0; j < numIndexed; ++j)
      dimSizes.push_back(static_cast<int32_t>(dataShape[j]));
    // coeff[j] = product of the indexed dimensions to the right of j, so that
    // the dot product with an index tuple yields its flattened (row-major)
    // offset into the [K] index space.
    int64_t stride = 1;
    coeffs.resize(numIndexed);
    for (int64_t j = numIndexed - 1; j >= 0; --j) {
      coeffs[j] = static_cast<int32_t>(stride);
      stride *= dataShape[j];
    }

    Value zero = tosaBuilder.getConst(ArrayRef<int32_t>{0}, {1, 1, 1});
    Value dimSizesConst = tosaBuilder.getConst(
        llvm::ArrayRef<int32_t>(dimSizes), {1, 1, numIndexed});
    Value isNegative = tosa::CreateOpAndInfer<mlir::tosa::GreaterOp>(rewriter,
        loc, RankedTensorType::get(idxShape, rewriter.getI1Type()), zero, idx);
    Value shifted = tosaBuilder.binaryOp<mlir::tosa::AddOp>(idx, dimSizesConst);
    idx = tosa::CreateOpAndInfer<mlir::tosa::SelectOp>(rewriter, loc,
        RankedTensorType::get(idxShape, rewriter.getI32Type()), isNegative,
        shifted, idx);

    // Flatten the ND index tuples to scalar offsets: reduce_sum(idx * coeff).
    Value coeffConst = tosaBuilder.getConst(
        llvm::ArrayRef<int32_t>(coeffs), {1, 1, numIndexed});
    Value weighted = tosaBuilder.mul(idx, coeffConst);
    Value flat = tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(rewriter, loc,
        RankedTensorType::get({1, W, 1}, rewriter.getI32Type()), weighted,
        rewriter.getI32IntegerAttr(2));
    flat = tosaBuilder.reshape(flat, {1, W});

    // input: collapse updates to [1, W, C].
    Value updatesInput = tosaBuilder.reshape(updates, {1, W, C});

    // tosa.scatter: [1, K, C] x [1, W] x [1, W, C] -> [1, K, C].
    Value scattered = tosa::CreateOpAndInfer<mlir::tosa::ScatterOp>(rewriter,
        loc, RankedTensorType::get({1, K, C}, elementType), values, flat,
        updatesInput);

    // Reshape back to the data (== output) shape.
    Value result = tosaBuilder.reshape(scattered, resultType.getShape());

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXScatterOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns
      .insert<ONNXScatterElementsLoweringToTOSA, ONNXScatterNDLoweringToTOSA>(
          typeConverter, ctx);
}

} // namespace onnx_mlir
