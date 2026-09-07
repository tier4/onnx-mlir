/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Gather Ops -----------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX Gather and GatherND operators to TOSA dialect.
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
// required by tosa.gather. Returns \p value unchanged if it is already i32.
Value castToI32(PatternRewriter &rewriter, Location loc, Value value) {
  auto type = mlir::cast<RankedTensorType>(value.getType());
  if (type.getElementType().isInteger(32))
    return value;
  return tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc,
      RankedTensorType::get(type.getShape(), rewriter.getI32Type()), value);
}

class ONNXGatherLoweringToTOSA : public OpConversionPattern<ONNXGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXGatherOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();

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

    int64_t axis = adaptor.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range.");

    // Map the gather onto tosa.gather (N = 1 as ONNX Gather has no batch dims):
    //   K = size of the gathered axis
    //   C = number of elements in each gathered slice
    //   W = number of indices
    int64_t K = dataShape[axis];
    int64_t C = 1;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        C *= dataShape[i];
    int64_t W = std::accumulate(indicesShape.begin(), indicesShape.end(),
        static_cast<int64_t>(1), std::multiplies<int64_t>());

    // Fast path: a single COMPILE-TIME index selects one plane of `data`
    // along `axis` -- that is a slice, not a gather. Lowering it to
    // tosa.slice keeps the consumer reading the original buffer through a
    // static offset; the tosa.gather form below materializes a copy of the
    // selected plane and is opaque to downstream view folding. (Observed as
    // QKV splits and K/V selects costing whole dispatches per layer.)
    if (resultType.hasStaticShape() && W == 1) {
      ElementsAttr indicesAttr =
          getElementAttributeFromONNXValue(op.getIndices());
      if (indicesAttr && indicesAttr.getNumElements() == 1) {
        int64_t index =
            (*indicesAttr.getValues<APInt>().begin()).getSExtValue();
        if (index < 0)
          index += K;
        if (index < 0 || index >= K)
          return rewriter.notifyMatchFailure(op, "constant index range.");
        llvm::SmallVector<int64_t, 4> starts(rank, 0);
        llvm::SmallVector<int64_t, 4> sizes(dataShape.begin(), dataShape.end());
        starts[axis] = index;
        sizes[axis] = 1;
        Value sliced = tosaBuilder.slice(data, sizes, starts);
        rewriter.replaceOp(
            op, tosaBuilder.reshape(sliced, resultType.getShape()));
        return success();
      }
    }

    // Move the gathered axis to the front, then collapse to [1, K, C].
    Value values = data;
    if (axis != 0) {
      llvm::SmallVector<int32_t, 4> perm;
      perm.push_back(static_cast<int32_t>(axis));
      for (int64_t i = 0; i < rank; ++i)
        if (i != axis)
          perm.push_back(static_cast<int32_t>(i));
      values = tosaBuilder.transpose(values, perm);
    }
    values = tosaBuilder.reshape(values, {1, K, C});

    // Reshape indices to [1, W], cast to i32 and fix up negative indices.
    Value idx = castToI32(rewriter, loc, indices);
    idx = tosaBuilder.reshape(idx, {1, W});
    idx = normalizeNegativeIndices(rewriter, loc, tosaBuilder, idx, {1, W}, K);

    // tosa.gather: [1, K, C] x [1, W] -> [1, W, C].
    Value gathered = tosa::CreateOpAndInfer<mlir::tosa::GatherOp>(rewriter, loc,
        RankedTensorType::get({1, W, C}, elementType), values, idx);

    // Reshape to [indices..., left..., right...] where left/right are the data
    // dimensions before/after the gathered axis.
    llvm::SmallVector<int64_t, 4> interShape(
        indicesShape.begin(), indicesShape.end());
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        interShape.push_back(dataShape[i]);
    Value result = tosaBuilder.reshape(gathered, interShape);

    // Transpose to the ONNX layout [left..., indices..., right...].
    int64_t numLeft = axis;
    int64_t numRight = rank - axis - 1;
    if (numLeft != 0 && indicesRank != 0) {
      llvm::SmallVector<int32_t, 4> perm;
      for (int64_t i = 0; i < numLeft; ++i)
        perm.push_back(static_cast<int32_t>(indicesRank + i));
      for (int64_t i = 0; i < indicesRank; ++i)
        perm.push_back(static_cast<int32_t>(i));
      for (int64_t i = 0; i < numRight; ++i)
        perm.push_back(static_cast<int32_t>(indicesRank + numLeft + i));
      result = tosaBuilder.transpose(result, perm);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  // Replace each negative index i (in [-K, K-1]) by i + K so that all indices
  // are non-negative, as required by tosa.gather. \p shape is the (i32) shape
  // of \p idx.
  static Value normalizeNegativeIndices(PatternRewriter &rewriter, Location loc,
      TosaBuilder &tosaBuilder, Value idx, llvm::ArrayRef<int64_t> shape,
      int64_t axisSize) {
    Type i32Type = rewriter.getI32Type();
    llvm::SmallVector<int64_t, 4> oneShape(shape.size(), 1);
    Value zero = tosaBuilder.getConst(ArrayRef<int32_t>{0}, oneShape);
    Value axisSizeConst = tosaBuilder.getConst(
        ArrayRef<int32_t>{static_cast<int32_t>(axisSize)}, oneShape);

    Value isNegative = tosa::CreateOpAndInfer<mlir::tosa::GreaterOp>(rewriter,
        loc, RankedTensorType::get(shape, rewriter.getI1Type()), zero, idx);
    Value shifted = tosaBuilder.binaryOp<mlir::tosa::AddOp>(idx, axisSizeConst);
    return tosa::CreateOpAndInfer<mlir::tosa::SelectOp>(rewriter, loc,
        RankedTensorType::get(shape, i32Type), isNegative, shifted, idx);
  }
};

class ONNXGatherNDLoweringToTOSA : public OpConversionPattern<ONNXGatherNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXGatherNDOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGatherNDOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();

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

    int64_t b = adaptor.getBatchDims();
    if (b < 0 || b >= rank || b >= indicesRank)
      return rewriter.notifyMatchFailure(op, "batch_dims out of range.");

    // ND = number of indexed dimensions (last dimension of indices).
    int64_t numIndexed = indicesShape[indicesRank - 1];
    if (b + numIndexed > rank)
      return rewriter.notifyMatchFailure(op, "indices[-1] out of range.");

    auto product = [](llvm::ArrayRef<int64_t> dims) {
      return std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1),
          std::multiplies<int64_t>());
    };

    // Map the gather onto tosa.gather:
    //   N = number of batches
    //   W = number of index tuples per batch
    //   K = flattened size of the indexed dimensions
    //   C = number of elements in each gathered slice
    int64_t N = product(dataShape.take_front(b));
    int64_t W = product(indicesShape.slice(b, indicesRank - 1 - b));
    int64_t K = product(dataShape.slice(b, numIndexed));
    int64_t C = product(dataShape.drop_front(b + numIndexed));

    // values: collapse data to [N, K, C] (dimensions are already in order).
    Value values = tosaBuilder.reshape(data, {N, K, C});

    // indices: reshape to [N, W, ND], cast to i32 and fix up negative indices.
    Value idx = castToI32(rewriter, loc, indices);
    idx = tosaBuilder.reshape(idx, {N, W, numIndexed});

    llvm::SmallVector<int64_t, 3> idxShape = {N, W, numIndexed};
    llvm::SmallVector<int32_t, 4> dimSizes, coeffs;
    for (int64_t j = 0; j < numIndexed; ++j)
      dimSizes.push_back(static_cast<int32_t>(dataShape[b + j]));
    // coeff[j] = product of the indexed dimensions to the right of j, so that
    // the dot product with an index tuple yields its flattened (row-major)
    // offset into the [K] index space.
    int64_t stride = 1;
    coeffs.resize(numIndexed);
    for (int64_t j = numIndexed - 1; j >= 0; --j) {
      coeffs[j] = static_cast<int32_t>(stride);
      stride *= dataShape[b + j];
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
        RankedTensorType::get({N, W, 1}, rewriter.getI32Type()), weighted,
        rewriter.getI32IntegerAttr(2));
    flat = tosaBuilder.reshape(flat, {N, W});

    // tosa.gather: [N, K, C] x [N, W] -> [N, W, C].
    Value gathered = tosa::CreateOpAndInfer<mlir::tosa::GatherOp>(rewriter, loc,
        RankedTensorType::get({N, W, C}, elementType), values, flat);

    // Reshape to the ONNX output shape.
    Value result = tosaBuilder.reshape(gathered, resultType.getShape());

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGatherLoweringToTOSA, ONNXGatherNDLoweringToTOSA>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
