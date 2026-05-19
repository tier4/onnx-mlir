/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Compress.cpp - Compress Op --------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX Compress operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXCompressOpLoweringToTOSA
    : public OpConversionPattern<ONNXCompressOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXCompressOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXCompressOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getInput();
    Value condition = adaptor.getCondition();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto condType = dyn_cast<RankedTensorType>(condition.getType());
    if (!inputType || !condType)
      return rewriter.notifyMatchFailure(
          op, "input and condition must be ranked tensors");
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "dynamic input shape not supported");
    if (condType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "condition must be rank 1");
    if (condType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op, "condition length must be statically known");

    Type elementType = inputType.getElementType();
    if (!(elementType.isF32() || elementType.isF16() || elementType.isBF16()))
      return rewriter.notifyMatchFailure(
          op, "only f32/f16/bf16 element types supported");

    int64_t Kc = condType.getShape()[0];

    // Normalize axis and flatten if no axis attribute.
    std::optional<int64_t> axisOpt = op.getAxis();
    bool noAxis = !axisOpt.has_value();
    int64_t axis = 0;
    Value workInput = input;
    SmallVector<int64_t> workShape;
    if (noAxis) {
      int64_t numElems = 1;
      for (int64_t d : inputType.getShape())
        numElems *= d;
      workShape = {numElems};
      workInput = tosaBuilder.reshape(input, workShape);
    } else {
      axis = axisOpt.value();
      int64_t rank = inputType.getRank();
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return rewriter.notifyMatchFailure(op, "axis out of range");
      workShape.assign(
          inputType.getShape().begin(), inputType.getShape().end());
    }

    int64_t K = workShape[axis];
    int64_t before = 1, after = 1;
    for (int64_t i = 0; i < axis; ++i)
      before *= workShape[i];
    for (int64_t i = axis + 1; i < (int64_t)workShape.size(); ++i)
      after *= workShape[i];

    // ONNX semantics for mismatched condition length:
    //   evaluate min(Kc, K) entries; positions beyond min are treated as false.
    if (std::min(Kc, K) <= 0)
      return rewriter.notifyMatchFailure(
          op, "empty effective condition length");

    Type f32Ty = rewriter.getF32Type();
    Type i1Ty = rewriter.getI1Type();
    Type i32Ty = rewriter.getI32Type();

    // 1) Cast condition (i1) -> f32 mask of length K. Slice/pad as needed
    //    (ONNX semantics evaluates min(Kc, K) entries; the rest are false).
    Value condF32Kc = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, RankedTensorType::get({Kc}, f32Ty), condition);

    Value condF32K = condF32Kc;
    if (Kc > K) {
      condF32K = tosaBuilder.slice(condF32Kc, {K}, {0});
    } else if (Kc < K) {
      auto padShape = mlir::tosa::getTosaConstShape(rewriter, loc, {0, K - Kc});
      Value zero = tosaBuilder.getSplattedConst(0.0f, {1});
      Type padTy = RankedTensorType::get({K}, f32Ty);
      condF32K = tosa::CreateOpAndInfer<mlir::tosa::PadOp>(
          rewriter, loc, padTy, condF32Kc, padShape, zero);
    }

    // 2) Inclusive cumsum of the f32 mask via Hillis-Steele scan, then
    //    pos = cumsum - 1, cast to i32.
    Value cumsumInc = tosa::inclusiveScanAlongAxis(rewriter, loc, tosaBuilder,
        condF32K, /*axis=*/0, /*forward=*/true, {K}, f32Ty);
    Value oneF = tosaBuilder.getSplattedConst(1.0f, {1});
    Value posF32 = tosaBuilder.binaryOp<mlir::tosa::SubOp>(cumsumInc, oneF);
    Value posI32 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, RankedTensorType::get({K}, i32Ty), posF32);

    // 3) Build the boolean mask (cond != 0) for `select`, and per-j unique
    //    sentinels (K + j) for entries where the mask is false. Using unique
    //    sentinels keeps tosa.scatter indices unique (TOSA spec requirement).
    Value zeroF = tosaBuilder.getSplattedConst(0.0f, {1});
    Type maskI1Ty = RankedTensorType::get({K}, i1Ty);
    Value eqZero = tosa::CreateOpAndInfer<mlir::tosa::EqualOp>(
        rewriter, loc, maskI1Ty, condF32K, zeroF);
    Value maskI1 = tosa::CreateOpAndInfer<mlir::tosa::LogicalNotOp>(
        rewriter, loc, maskI1Ty, eqZero);

    SmallVector<int32_t> sentinelData((size_t)K);
    for (int64_t j = 0; j < K; ++j)
      sentinelData[j] = (int32_t)(K + j);
    Value sentinel = tosaBuilder.getConst(ArrayRef<int32_t>(sentinelData), {K});

    Type selI32Ty = RankedTensorType::get({K}, i32Ty);
    Value maskedPos = tosa::CreateOpAndInfer<mlir::tosa::SelectOp>(
        rewriter, loc, selI32Ty, maskI1, posI32, sentinel);

    // 4) Scatter j into a [1, 2K, 1] zero-initialized buffer, then slice the
    //    first K slots: the inverse-permutation indices for the gather. The
    //    upper K slots absorb the false-entry sentinels and are discarded.
    SmallVector<int32_t> iotaI32Data((size_t)K);
    for (int64_t j = 0; j < K; ++j)
      iotaI32Data[j] = (int32_t)j;
    Value iotaI32 =
        tosaBuilder.getConst(ArrayRef<int32_t>(iotaI32Data), {1, K, 1});

    int64_t bufferLen = 2 * K;
    SmallVector<int32_t> zeroBufData((size_t)bufferLen, 0);
    Value scatterBuf =
        tosaBuilder.getConst(ArrayRef<int32_t>(zeroBufData), {1, bufferLen, 1});

    Value maskedPos2D = tosaBuilder.reshape(maskedPos, {1, K});
    Type scatterTy = RankedTensorType::get({1, bufferLen, 1}, i32Ty);
    Value scattered = tosa::CreateOpAndInfer<mlir::tosa::ScatterOp>(
        rewriter, loc, scatterTy, scatterBuf, maskedPos2D, iotaI32);

    Value indicesSlice = tosaBuilder.slice(scattered, {1, K, 1}, {0, 0, 0});
    Value indices = tosaBuilder.reshape(indicesSlice, {1, K});
    if (before > 1) {
      auto multiples =
          mlir::tosa::getTosaConstShape(rewriter, loc, {before, 1});
      Type tileTy = RankedTensorType::get({before, K}, i32Ty);
      indices = tosa::CreateOpAndInfer<mlir::tosa::TileOp>(
          rewriter, loc, tileTy, indices, multiples);
    }

    // 5) Reshape input to 3D [before, K, after] and gather along K.
    SmallVector<int64_t> shape3D = {before, K, after};
    Value data3D = tosaBuilder.reshape(workInput, shape3D);
    Type gatherTy = RankedTensorType::get({before, K, after}, elementType);
    Value out3D = tosa::CreateOpAndInfer<mlir::tosa::GatherOp>(
        rewriter, loc, gatherTy, data3D, indices);

    // 6) Post-mask: count = reduce_sum(mask), valid[i] = i < count, then
    //    multiply (broadcast over the inner dim) so padded rows become zero.
    Type reduceTy = RankedTensorType::get({1}, f32Ty);
    Value count = tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(
        rewriter, loc, reduceTy, condF32K, rewriter.getI32IntegerAttr(0));

    SmallVector<float> iotaFData((size_t)K);
    for (int64_t i = 0; i < K; ++i)
      iotaFData[i] = (float)i;
    Value iotaF = tosaBuilder.getConst(ArrayRef<float>(iotaFData), {K});

    Type validI1Ty = RankedTensorType::get({K}, i1Ty);
    Value validI1 = tosa::CreateOpAndInfer<mlir::tosa::GreaterOp>(
        rewriter, loc, validI1Ty, count, iotaF);
    Type validF32Ty = RankedTensorType::get({K}, f32Ty);
    Value validF32 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, validF32Ty, validI1);
    Value validElem = validF32;
    if (!elementType.isF32()) {
      validElem = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
          rewriter, loc, RankedTensorType::get({K}, elementType), validF32);
    }
    Value valid3D = tosaBuilder.reshape(validElem, {1, K, 1});
    out3D = tosaBuilder.mul(out3D, valid3D, 0);

    // 7) Reshape to ONNX-typed shape (compressed axis kept at K, padded).
    SmallVector<int64_t> finalShape;
    if (noAxis) {
      finalShape = {K};
    } else {
      finalShape.assign(
          inputType.getShape().begin(), inputType.getShape().end());
      finalShape[axis] = K;
    }
    Value result = tosaBuilder.reshape(out3D, finalShape);

    // 8) Bridge to the (possibly dynamic) ONNX-converted result type.
    Type expectedType =
        getTypeConverter()->convertType(op.getResult().getType());
    Value finalResult = result;
    if (expectedType && expectedType != result.getType()) {
      finalResult = tensor::CastOp::create(rewriter, loc, expectedType, result);
    }
    rewriter.replaceOp(op, {finalResult});
    return success();
  }
};

} // namespace

void populateLoweringONNXCompressOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXCompressOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
