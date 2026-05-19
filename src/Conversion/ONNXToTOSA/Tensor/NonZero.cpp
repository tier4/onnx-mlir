/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NonZero.cpp - NonZero Op ----------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX NonZero operator to TOSA dialect.
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

class ONNXNonZeroOpLoweringToTOSA : public OpConversionPattern<ONNXNonZeroOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXNonZeroOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXNonZeroOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getX();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "input must be a ranked tensor with static shape");
    int64_t rank = inputType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "scalar input not supported");

    int64_t N = 1;
    for (int64_t d : inputType.getShape())
      N *= d;
    if (N <= 0)
      return rewriter.notifyMatchFailure(op, "empty input not supported");

    Type elementType = inputType.getElementType();
    Type f32Ty = rewriter.getF32Type();
    Type i1Ty = rewriter.getI1Type();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();

    // 1) Flatten input to 1D and cast to f32 (needed for the mask + scan).
    Value flat = tosaBuilder.reshape(input, {N});
    Value xF32 = flat;
    if (!elementType.isF32()) {
      xF32 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
          rewriter, loc, RankedTensorType::get({N}, f32Ty), flat);
    }

    // 2) mask_f32 = (x != 0). Built via equal+logical_not -> cast.
    Value zeroF = tosaBuilder.getSplattedConst(0.0f, {1});
    Type maskI1Ty = RankedTensorType::get({N}, i1Ty);
    Value eqZero = tosa::CreateOpAndInfer<mlir::tosa::EqualOp>(
        rewriter, loc, maskI1Ty, xF32, zeroF);
    Value maskI1 = tosa::CreateOpAndInfer<mlir::tosa::LogicalNotOp>(
        rewriter, loc, maskI1Ty, eqZero);
    Type maskF32Ty = RankedTensorType::get({N}, f32Ty);
    Value maskF32 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, maskF32Ty, maskI1);

    // 3) Inclusive cumsum via Hillis-Steele scan, then pos = cumsum - 1.
    Value cumsumInc = tosa::inclusiveScanAlongAxis(rewriter, loc, tosaBuilder,
        maskF32, /*axis=*/0, /*forward=*/true, {N}, f32Ty);
    Value oneF = tosaBuilder.getSplattedConst(1.0f, {1});
    Value posF32 = tosaBuilder.binaryOp<mlir::tosa::SubOp>(cumsumInc, oneF);
    Value posI32 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, RankedTensorType::get({N}, i32Ty), posF32);

    // 4) Sentinel for zero entries (unique per j: N + j).
    SmallVector<int32_t> sentinelData((size_t)N);
    for (int64_t j = 0; j < N; ++j)
      sentinelData[j] = (int32_t)(N + j);
    Value sentinel = tosaBuilder.getConst(ArrayRef<int32_t>(sentinelData), {N});
    Value maskedPos = tosa::CreateOpAndInfer<mlir::tosa::SelectOp>(rewriter,
        loc, RankedTensorType::get({N}, i32Ty), maskI1, posI32, sentinel);

    // 5) Scatter j into a [1, 2N, 1] zero-initialized buffer, then slice the
    //    first N slots: the inverse-permutation indices for the gather.
    SmallVector<int32_t> iotaI32Data((size_t)N);
    for (int64_t j = 0; j < N; ++j)
      iotaI32Data[j] = (int32_t)j;
    Value iotaI32_3D =
        tosaBuilder.getConst(ArrayRef<int32_t>(iotaI32Data), {1, N, 1});

    int64_t bufferLen = 2 * N;
    SmallVector<int32_t> zeroBufData((size_t)bufferLen, 0);
    Value scatterBuf =
        tosaBuilder.getConst(ArrayRef<int32_t>(zeroBufData), {1, bufferLen, 1});

    Value maskedPos2D = tosaBuilder.reshape(maskedPos, {1, N});
    Value scattered = tosa::CreateOpAndInfer<mlir::tosa::ScatterOp>(rewriter,
        loc, RankedTensorType::get({1, bufferLen, 1}, i32Ty), scatterBuf,
        maskedPos2D, iotaI32_3D);
    Value invertedSlice = tosaBuilder.slice(scattered, {1, N, 1}, {0, 0, 0});
    Value inverted = tosaBuilder.reshape(invertedSlice, {1, N});

    // 6) Precompute the flat-to-multi-index table I[N, rank] as f32 and use
    //    tosa.gather to pick rows according to `inverted`. For unused slots
    //    `inverted` holds the buffer initial value 0, so the gather pulls
    //    I[0, :] = [0, ..., 0], naturally giving zero-valued padding.
    SmallVector<int64_t> strides(rank);
    strides[rank - 1] = 1;
    for (int64_t d = rank - 2; d >= 0; --d)
      strides[d] = strides[d + 1] * inputType.getShape()[d + 1];
    SmallVector<float> I_data((size_t)N * rank);
    for (int64_t j = 0; j < N; ++j) {
      for (int64_t d = 0; d < rank; ++d) {
        int64_t coord = (j / strides[d]) % inputType.getShape()[d];
        I_data[j * rank + d] = (float)coord;
      }
    }
    Value I_3D = tosaBuilder.getConst(ArrayRef<float>(I_data), {1, N, rank});

    Type gatheredTy = RankedTensorType::get({1, N, rank}, f32Ty);
    Value gathered = tosa::CreateOpAndInfer<mlir::tosa::GatherOp>(
        rewriter, loc, gatheredTy, I_3D, inverted);

    // 7) Reshape to [N, rank], transpose to [rank, N], cast to i64.
    Value gathered2D = tosaBuilder.reshape(gathered, {N, rank});
    SmallVector<int32_t> perm = {1, 0};
    Value outF32 = tosaBuilder.transpose(gathered2D, perm);
    Type outI64StaticTy = RankedTensorType::get({rank, N}, i64Ty);
    Value outI64 = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, outI64StaticTy, outF32);

    // 8) Bridge to the (possibly dynamic) ONNX-converted result type.
    Type expectedType =
        getTypeConverter()->convertType(op.getResult().getType());
    Value finalResult = outI64;
    if (expectedType && expectedType != outI64.getType()) {
      finalResult = tensor::CastOp::create(rewriter, loc, expectedType, outI64);
    }
    rewriter.replaceOp(op, {finalResult});
    return success();
  }
};

} // namespace

void populateLoweringONNXNonZeroOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXNonZeroOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
