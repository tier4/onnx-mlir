/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- Cast.cpp - Cast Op ----------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX cast operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXCastOpLoweringToTOSA : public OpConversionPattern<ONNXCastOp> {
public:
  using OpConversionPattern<ONNXCastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return rewriter.notifyMatchFailure(op, "result is not a ranked tensor");

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");

    // A cast to the same element type is a no-op.
    if (inputType.getElementType() == resultTensorType.getElementType()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // Note: ONNX casts from floating point to integer truncate toward zero,
    // whereas tosa.cast rounds to nearest with ties to even. This mirrors the
    // behaviour of other TOSA frontends and is accepted as a known divergence.
    rewriter.replaceOpWithNewOp<mlir::tosa::CastOp>(op, resultType, input);
    return success();
  }
};

} // namespace

void populateLoweringONNXCastOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXCastOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
