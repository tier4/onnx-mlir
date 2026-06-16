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

// Truncate a floating point tensor toward zero (round-to-zero), matching the
// ONNX float-to-integer cast semantics:
//   trunc(x) = (x >= 0) ? floor(x) : ceil(x)
// The result is still a floating point tensor whose values are exact integers,
// so the subsequent tosa.cast to an integer type is exact regardless of its
// rounding mode.
Value truncateTowardZero(ConversionPatternRewriter &rewriter, Location loc,
    Value input, RankedTensorType inputType) {
  TosaBuilder tosaBuilder(rewriter, loc);

  Value floorVal = tosa::CreateOpAndInfer<mlir::tosa::FloorOp>(
      rewriter, loc, inputType, input);
  Value ceilVal = tosa::CreateOpAndInfer<mlir::tosa::CeilOp>(
      rewriter, loc, inputType, input);

  Value zero = tosaBuilder.getSplattedConst(
      0.0f, inputType.getShape(), inputType.getElementType());
  auto i1Type =
      RankedTensorType::get(inputType.getShape(), rewriter.getI1Type());
  Value isNonNegative = tosa::CreateOpAndInfer<mlir::tosa::GreaterEqualOp>(
      rewriter, loc, i1Type, input, zero);

  return tosaBuilder.select(isNonNegative, floorVal, ceilVal);
}

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

    Type inputElementType = inputType.getElementType();
    Type resultElementType = resultTensorType.getElementType();

    // A cast to the same element type is a no-op.
    if (inputElementType == resultElementType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // ONNX casts from floating point to integer truncate toward zero, whereas
    // tosa.cast rounds to nearest with ties to even. To preserve ONNX
    // semantics, explicitly truncate the floating point value toward zero
    // before casting it to an integer type.
    // Casts to bool (i1) are excluded: their ONNX semantics are "value != 0",
    // which truncation toward zero would not preserve.
    if (mlir::isa<FloatType>(inputElementType) &&
        resultElementType.isInteger() && !resultElementType.isInteger(1)) {
      input = truncateTowardZero(rewriter, op.getLoc(), input, inputType);
    }

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
