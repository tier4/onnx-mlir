/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Expand.cpp - Expand Op ------------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Expand operator to TOSA dialect.
//
// ONNX Expand broadcasts the input to the output shape following numpy
// broadcasting rules (right-aligned dimensions). Since the output shape is
// static, the broadcast is realized by reshaping the input up to the output
// rank (prepending size-1 dimensions) and replicating it with tosa.tile.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXExpandLoweringToTOSA : public OpConversionPattern<ONNXExpandOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXExpandOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXExpandOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getInput();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto resultType = mlir::dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types.");
    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes.");

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = resultType.getShape();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = resultType.getRank();
    if (outputRank < inputRank)
      return rewriter.notifyMatchFailure(op, "output rank below input rank.");

    // Right-align the input shape to the output rank by prepending 1s.
    llvm::SmallVector<int64_t, 4> alignedShape(outputRank, 1);
    for (int64_t i = 0; i < inputRank; ++i)
      alignedShape[outputRank - inputRank + i] = inputShape[i];

    // multiples[d] = outputShape[d] / alignedShape[d]; each aligned dimension
    // either matches the output or is 1 (and gets replicated).
    llvm::SmallVector<int64_t, 4> multiples(outputRank, 1);
    bool needsTile = false;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (alignedShape[d] == outputShape[d])
        continue;
      if (alignedShape[d] != 1)
        return rewriter.notifyMatchFailure(op, "incompatible broadcast dim.");
      multiples[d] = outputShape[d];
      needsTile = true;
    }

    // Reshape the input up to the output rank when necessary.
    Value value = input;
    if (outputRank != inputRank)
      value = tosaBuilder.reshape(value, alignedShape);

    // An all-ones tile is the identity; the reshaped value is already correct.
    if (!needsTile) {
      rewriter.replaceOp(op, value);
      return success();
    }

    Value multiplesVal =
        mlir::tosa::getTosaConstShape(rewriter, loc, multiples);
    Value result = tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(
            llvm::SmallVector<int64_t, 4>(outputRank, ShapedType::kDynamic),
            resultType.getElementType()),
        value, multiplesVal);

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXExpandOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXExpandLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
