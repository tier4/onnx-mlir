/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Slice.cpp - Slice Op -----------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX slice operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Create a rank-1, single element tensor holding zero, used as the padding
// value for tosa.pad. The padded elements are discarded afterwards, so the
// concrete value is irrelevant; only the element type must match.
Value createZeroScalar(PatternRewriter &rewriter, Location loc,
    TosaBuilder &tosaBuilder, Type et) {
  if (mlir::isa<FloatType>(et))
    return tosaBuilder.getSplattedConst(0.0, {1}, et);
  auto constType = RankedTensorType::get({1}, et);
  auto constAttr =
      DenseElementsAttr::get(constType, APInt(et.getIntOrFloatBitWidth(), 0));
  return mlir::tosa::ConstOp::create(rewriter, loc, constType, constAttr);
}

class ONNXSliceOpLoweringToTOSA : public OpConversionPattern<ONNXSliceOp> {
public:
  using OpConversionPattern<ONNXSliceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());
    Location loc = op.getLoc();

    Value data = adaptor.getData();
    auto inputType = mlir::dyn_cast<RankedTensorType>(data.getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    auto resultType = mlir::dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "result must have a static shape");

    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXSliceOpShapeHelper shapeHelper(
        op.getOperation(), op->getOperands(), &createTosaIE);
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(
          op, "could not compute slice parameters");

    if (!IndexExpr::isLiteral(shapeHelper.starts) ||
        !IndexExpr::isLiteral(shapeHelper.ends) ||
        !IndexExpr::isLiteral(shapeHelper.steps))
      return rewriter.notifyMatchFailure(
          op, "starts/ends/steps must be known at compile time");

    llvm::SmallVector<int64_t, 4> starts, ends, steps;
    IndexExpr::getLiteral(shapeHelper.starts, starts);
    IndexExpr::getLiteral(shapeHelper.ends, ends);
    IndexExpr::getLiteral(shapeHelper.steps, steps);

    int64_t rank = inputType.getRank();
    ArrayRef<int64_t> inShape = inputType.getShape();
    ArrayRef<int64_t> outShape = resultType.getShape();

    // Normalize each axis to a non-negative start and a positive step. Axes
    // with a negative step are reversed up front so that they can be handled
    // identically to positive strides afterwards.
    llvm::SmallVector<int64_t, 4> normStarts(rank);
    llvm::SmallVector<int64_t, 4> normSteps(rank);
    Value value = data;
    for (int64_t i = 0; i < rank; ++i) {
      if (steps[i] < 0) {
        value = tosa::CreateOpAndInfer<mlir::tosa::ReverseOp>(rewriter, loc,
            UnrankedTensorType::get(inputType.getElementType()), value,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(i)));
        normStarts[i] = inShape[i] - 1 - starts[i];
        normSteps[i] = -steps[i];
      } else {
        normStarts[i] = starts[i];
        normSteps[i] = steps[i];
      }
    }

    // Take the contiguous span [start, start + (k - 1) * step + 1) that covers
    // every selected element along each axis.
    llvm::SmallVector<int64_t, 4> sliceSizes(rank);
    for (int64_t i = 0; i < rank; ++i)
      sliceSizes[i] = (outShape[i] - 1) * normSteps[i] + 1;
    value = tosaBuilder.slice(value, sliceSizes, normStarts);

    bool hasStride =
        llvm::any_of(normSteps, [](int64_t step) { return step != 1; });
    if (!hasStride) {
      // Pure contiguous slice (all steps are 1 after normalization).
      rewriter.replaceOp(op, value);
      return success();
    }

    // Pad each strided axis so its length becomes a multiple of the step, then
    // reshape it into [k, step], keep index 0 of the step sub-dimension and
    // reshape back. This extracts every step-th element.
    llvm::SmallVector<int64_t, 8> paddingVec(2 * rank, 0);
    for (int64_t i = 0; i < rank; ++i)
      paddingVec[2 * i + 1] = normSteps[i] - 1; // high padding only
    Value padConst = createZeroScalar(
        rewriter, loc, tosaBuilder, inputType.getElementType());
    Value padding = mlir::tosa::getTosaConstShape(rewriter, loc, paddingVec);
    value = tosa::CreateOpAndInfer<mlir::tosa::PadOp>(rewriter, loc,
        UnrankedTensorType::get(inputType.getElementType()), value, padding,
        padConst);

    // Reshape: split each strided axis into [k, step], keep others as [k].
    llvm::SmallVector<int64_t, 8> splitShape;
    for (int64_t i = 0; i < rank; ++i) {
      splitShape.push_back(outShape[i]);
      if (normSteps[i] != 1)
        splitShape.push_back(normSteps[i]);
    }
    value = tosaBuilder.reshape(value, splitShape);

    // Slice out index 0 of every step sub-dimension.
    llvm::SmallVector<int64_t, 8> innerStarts(splitShape.size(), 0);
    llvm::SmallVector<int64_t, 8> innerSizes;
    for (int64_t i = 0; i < rank; ++i) {
      innerSizes.push_back(outShape[i]);
      if (normSteps[i] != 1)
        innerSizes.push_back(1);
    }
    value = tosaBuilder.slice(value, innerSizes, innerStarts);

    // Collapse the size-1 step sub-dimensions back to the output shape.
    value = tosaBuilder.reshape(value, outShape);

    rewriter.replaceOp(op, value);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
