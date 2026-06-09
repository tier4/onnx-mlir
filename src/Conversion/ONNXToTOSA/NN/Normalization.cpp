/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Normalization.cpp - Normalization Ops ------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX Normalization operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXLayerNormalizationLoweringToTOSA
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXLayerNormalizationOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value X = adaptor.getX();
    Value scale = adaptor.getScale();
    Value B = adaptor.getB();

    auto inputType = mlir::dyn_cast<RankedTensorType>(X.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input type not a ranked tensor.");

    auto outputType = mlir::dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getY().getType()));
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "output type not a ranked tensor.");

    Type elementType = inputType.getElementType();
    int64_t rank = inputType.getRank();

    // The normalized axes are [axis, ..., rank - 1].
    int64_t axis = adaptor.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range.");

    // TOSA needs static shapes on the reduced axes to compute the mean divisor.
    llvm::SmallVector<int64_t, 4> axes;
    for (int64_t a = axis; a < rank; ++a) {
      if (inputType.isDynamicDim(a))
        return rewriter.notifyMatchFailure(
            op, "only static normalized dimensions are supported.");
      axes.push_back(a);
    }

    // Mean = ReduceMean(X) over the normalized axes (keepdims).
    Value mean =
        reduceMeanKeepDims(rewriter, op, X, inputType, axes, elementType);
    if (!mean)
      return rewriter.notifyMatchFailure(op, "could not lower mean reduction.");

    // D = X - Mean.
    Value d = tosaBuilder.binaryOp<mlir::tosa::SubOp>(X, mean);
    // DD = D * D.
    Value dd = tosaBuilder.mul(d, d);
    // Var = ReduceMean(DD) over the normalized axes (keepdims).
    Value var =
        reduceMeanKeepDims(rewriter, op, dd, inputType, axes, elementType);
    if (!var)
      return rewriter.notifyMatchFailure(
          op, "could not lower variance reduction.");

    // VarEps = Var + epsilon.
    Value epsilon = tosaBuilder.getSplattedConst(
        adaptor.getEpsilon().convertToFloat(), /*shape=*/{}, elementType);
    Value varEps = tosaBuilder.binaryOp<mlir::tosa::AddOp>(var, epsilon);

    // InvStdDev = rsqrt(VarEps).
    auto invStdDevType = mlir::cast<ShapedType>(varEps.getType());
    Value invStdDev = tosa::CreateOpAndInfer<mlir::tosa::RsqrtOp>(rewriter, loc,
        RankedTensorType::get(
            llvm::SmallVector<int64_t, 4>(
                invStdDevType.getRank(), ShapedType::kDynamic),
            elementType),
        varEps);

    // Normalized = D * InvStdDev.
    Value normalized = tosaBuilder.mul(d, invStdDev);
    // NormalizedScaled = Normalized * Scale.
    Value y = tosaBuilder.mul(normalized, scale);
    // Y = NormalizedScaled + B (B is optional).
    if (!isNoneValue(B))
      y = tosaBuilder.binaryOp<mlir::tosa::AddOp>(y, B);

    // Build the (possibly none) optional outputs Mean and InvStdDev.
    auto castIfNeeded = [&](Value value, Value origResult) -> Value {
      if (isNoneValue(origResult))
        return Value();
      auto resultType = mlir::dyn_cast_or_null<RankedTensorType>(
          getTypeConverter()->convertType(origResult.getType()));
      if (!resultType || resultType.getElementType() == elementType)
        return value;
      // The spec allows Mean/InvStdDev to use a different (stash) float type.
      return tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
          rewriter, loc, resultType, value);
    };

    Value meanOut = castIfNeeded(mean, op.getMean());
    Value invStdDevOut = castIfNeeded(invStdDev, op.getInvStdDev());

    rewriter.replaceOp(op, {y, meanOut, invStdDevOut});
    return success();
  }

private:
  // Compute ReduceMean of \p input over \p axes, keeping the reduced dimensions
  // (their extent becomes 1). Lowered as reduce_sum followed by a
  // multiplication by 1 / numberOfReducedElements, mirroring the ReduceMean
  // lowering.
  static Value reduceMeanKeepDims(PatternRewriter &rewriter, Operation *op,
      Value input, RankedTensorType inputType, llvm::ArrayRef<int64_t> axes,
      Type elementType) {
    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    DenseElementsAttr axesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(axes.size())}, rewriter.getI64Type()),
        axes);

    // Build the keepdims output shape and the number of reduced elements.
    llvm::SmallVector<int64_t, 4> reducedShape(inputType.getShape());
    int64_t numReducedElements = 1;
    for (int64_t axis : axes) {
      numReducedElements *= inputType.getShape()[axis];
      reducedShape[axis] = 1;
    }
    RankedTensorType outputType =
        RankedTensorType::get(reducedShape, elementType);

    std::optional<Value> reducedSum =
        tosa::convertReduceOpCommon<mlir::tosa::ReduceSumOp>(rewriter, op,
            outputType, input, axesAttr, /*keepDims=*/true, elementType);
    if (!reducedSum.has_value())
      return nullptr;

    double divScale = 1.0 / static_cast<double>(numReducedElements);
    Value divConst = tosaBuilder.getSplattedConst(
        static_cast<float>(divScale), /*shape=*/{}, elementType);
    return tosaBuilder.mul(reducedSum.value(), divConst);
  }
};

} // namespace

void populateLoweringONNXLayerNormalizationOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLayerNormalizationLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
