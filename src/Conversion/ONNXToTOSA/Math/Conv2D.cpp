/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Conv2D.cpp - Conv2D Op ------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX conv operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

/// tosa.conv2d does not support dividing the channels into groups.
/// When the onnx operator requires this, we have to create multiple ops
/// where the input, kernel and bias is a slice of the original inputs.
/// Afterwards we have to concat the results of these into a single tensor.
Value createConvInGroups(PatternRewriter &rewriter, Operation *op,
    TosaBuilder &tosaBuilder, Type &resultType,
    const llvm::ArrayRef<int64_t> weightShape, Value &newInput,
    Value &newWeight, Value &bias, const int64_t groups,
    DenseI64ArrayAttr &pads, DenseI64ArrayAttr &strides,
    DenseI64ArrayAttr &dilations, TypeAttr &accType) {
  // Set up constants outside of loop
  const int64_t sizeOfSliceInput = weightShape[1];
  const int64_t sizeOfSliceKernel = weightShape[0] / groups;
  auto newInputShape = mlir::cast<ShapedType>(newInput.getType()).getShape();

  llvm::SmallVector<int64_t, 4> inputSize = {
      newInputShape[0], newInputShape[1], newInputShape[2], sizeOfSliceInput};
  llvm::SmallVector<int64_t, 4> kernelSize = {
      sizeOfSliceKernel, weightShape[2], weightShape[3], weightShape[1]};
  llvm::SmallVector<Value> sliceValues;

  for (int64_t i = 0; i < groups; i++) {
    // Slice input
    Value newSliceInput =
        tosaBuilder.slice(newInput, inputSize, {0, 0, 0, i * sizeOfSliceInput});

    // Slice kernel
    Value newSliceWeight = tosaBuilder.slice(
        newWeight, kernelSize, {i * sizeOfSliceKernel, 0, 0, 0});

    // Slice bias
    Value newSliceBias =
        tosaBuilder.slice(bias, {sizeOfSliceKernel}, {i * sizeOfSliceKernel});

    // Create conv
    Type newConvOutputType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
        mlir::cast<ShapedType>(resultType).getElementType());
    Value tempConv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
        op->getLoc(), newConvOutputType, newSliceInput, newSliceWeight,
        newSliceBias, pads, strides, dilations, accType);
    // Add value to vector
    sliceValues.push_back(tempConv2D);
  }
  // Create concat op
  Type newConcatOutputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
      mlir::cast<ShapedType>(resultType).getElementType());
  Value conv2D = tosa::CreateOpAndInfer<mlir::tosa::ConcatOp>(
      rewriter, op->getLoc(), newConcatOutputType, sliceValues, 3);
  return conv2D;
}

/// ONNX rounds the convolution output size down, while tosa.conv2d requires
/// `input - 1 + padBegin + padEnd - (kernel - 1) * dilation` to be a multiple
/// of the stride. The remainder counts trailing padding that sits past the
/// last window and never contributes to a result element, so dropping it from
/// padEnd leaves both the output size and its values unchanged.
void trimUnusedTrailingPads(llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> weightShape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> dilations, llvm::MutableArrayRef<int64_t> pads) {
  const int64_t spatialRank = pads.size() / 2;
  for (int64_t i = 0; i < spatialRank; ++i) {
    const int64_t inputDim = inputShape[i + 2];
    const int64_t kernelDim = weightShape[i + 2];
    if (ShapedType::isDynamic(inputDim) || ShapedType::isDynamic(kernelDim))
      continue;
    int64_t &padEnd = pads[i + spatialRank];
    const int64_t span =
        inputDim - 1 + pads[i] + padEnd - (kernelDim - 1) * dilations[i];
    if (span <= 0)
      continue;
    const int64_t unused = span % strides[i];
    if (unused <= padEnd)
      padEnd -= unused;
  }
}

class ONNXConvOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXConvOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, ctx) {}

  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto convOp = mlir::dyn_cast<ONNXConvOp>(op);
    OpAdaptor adaptor(operands, convOp);

    TosaBuilder tosaBuilder(rewriter, loc);

    auto input = adaptor.getX();
    auto weights = adaptor.getW();
    auto bias = adaptor.getB();

    auto inputType = mlir::cast<TensorType>(input.getType());
    auto weightType = mlir::cast<ShapedType>(weights.getType());

    // Get shapehelper for autopad attributes
    IndexExprBuilderForTosa createTosaIE(rewriter, convOp->getLoc());
    ONNXConvOpShapeHelper shapeHelper(op, operands, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type resultType = convOp.getResult().getType();

    const int64_t rank = inputType.getRank();
    if (rank != 3 && rank != 4) {
      return rewriter.notifyMatchFailure(
          convOp, "only 1d and 2d convolutions are implemented");
    }
    const bool is1D = rank == 3;
    if (is1D && !(inputType.hasStaticShape() && weightType.hasStaticShape())) {
      return rewriter.notifyMatchFailure(
          convOp, "1d convolution needs static input and weight shapes");
    }

    if (!IndexExpr::isLiteral(shapeHelper.pads))
      return rewriter.notifyMatchFailure(op, "pads is not a literal.");
    llvm::SmallVector<int64_t, 4> pads;
    IndexExpr::getLiteral(shapeHelper.pads, pads);
    llvm::SmallVector<int64_t, 2> strides(shapeHelper.strides);
    llvm::SmallVector<int64_t, 2> dilations(shapeHelper.dilations);
    llvm::SmallVector<int64_t, 4> inputShape(inputType.getShape());
    llvm::SmallVector<int64_t, 4> weightShape(weightType.getShape());

    // tosa has no conv1d, so a [N,C,W] convolution runs as a 2d one over a
    // trailing width of 1. No stride, pad or dilation reaches that dim, so the
    // result is the 1d one with a trailing 1 that is reshaped away below.
    if (is1D) {
      inputShape.push_back(1);
      weightShape.push_back(1);
      input = tosaBuilder.reshape(input, inputShape);
      weights = tosaBuilder.reshape(weights, weightShape);
      pads = {pads[0], 0, pads[1], 0};
      strides.push_back(1);
      dilations.push_back(1);
    }

    trimUnusedTrailingPads(inputShape, weightShape, strides, dilations, pads);

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput = tosaBuilder.transpose(input, {0, 2, 3, 1});

    // Convert weights [OC,IC,KH,KW] -> [OC,KH,KW,IC]
    Value newWeight = tosaBuilder.transpose(weights, {0, 2, 3, 1});

    if (mlir::isa<NoneType>(bias.getType())) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      bias = mlir::tosa::ConstOp::create(
          rewriter, convOp->getLoc(), newBiasAttr.getType(), newBiasAttr);
    }

    DenseI64ArrayAttr newDilations = rewriter.getDenseI64ArrayAttr(dilations);
    DenseI64ArrayAttr newStrides = rewriter.getDenseI64ArrayAttr(strides);

    // reorder padding values
    DenseI64ArrayAttr newPads =
        rewriter.getDenseI64ArrayAttr({pads[0], pads[2], pads[1], pads[3]});

    Type convType =
        (resultType.isF16()) ? rewriter.getF16Type() : rewriter.getF32Type();
    TypeAttr accType = mlir::TypeAttr::get(convType);

    // Handle group parameter by creating multiple convs
    const int64_t group = adaptor.getGroup();
    Value conv2D = NULL;
    if (group == 1) {
      Type newConvOutputType = RankedTensorType::get(
          llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
          mlir::cast<ShapedType>(resultType).getElementType());

      conv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter,
          convOp->getLoc(), newConvOutputType, newInput, newWeight, bias,
          newPads, newStrides, newDilations, accType);
    } else {
      conv2D = createConvInGroups(rewriter, convOp, tosaBuilder, resultType,
          weightShape, newInput, newWeight, bias, group, newPads, newStrides,
          newDilations, accType);
    }

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput = tosaBuilder.transpose(conv2D, {0, 3, 1, 2});

    if (is1D) {
      llvm::SmallVector<int64_t, 3> outputShape(
          mlir::cast<ShapedType>(resultType).getShape());
      newOutput = tosaBuilder.reshape(newOutput, outputShape);
    }

    rewriter.replaceOp(convOp, {newOutput});
    return success();
  }
};
} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
