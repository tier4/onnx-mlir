/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Softmax.cpp - Softmax Op ----------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Applies TosaReduceOp over the same axes that the softmax normalizes over.
// Before opset 13 that is `axis` and every dimension following it; from opset
// 13 it is `axis` alone. TOSA reduce ops keep the reduced dimensions at size 1,
// so the result broadcasts back against the input.
template <typename Softmax, typename TosaReduceOp>
Value computeReduce(PatternRewriter &rewriter, Operation *op,
    RankedTensorType resultType, const Value &input, int axis) = delete;

template <>
Value computeReduce<ONNXSoftmaxV11Op, mlir::tosa::ReduceSumOp>(
    PatternRewriter &rewriter, Operation *op, RankedTensorType resultType,
    const Value &input, int axis);

// Shared implementation for the pre-opset-13 multi-axis reduction.
template <typename TosaReduceOp>
Value computeReduceV11(PatternRewriter &rewriter, Operation *op,
    RankedTensorType resultType, const Value &input, int axis) {
  const int64_t inputRank = resultType.getRank();
  // Create shared outputType with dynamic shape. Infer method when creating
  // ops will insert a static shape if possible
  Type outputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(inputRank, ShapedType::kDynamic),
      resultType.getElementType());
  // Create first reduce with input from function operands
  Value reduced = tosa::CreateOpAndInfer<TosaReduceOp>(
      rewriter, op->getLoc(), outputType, input, rewriter.getI32IntegerAttr(axis));
  // Loop over all following dimensions with last reduce as input
  for (int i = axis + 1; i < inputRank; i++) {
    reduced = tosa::CreateOpAndInfer<TosaReduceOp>(rewriter, op->getLoc(),
        outputType, reduced, rewriter.getI32IntegerAttr(i));
  }
  return reduced;
}

template <>
Value computeReduce<ONNXSoftmaxV11Op, mlir::tosa::ReduceSumOp>(
    PatternRewriter &rewriter, Operation *op, RankedTensorType resultType,
    const Value &input, int axis) {
  return computeReduceV11<mlir::tosa::ReduceSumOp>(
      rewriter, op, resultType, input, axis);
}

template <>
Value computeReduce<ONNXSoftmaxV11Op, mlir::tosa::ReduceMaxOp>(
    PatternRewriter &rewriter, Operation *op, RankedTensorType resultType,
    const Value &input, int axis) {
  return computeReduceV11<mlir::tosa::ReduceMaxOp>(
      rewriter, op, resultType, input, axis);
}

// From opset 13, softmax uses axis as the reduce axis.
template <>
Value computeReduce<ONNXSoftmaxOp, mlir::tosa::ReduceSumOp>(
    PatternRewriter &rewriter, Operation *op, RankedTensorType resultType,
    const Value &input, int axis) {
  return tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(rewriter, op->getLoc(),
      resultType, input, rewriter.getI32IntegerAttr(axis));
}

template <>
Value computeReduce<ONNXSoftmaxOp, mlir::tosa::ReduceMaxOp>(
    PatternRewriter &rewriter, Operation *op, RankedTensorType resultType,
    const Value &input, int axis) {
  return tosa::CreateOpAndInfer<mlir::tosa::ReduceMaxOp>(rewriter, op->getLoc(),
      resultType, input, rewriter.getI32IntegerAttr(axis));
}

template <typename SoftmaxOp>
class ONNXSoftmaxLoweringToTOSA : public OpConversionPattern<SoftmaxOp> {
public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  using OpAdaptor = typename SoftmaxOp::Adaptor;
  LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getInput();
    // softmax = exp(logits) / reduce_sum(exp(logits), -1)
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());

    // Not a ranked tensor input/output
    if (!outputType || !inputType) {
      return rewriter.notifyMatchFailure(
          op, "input and result not ranked tensors");
    }

    // Get ONNX softmax axis
    int64_t axis = adaptor.getAxis();
    // Tosa only supports positive values
    int64_t inputRank = inputType.getRank();
    if (axis < 0) {
      axis += inputRank;
    }
    // The legalization below is based on convertSoftmaxOp in
    // tensorflow tosa/transforms/legalize_common.cc, with the
    // addition of handling for axis.

    // Floating-point lowering, in the numerically stable (max-subtracted) form:
    //
    // op0 = reduce_max(logits, axis)
    // op1 = exp(logits - op0)
    // op2 = reduce_sum(op1, -1)
    // op3 = reciprocal(op2)
    // op4 = mul(op1, op3)
    //
    // Subtracting the max is mathematically a no-op (it cancels between the
    // numerator and the denominator) but it bounds the exp() argument at 0, so
    // exp() can never overflow. Without it the raw logits reach the exponent
    // range directly: this model peaks at 82, which is finite in f32 but
    // saturates f16 (exp overflows past 11.09) and turns the whole graph into
    // NaN once activations are demoted to f16.
    RankedTensorType reduceType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(inputRank, ShapedType::kDynamic),
        outputType.getElementType());

    Value op0ReducemaxIn = computeReduce<SoftmaxOp, mlir::tosa::ReduceMaxOp>(
        rewriter, op, reduceType, input, axis);
    Value shiftedIn =
        tosaBuilder.binaryOp<mlir::tosa::SubOp>(input, op0ReducemaxIn);

    Value op1ExpIn = tosa::CreateOpAndInfer<mlir::tosa::ExpOp>(
        rewriter, loc, outputType, shiftedIn);

    Value op2ReducesumOp1 = computeReduce<SoftmaxOp, mlir::tosa::ReduceSumOp>(
        rewriter, op, reduceType, op1ExpIn, axis);

    Value op3ReciprocalOp2 = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
        rewriter, loc, op2ReducesumOp1.getType(), op2ReducesumOp1);

    Value mulOp = tosaBuilder.mul(op1ExpIn, op3ReciprocalOp2);
    rewriter.replaceOp(op, {mulOp});

    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxLoweringToTOSA<ONNXSoftmaxOp>,
      ONNXSoftmaxLoweringToTOSA<ONNXSoftmaxV11Op>>(typeConverter, ctx);
}

} // namespace onnx_mlir