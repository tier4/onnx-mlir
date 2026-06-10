/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- CumSum.cpp - CumSum Op ------------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX CumSum operator to TOSA dialect.
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

static std::optional<int64_t> tryGetConstScalarInt(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  DenseElementsAttr attr;
  if (auto onnxConst = dyn_cast<ONNXConstantOp>(defOp)) {
    if (onnxConst.getValue().has_value())
      attr = dyn_cast<DenseElementsAttr>(onnxConst.getValueAttr());
  } else if (auto tosaConst = dyn_cast<mlir::tosa::ConstOp>(defOp)) {
    attr = dyn_cast<DenseElementsAttr>(tosaConst.getValuesAttr());
  }
  if (!attr || attr.getNumElements() != 1)
    return std::nullopt;
  auto it = attr.getValues<APInt>().begin();
  return (*it).getSExtValue();
}

class ONNXCumSumOpLoweringToTOSA : public OpConversionPattern<ONNXCumSumOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXCumSumOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXCumSumOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getX();
    int64_t exclusive = op.getExclusive();
    int64_t reverse = op.getReverse();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "input must be a ranked tensor with static shape");

    Type elementType = inputType.getElementType();
    if (!(elementType.isF32() || elementType.isF16() || elementType.isBF16()))
      return rewriter.notifyMatchFailure(
          op, "only f32/f16/bf16 element types supported");

    auto axisOpt = tryGetConstScalarInt(adaptor.getAxis());
    if (!axisOpt)
      return rewriter.notifyMatchFailure(
          op, "axis must be a compile-time constant scalar");
    int64_t rank = inputType.getRank();
    int64_t axis = axisOpt.value();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    int64_t K = inputType.getShape()[axis];
    SmallVector<int64_t> shape(
        inputType.getShape().begin(), inputType.getShape().end());

    // Direction: forward => shift right, reverse => shift left.
    bool shiftRight = (reverse == 0);

    // Hillis-Steele inclusive scan: log2(K) steps of shift+add.
    Value result = tosaBuilder.inclusiveScanAlongAxis(
        input, axis, shiftRight, shape, elementType);

    // For exclusive mode, shift one more step in the same direction so each
    // position holds the sum that strictly precedes (or follows) it.
    if (exclusive != 0 && K > 0) {
      result = tosaBuilder.shiftAlongAxis(result,
          /*offset=*/1, axis, shiftRight, shape, elementType);
    }

    rewriter.replaceOp(op, {result});
    return success();
  }
};

} // namespace

void populateLoweringONNXCumSumOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXCumSumOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
