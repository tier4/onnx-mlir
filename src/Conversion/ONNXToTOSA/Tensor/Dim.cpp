/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Dim.cpp - Dim Op ---------------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Dim operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXDimOpLoweringToTOSA : public OpConversionPattern<ONNXDimOp> {
public:
  using OpConversionPattern<ONNXDimOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Value data = adaptor.getData();
    auto inputType = mlir::dyn_cast<RankedTensorType>(data.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");

    // The verifier guarantees axis is within [0, rank).
    int64_t axis = op.getAxis();

    // The dimension must be known at compile time to become a constant.
    if (inputType.isDynamicDim(axis))
      return rewriter.notifyMatchFailure(op, "dynamic dimension not supported");

    Value dimConst = tosaBuilder.getConst(
        llvm::ArrayRef<int64_t>{inputType.getDimSize(axis)}, {1});
    rewriter.replaceOp(op, dimConst);
    return success();
  }
};

} // namespace

void populateLoweringONNXDimOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXDimOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
