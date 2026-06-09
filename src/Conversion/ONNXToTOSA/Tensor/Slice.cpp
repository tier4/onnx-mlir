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

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXSliceOpLoweringToTOSA : public OpConversionPattern<ONNXSliceOp> {
public:
  using OpConversionPattern<ONNXSliceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Value data = adaptor.getData();
    if (!hasStaticShape(data.getType()))
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    IndexExprBuilderForTosa createTosaIE(rewriter, op.getLoc());
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

    if (llvm::any_of(steps, [](int64_t step) { return step != 1; }))
      return rewriter.notifyMatchFailure(op, "only step == 1 is supported");

    llvm::SmallVector<int64_t, 4> sizes;
    for (auto [start, end] : llvm::zip(starts, ends))
      sizes.push_back(end - start);

    Value sliceOp = tosaBuilder.slice(data, sizes, starts);
    rewriter.replaceOp(op, {sliceOp});
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
