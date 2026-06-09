/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Transpose Op ------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX transpose operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXTransposeOpLoweringToTOSA
    : public OpConversionPattern<ONNXTransposeOp> {
public:
  using OpConversionPattern<ONNXTransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Value data = adaptor.getData();
    auto dataType = mlir::dyn_cast<RankedTensorType>(data.getType());
    if (!dataType)
      return rewriter.notifyMatchFailure(op, "data is not a ranked tensor");

    int64_t rank = dataType.getRank();

    // Build the permutation vector. If `perm` is omitted, ONNX defaults to
    // reversing the dimensions, i.e. (rank-1, ..., 0).
    llvm::SmallVector<int32_t, 4> perm;
    if (std::optional<ArrayAttr> permAttr = op.getPerm()) {
      for (Attribute attr : *permAttr)
        perm.push_back(
            static_cast<int32_t>(mlir::cast<IntegerAttr>(attr).getInt()));
    } else {
      for (int64_t i = rank - 1; i >= 0; --i)
        perm.push_back(static_cast<int32_t>(i));
    }

    if (static_cast<int64_t>(perm.size()) != rank)
      return rewriter.notifyMatchFailure(
          op, "perm size does not match the rank of the input");

    Value transposedOp = tosaBuilder.transpose(data, perm);
    rewriter.replaceOp(op, {transposedOp});
    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
