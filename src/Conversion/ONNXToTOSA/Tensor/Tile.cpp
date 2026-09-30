/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Tile.cpp - Tile Op ----------------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Tile operator to TOSA dialect.
//
// ONNX Tile replicates the input `repeats[d]` times along each dimension d,
// which is exactly tosa.tile. The multiples are taken from the static input
// and output shapes (output = input * repeats), so the repeats operand does
// not have to be a constant itself -- only the shapes must be static.
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

class ONNXTileLoweringToTOSA : public OpConversionPattern<ONNXTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXTileOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    Value input = adaptor.getInput();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto resultType = mlir::dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types.");
    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes.");
    if (inputType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(op, "rank mismatch.");

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = resultType.getShape();
    const int64_t rank = inputType.getRank();

    llvm::SmallVector<int64_t, 4> multiples(rank, 1);
    bool needsTile = false;
    for (int64_t d = 0; d < rank; ++d) {
      if (inputShape[d] == 0 || outputShape[d] % inputShape[d] != 0)
        return rewriter.notifyMatchFailure(
            op, "output dim is not a multiple of the input dim.");
      multiples[d] = outputShape[d] / inputShape[d];
      needsTile |= multiples[d] != 1;
    }

    // An all-ones tile is the identity.
    if (!needsTile) {
      rewriter.replaceOp(op, input);
      return success();
    }

    Value multiplesVal =
        mlir::tosa::getTosaConstShape(rewriter, loc, multiples);
    Value result = tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(
            llvm::SmallVector<int64_t, 4>(rank, ShapedType::kDynamic),
            resultType.getElementType()),
        input, multiplesVal);

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXTileOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXTileLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
