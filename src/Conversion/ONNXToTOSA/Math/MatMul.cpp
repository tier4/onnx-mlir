/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- MatMul.cpp - Lowering MatMul Op --------------------===//
//
// Copyright 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to TOSA dialect.
//
// Only fully static shapes are supported: any dynamic dimension on an input
// makes the pattern bail out via notifyMatchFailure. This keeps the lowering
// simple, as every batch/row/column size is known at compile time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMatMulOpLoweringToTOSA : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value A(adaptor.getA());
    Value B(adaptor.getB());
    auto aType = mlir::dyn_cast<RankedTensorType>(A.getType());
    auto bType = mlir::dyn_cast<RankedTensorType>(B.getType());

    if (!aType || !bType)
      return rewriter.notifyMatchFailure(op, "inputs must be ranked tensors");

    if (!aType.hasStaticShape() || !bType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "TOSA MatMul lowering only supports static shapes");

    auto resultType = mlir::dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXMatMulOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t> aDims, bDims;
    IndexExpr::getLiteral(shapeHelper.aDims, aDims);
    IndexExpr::getLiteral(shapeHelper.bDims, bDims);
    int64_t rank = aDims.size(); // Padded rank, == bDims.size(), >= 2.

    // Collapse each padded operand to the 3-D form tosa.matmul expects:
    // [batch, rows, cols]. Padding only inserts size-1 dims, so a single
    // reshape from the original operand performs padding, 1-D promotion and
    // batch folding all at once.
    int64_t batchA = dimProduct(aDims, 0, rank - 2);
    int64_t batchB = dimProduct(bDims, 0, rank - 2);
    Value a3d = reshapeTo3D(A, aType.getElementType(),
        {batchA, aDims[rank - 2], aDims[rank - 1]}, loc, rewriter);
    Value b3d = reshapeTo3D(B, bType.getElementType(),
        {batchB, bDims[rank - 2], bDims[rank - 1]}, loc, rewriter);

    // Reconcile batch dims by tiling the broadcast (size-1) side.
    if (batchA != batchB) {
      if (batchA == 1)
        a3d = tileBatch3D(a3d, aType.getElementType(), batchB, loc, rewriter);
      else if (batchB == 1)
        b3d = tileBatch3D(b3d, bType.getElementType(), batchA, loc, rewriter);
      else
        return rewriter.notifyMatchFailure(op,
            "MatMul batch dims are not broadcast-compatible for TOSA lowering");
    }

    // tosa.matmul on the 3-D operands, then restore the ONNX result shape.
    Value mm = tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, resultType.getElementType()), a3d,
        b3d)
                   .getResult();
    Value result = tosaBuilder.reshape(mm, resultType.getShape());
    rewriter.replaceOp(op, {result});
    return success();
  }

private:
  static constexpr int64_t kDyn = ShapedType::kDynamic;
  static constexpr std::array<int64_t, 3> kDynamic3D = {kDyn, kDyn, kDyn};

  /// Reshape \p v to a 3-D tensor with the given target shape.
  static Value reshapeTo3D(Value v, Type elemType, ArrayRef<int64_t> shape3D,
      Location loc, ConversionPatternRewriter &rewriter) {
    return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, elemType), v,
        mlir::tosa::getTosaConstShape(rewriter, loc, shape3D))
        .getResult();
  }

  /// Tile a 3-D value along its batch axis so that axis becomes \p targetBatch.
  static Value tileBatch3D(Value v, Type elemType, int64_t targetBatch,
      Location loc, ConversionPatternRewriter &rewriter) {
    Value multiples =
        mlir::tosa::getTosaConstShape(rewriter, loc, {targetBatch, 1LL, 1LL});
    return tosa::CreateOpAndInfer<mlir::tosa::TileOp>(rewriter, loc,
        RankedTensorType::get(kDynamic3D, elemType), v, multiples)
        .getResult();
  }

  /// Product of \p shape[start..end). All dims are static literals.
  static int64_t dimProduct(
      ArrayRef<int64_t> shape, int64_t start, int64_t end) {
    int64_t product = 1;
    for (int64_t i = start; i < end; ++i)
      product *= shape[i];
    return product;
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
