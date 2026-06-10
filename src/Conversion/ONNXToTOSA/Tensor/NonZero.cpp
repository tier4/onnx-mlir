/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NonZero.cpp - NonZero Op ----------------------------===//
//
// =============================================================================
//
// This file rewrites the ONNX NonZero compress/compute/scatter idiom into
// static-shape operations so that the graph can be fully lowered to TOSA.
//
// The idiom, as emitted by e.g. torch.onnx.export for `x[mask] = f(y[mask])`,
// looks like:
//
//   %nz1   = NonZero(%mask : tensor<Kxi1>) : tensor<1x?xi64>
//   %idx1  = Transpose(%nz1) {perm = [1, 0]} : tensor<?x1xi64>
//   %rows  = GatherND(%data : tensor<KxR...>, %idx1) : tensor<?xR...>
//   ...    = row-independent computation on %rows ...
//   %x     = ... : tensor<?xCxf32>
//   %m2    = Expand(Unsqueeze(%mask), [K, C]) : tensor<KxCxi1>
//   %nz2   = NonZero(%m2) : tensor<2x?xi64>
//   %idx2  = Transpose(%nz2) {perm = [1, 0]} : tensor<?x2xi64>
//   %flat  = Reshape(%x, [-1]) : tensor<?xf32>
//   %n     = Dim(%nz2) {axis = 1} : tensor<1xi64>
//   %upd   = Slice(%flat, 0, %n, 0, 1) : tensor<?xf32>
//   %out   = ScatterND(%init : tensor<KxCxf32>, %idx2, %upd)
//
// NonZero has a data-dependent result shape, which cannot be expressed in
// TOSA. Since the computation between the gather and the scatter is applied
// row by row, gathering only the selected rows is an optimization, not a
// semantic requirement. The idiom is therefore rewritten to compute on all
// K rows and select the result at the end:
//
//   %rows = %data                       (gather becomes the identity)
//   %out  = Where(%m2, Reshape(%x, [K, C]), %init)
//
// This is exact for the selected rows and yields %init for the masked-out
// rows, matching the original semantics. The rewrite assumes the computation
// between gather and scatter is row-independent; this holds for the
// MLP-style encoders this pattern targets but is not verified here.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool hasStaticShapeOf(Type type, ArrayRef<int64_t> shape) {
  auto rankedType = mlir::dyn_cast<RankedTensorType>(type);
  return rankedType && rankedType.hasStaticShape() &&
         rankedType.getShape() == shape;
}

bool isTransposePerm(ONNXTransposeOp op, ArrayRef<int64_t> perm) {
  auto permAttr = op.getPermAttr();
  if (!permAttr || permAttr.size() != perm.size())
    return false;
  for (size_t i = 0; i < perm.size(); ++i)
    if (ArrayAttrIntVal(permAttr, i) != perm[i])
      return false;
  return true;
}

bool isScalarConstInt64(Value value, int64_t expected) {
  ONNXConstantOp constOp = getONNXConstantOp(value);
  if (!constOp)
    return false;
  auto type = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() != 1 ||
      !type.getElementType().isInteger(64))
    return false;
  return getScalarValue<int64_t>(constOp) == expected;
}

class RewriteNonZeroCompressScatter : public OpRewritePattern<ONNXScatterNDOp> {
public:
  using OpRewritePattern<ONNXScatterNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXScatterNDOp scatterOp, PatternRewriter &rewriter) const override {
    if (scatterOp.getReduction() != "none")
      return rewriter.notifyMatchFailure(scatterOp, "reduction must be none");

    // The scattered-into tensor must have a static KxC shape.
    auto initType =
        mlir::dyn_cast<RankedTensorType>(scatterOp.getData().getType());
    if (!initType || !initType.hasStaticShape() || initType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          scatterOp, "data must have a static rank-2 shape");
    int64_t K = initType.getDimSize(0);
    int64_t C = initType.getDimSize(1);

    // indices == Transpose(NonZero(%m2), [1, 0]) with %m2 : tensor<KxCxi1>.
    auto transposeOp = scatterOp.getIndices().getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp || !isTransposePerm(transposeOp, {1, 0}))
      return rewriter.notifyMatchFailure(
          scatterOp, "indices is not Transpose(*, [1, 0])");
    auto nonZero2 = transposeOp.getData().getDefiningOp<ONNXNonZeroOp>();
    if (!nonZero2)
      return rewriter.notifyMatchFailure(
          scatterOp, "indices is not driven by NonZero");
    Value mask2d = nonZero2.getX();
    if (!hasStaticShapeOf(mask2d.getType(), {K, C}))
      return rewriter.notifyMatchFailure(
          scatterOp, "NonZero input does not match the data shape");

    // %m2 == Expand(Unsqueeze(%mask)) with %mask : tensor<Kxi1>.
    auto expandOp = mask2d.getDefiningOp<ONNXExpandOp>();
    if (!expandOp)
      return rewriter.notifyMatchFailure(scatterOp, "mask is not an Expand");
    auto unsqueezeOp = expandOp.getInput().getDefiningOp<ONNXUnsqueezeOp>();
    if (!unsqueezeOp || !hasStaticShapeOf(unsqueezeOp.getType(), {K, 1}))
      return rewriter.notifyMatchFailure(
          scatterOp, "mask is not expanded from a Kx1 Unsqueeze");
    Value mask = unsqueezeOp.getData();
    if (!hasStaticShapeOf(mask.getType(), {K}))
      return rewriter.notifyMatchFailure(scatterOp, "mask is not rank 1");

    // updates == Slice(Reshape(%x, [-1]), 0, Dim(%nz2, 1), 0, 1).
    auto sliceOp = scatterOp.getUpdates().getDefiningOp<ONNXSliceOp>();
    if (!sliceOp)
      return rewriter.notifyMatchFailure(scatterOp, "updates is not a Slice");
    auto dimOp = sliceOp.getEnds().getDefiningOp<ONNXDimOp>();
    if (!dimOp || dimOp.getData() != nonZero2.getResult() ||
        dimOp.getAxis() != 1)
      return rewriter.notifyMatchFailure(
          scatterOp, "slice end is not the NonZero count");
    if (!isScalarConstInt64(sliceOp.getStarts(), 0) ||
        !isScalarConstInt64(sliceOp.getAxes(), 0) ||
        !isScalarConstInt64(sliceOp.getSteps(), 1))
      return rewriter.notifyMatchFailure(
          scatterOp, "slice is not a length-n prefix");
    auto flattenOp = sliceOp.getData().getDefiningOp<ONNXReshapeOp>();
    if (!flattenOp)
      return rewriter.notifyMatchFailure(
          scatterOp, "sliced value is not a flattening Reshape");
    auto flatType = mlir::dyn_cast<RankedTensorType>(flattenOp.getType());
    if (!flatType || flatType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          scatterOp, "sliced value is not rank 1");
    Value x = flattenOp.getData();
    auto xType = mlir::dyn_cast<RankedTensorType>(x.getType());
    if (!xType || xType.getRank() != 2 || xType.getDimSize(1) != C)
      return rewriter.notifyMatchFailure(
          scatterOp, "scattered value is not a ?xC tensor");

    // Collect the row gathers driven by NonZero(%mask). Bail out if the
    // dynamically-sized NonZero indices escape into anything else, since the
    // rewrite changes the number of gathered rows from n to K.
    SmallVector<ONNXGatherNDOp> gathers;
    for (Operation *maskUser : mask.getUsers()) {
      auto nonZero1 = dyn_cast<ONNXNonZeroOp>(maskUser);
      if (!nonZero1)
        continue;
      for (Operation *nzUser : nonZero1->getUsers()) {
        auto trOp = dyn_cast<ONNXTransposeOp>(nzUser);
        if (!trOp || !isTransposePerm(trOp, {1, 0}))
          return rewriter.notifyMatchFailure(
              scatterOp, "NonZero(mask) has a non-Transpose user");
        for (Operation *trUser : trOp->getUsers()) {
          auto gatherOp = dyn_cast<ONNXGatherNDOp>(trUser);
          if (!gatherOp || gatherOp.getIndices() != trOp.getResult() ||
              gatherOp.getBatchDims() != 0)
            return rewriter.notifyMatchFailure(
                scatterOp, "NonZero(mask) indices have a non-GatherND user");
          auto dataType =
              mlir::dyn_cast<RankedTensorType>(gatherOp.getData().getType());
          if (!dataType || !dataType.hasStaticShape() ||
              dataType.getRank() < 1 || dataType.getDimSize(0) != K)
            return rewriter.notifyMatchFailure(
                scatterOp, "gathered data does not have K static rows");
          gathers.push_back(gatherOp);
        }
      }
    }
    if (gathers.empty())
      return rewriter.notifyMatchFailure(
          scatterOp, "no matching row gather found");

    // Compute on all K rows: the gathers become the identity. The consumers
    // keep their dynamically-shaped result types, which now hold exactly K
    // rows at runtime.
    for (ONNXGatherNDOp gatherOp : gathers)
      rewriter.replaceOp(gatherOp, gatherOp.getData());

    // out = Where(%m2, Reshape(%x, [K, C]), %init).
    OnnxBuilder createONNX(rewriter, scatterOp.getLoc());
    Value shapeConst = createONNX.constantInt64({K, C});
    Value reshaped = createONNX.reshape(
        RankedTensorType::get({K, C}, xType.getElementType()), x, shapeConst);
    Value result =
        createONNX.where(initType, mask2d, reshaped, scatterOp.getData());
    rewriter.replaceOp(scatterOp, result);
    return success();
  }
};

} // namespace

void populateRewriteONNXNonZeroCompressScatterPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<RewriteNonZeroCompressScatter>(ctx);
}

} // namespace onnx_mlir
