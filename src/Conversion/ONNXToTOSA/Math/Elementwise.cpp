/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct TOSADialectOp<ONNXNegOp> {
  using Op = mlir::tosa::NegateOp;
};

namespace {

// Element-wise unary ops lowering to TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
class ONNXElementwiseUnaryOpLoweringToTOSA
    : public OpConversionPattern<ElementwiseUnaryOp> {
public:
  using OpConversionPattern<ElementwiseUnaryOp>::OpConversionPattern;
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  LogicalResult matchAndRewrite(ElementwiseUnaryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TOSAOp<ElementwiseUnaryOp>>(
        op, op.getType(), adaptor.getX());
    return success();
  }
};

template <typename ONNXOpT, typename TosaOpT, bool SwapOperands = false>
class ONNXBinaryElementwiseOpLoweringToTOSA
    : public OpConversionPattern<ONNXOpT> {
public:
  using OpConversionPattern<ONNXOpT>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpT::Adaptor;
  LogicalResult matchAndRewrite(ONNXOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value lhs = adaptor.getA();
    auto lhsType = mlir::dyn_cast<TensorType>(lhs.getType());

    Value rhs = adaptor.getB();
    auto rhsType = mlir::dyn_cast<TensorType>(rhs.getType());

    auto resultType = mlir::dyn_cast<TensorType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type resultElementType = resultType.getElementType();

    if (!resultElementType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "only int and float are supported");
    }

    if (TosaOpT::template hasTrait<
            mlir::OpTrait::ResultsBroadcastableShape>()) {

      IndexExprBuilderForTosa createTosaIE(rewriter, op->getLoc());
      ONNXBroadcastOpShapeHelper shapeHelper(op, {}, &createTosaIE);
      shapeHelper.computeShapeAndAssertOnFailure();

      if (shapeHelper.hasRankBroadcast()) {
        TosaBuilder tosaBuilder(rewriter, loc);
        llvm::SmallVector<Value, 4> newValues =
            tosaBuilder.equalizeRanks({lhs, rhs});
        lhs = newValues[0];
        rhs = newValues[1];
      }
    }

    if (SwapOperands)
      std::swap(lhs, rhs);

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

    return success();
  }
};

class ONNXSinOpLoweringToTOSA : public OpConversionPattern<ONNXSinOp> {
public:
  using OpConversionPattern<ONNXSinOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXSinOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::SinOp>(
        op, op.getType(), adaptor.getInput());
    return success();
  }
};

class ONNXCosOpLoweringToTOSA : public OpConversionPattern<ONNXCosOp> {
public:
  using OpConversionPattern<ONNXCosOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXCosOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXCosOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::tosa::CosOp>(
        op, op.getType(), adaptor.getInput());
    return success();
  }
};

class ONNXGeluOpLoweringToTOSA : public OpConversionPattern<ONNXGeluOp> {
public:
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXGeluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value x = adaptor.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(x.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
    auto elementType = mlir::dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "tosa.gelu lowering only supports float types");

    TosaBuilder tosaBuilder(rewriter, loc);
    StringRef approximate = adaptor.getApproximate();
    ArrayRef<int64_t> shape = inputType.getShape();
    Value half = tosaBuilder.getSplattedConst(0.5, shape, elementType);
    Value one = tosaBuilder.getSplattedConst(1.0, shape, elementType);

    Value inner;
    if (approximate == "none") {
      // y = 0.5 * x * (1 + erf(x / sqrt(2)))
      Value invSqrt2 = tosaBuilder.getSplattedConst(
          0.70710678118654752440, shape, elementType);
      Value scaled = tosaBuilder.mul(x, invSqrt2);
      inner = tosaBuilder.erf(scaled);
    } else if (approximate == "tanh") {
      // y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      Value coeff = tosaBuilder.getSplattedConst(0.044715, shape, elementType);
      Value sqrt2OverPi = tosaBuilder.getSplattedConst(
          0.79788456080286535588, shape, elementType);
      Value xSquared = tosaBuilder.mul(x, x);
      Value xCubed = tosaBuilder.mul(xSquared, x);
      Value coeffXCubed = tosaBuilder.mul(coeff, xCubed);
      Value sum = tosaBuilder.binaryOp<mlir::tosa::AddOp>(x, coeffXCubed);
      Value scaled = tosaBuilder.mul(sqrt2OverPi, sum);
      inner = tosaBuilder.tanh(scaled);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported 'approximate' attribute value");
    }

    Value addOne = tosaBuilder.binaryOp<mlir::tosa::AddOp>(inner, one);
    Value mulX = tosaBuilder.mul(x, addOne);
    Value result = tosaBuilder.mul(mulX, half);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower ONNXAtanOp to a composition of TOSA ops via range reduction
// followed by a 9th-order polynomial approximation.
//   atan(x) for x < 0      = -atan(-x)
//   atan(y) for y > 1      = pi/2 - atan(1/y)
//   atan(u) for u > sqrt(2)-1
//                          = pi/4 + atan((u-1)/(u+1))
// After these reductions |z| <= sqrt(2)-1, where the Taylor expansion
//   atan(z) ~= z*(1 - z^2/3 + z^4/5 - z^6/7 + z^8/9)
// is accurate to ~f32 ULP and evaluated via Horner's scheme.
class ONNXAtanOpLoweringToTOSA : public OpConversionPattern<ONNXAtanOp> {
public:
  using OpConversionPattern<ONNXAtanOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXAtanOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXAtanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getInput();

    auto inputType = mlir::dyn_cast<RankedTensorType>(x.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "ONNXAtanOp lowering to TOSA requires a ranked tensor input");
    Type elementType = inputType.getElementType();
    if (!isTOSAFloat(elementType))
      return rewriter.notifyMatchFailure(op,
          "ONNXAtanOp lowering to TOSA only supports f32/f16/bf16 element "
          "types");

    Type outputType = op.getType();
    Type predType = rewriter.getI1Type();
    // Splat constants share the same rank as the input so that no rank
    // broadcast is needed for the elementwise TOSA ops below.
    llvm::SmallVector<int64_t, 4> splatShape(inputType.getRank(), 1);

    TosaBuilder tosaBuilder(rewriter, loc);

    Value zero = tosaBuilder.getSplattedConst(0.0f, splatShape, elementType);
    Value one = tosaBuilder.getSplattedConst(1.0f, splatShape, elementType);
    Value sqrt2m1 = tosaBuilder.getSplattedConst(
        0.41421356237309515f, splatShape, elementType);
    Value piOver2 = tosaBuilder.getSplattedConst(
        1.5707963267948966f, splatShape, elementType);
    Value piOver4 = tosaBuilder.getSplattedConst(
        0.7853981633974483f, splatShape, elementType);

    // Taylor coefficients for the odd terms of atan(z).
    Value c1 = tosaBuilder.getSplattedConst(1.0f, splatShape, elementType);
    Value c3 =
        tosaBuilder.getSplattedConst(-0x1.55543ap-2, splatShape, elementType);
    Value c5 =
        tosaBuilder.getSplattedConst(0x1.992194p-3, splatShape, elementType);
    Value c7 =
        tosaBuilder.getSplattedConst(-0x1.1c4eccp-3, splatShape, elementType);
    Value c9 =
        tosaBuilder.getSplattedConst(0x1.4e0b5p-4, splatShape, elementType);
    Value c11 =
        tosaBuilder.getSplattedConst(-0x1.ee3dfap-9, splatShape, elementType);

    // Stage 1: take |x| and remember the sign predicate.
    Value absX =
        tosa::CreateOpAndInfer<mlir::tosa::AbsOp>(rewriter, loc, outputType, x);
    // negPred : x < 0  (equivalently, 0 > x)
    Value negPred = tosaBuilder.binaryOp<mlir::tosa::GreaterOp>(zero, x, predType);

    // Stage 2: if |x| > 1 use 1/|x|, so the working value y lies in [0, 1].
    Value gt1 = tosaBuilder.binaryOp<mlir::tosa::GreaterOp>(absX, one, predType);
    Value recAbs = tosaBuilder.reciprocal(absX);
    Value y = tosaBuilder.select(gt1, recAbs, absX);

    // Stage 3: if y > sqrt(2)-1 use (y-1)/(y+1) so |z| <= sqrt(2)-1.
    Value gtC0 = tosaBuilder.binaryOp<mlir::tosa::GreaterOp>(y, sqrt2m1, predType);
    Value yPlus1 = tosaBuilder.binaryOp<mlir::tosa::AddOp>(y, one);
    Value yMinus1 = tosaBuilder.binaryOp<mlir::tosa::SubOp>(y, one);
    Value invYPlus1 = tosaBuilder.reciprocal(yPlus1);
    Value frac = tosaBuilder.mul(yMinus1, invYPlus1);
    Value z = tosaBuilder.select(gtC0, frac, y);

    // Horner evaluation of p(z) = z * (c1 + z^2*(c3 + z^2*(c5 + z^2*(c7 +
    // z^2*c9)))).
    Value z2 = tosaBuilder.mul(z, z);
    Value t = tosaBuilder.mul(z2, c11);
    t = tosaBuilder.binaryOp<mlir::tosa::AddOp>(t, c9);
    t = tosaBuilder.mul(t, z2);
    t = tosaBuilder.binaryOp<mlir::tosa::AddOp>(t, c7);
    t = tosaBuilder.mul(t, z2);
    t = tosaBuilder.binaryOp<mlir::tosa::AddOp>(t, c5);
    t = tosaBuilder.mul(t, z2);
    t = tosaBuilder.binaryOp<mlir::tosa::AddOp>(t, c3);
    t = tosaBuilder.mul(t, z2);
    t = tosaBuilder.binaryOp<mlir::tosa::AddOp>(t, c1);
    Value atanZ = tosaBuilder.mul(t, z);

    // Stage 3 restore: atan(u) = pi/4 + atan(z) when the reduction was taken.
    Value atanZPlusPi4 =
        tosaBuilder.binaryOp<mlir::tosa::AddOp>(atanZ, piOver4);
    Value atanU = tosaBuilder.select(gtC0, atanZPlusPi4, atanZ);

    // Stage 2 restore: atan(|x|) = pi/2 - atan(u) when the reduction was
    // taken.
    Value pi2MinusAtanU =
        tosaBuilder.binaryOp<mlir::tosa::SubOp>(piOver2, atanU);
    Value atanAbsX = tosaBuilder.select(gt1, pi2MinusAtanU, atanU);

    // Stage 1 restore: atan(x) = -atan(|x|) for x < 0.
    Value negAtanAbsX = tosa::CreateOpAndInfer<mlir::tosa::NegateOp>(
        rewriter, loc, outputType, atanAbsX);
    Value result = tosaBuilder.select(negPred, negAtanAbsX, atanAbsX);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ONNXErfOpLoweringToTOSA : public OpConversionPattern<ONNXErfOp> {
public:
  using OpConversionPattern<ONNXErfOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXErfOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXErfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value input = adaptor.getInput();
    rewriter.replaceOp(op, tosaBuilder.erf(input));
    return success();
  }
};

class ONNXTanhOpLoweringToTOSA : public OpConversionPattern<ONNXTanhOp> {
public:
  using OpConversionPattern<ONNXTanhOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXTanhOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXTanhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value input = adaptor.getInput();
    rewriter.replaceOp(op, tosaBuilder.tanh(input));
    return success();
  }
};

class ONNXFloorOpLoweringToTOSA : public OpConversionPattern<ONNXFloorOp> {
public:
  using OpConversionPattern<ONNXFloorOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXFloorOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXFloorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto scalarType = getElementTypeOrSelf(adaptor.getX());
    if (!isTOSAFloat(scalarType))
      return rewriter.notifyMatchFailure(
          op, "`tosa.floor` only supports float types");

    rewriter.replaceOpWithNewOp<mlir::tosa::FloorOp>(
        op, op.getType(), adaptor.getX());
    return success();
  }
};

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getX();

    // Quantized types are not supported right now (in type conversion).
    // Once they are, the input should be rescaled for quantized types. (TBD)
    // Maps to `tosa.clamp` which has both int and fp limits.
    auto inputElementType =
        llvm::cast<TensorType>(op.getType()).getElementType();
    if (llvm::isa<IntegerType>(inputElementType)) {
      auto minClamp = rewriter.getI64IntegerAttr(0);
      auto maxClamp =
          rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max());
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
          op, op.getType(), input, minClamp, maxClamp);
    } else {
      auto minClamp = rewriter.getF32FloatAttr(0.0f);
      auto maxClamp =
          rewriter.getF32FloatAttr(std::numeric_limits<float>::max());
      rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
          op, op.getType(), input, minClamp, maxClamp);
    }
    return success();
  }
};

// Extract a scalar ElementsAttr from a value defined either by
// onnx.Constant (before conversion) or tosa.const (after conversion).
static ElementsAttr getScalarConstantElementsAttr(Value v) {
  if (auto onnxConst =
          mlir::dyn_cast_or_null<ONNXConstantOp>(v.getDefiningOp()))
    return mlir::dyn_cast_or_null<ElementsAttr>(onnxConst.getValueAttr());
  if (auto tosaConst =
          mlir::dyn_cast_or_null<mlir::tosa::ConstOp>(v.getDefiningOp()))
    return tosaConst.getValues();
  return nullptr;
}

class ONNXClipOpLoweringToTOSA : public OpConversionPattern<ONNXClipOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXClipOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value min = adaptor.getMin();
    Value max = adaptor.getMax();

    auto inputType = mlir::dyn_cast<TensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    Type elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "only int and float types are supported");

    Attribute minAttr;
    Attribute maxAttr;

    if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
      const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
      APFloat minVal = APFloat::getLargest(semantics, /*Negative=*/true);
      APFloat maxVal = APFloat::getLargest(semantics, /*Negative=*/false);

      if (!isNoneValue(min)) {
        ElementsAttr minElems = getScalarConstantElementsAttr(min);
        if (!minElems)
          return rewriter.notifyMatchFailure(
              op, "min must be a constant for tosa.clamp");
        minVal = *minElems.getValues<APFloat>().begin();
      }
      if (!isNoneValue(max)) {
        ElementsAttr maxElems = getScalarConstantElementsAttr(max);
        if (!maxElems)
          return rewriter.notifyMatchFailure(
              op, "max must be a constant for tosa.clamp");
        maxVal = *maxElems.getValues<APFloat>().begin();
      }
      minAttr = rewriter.getFloatAttr(elementType, minVal);
      maxAttr = rewriter.getFloatAttr(elementType, maxVal);
    } else {
      auto intType = mlir::cast<IntegerType>(elementType);
      unsigned width = intType.getWidth();
      APInt minVal = intType.isUnsigned() ? APInt::getMinValue(width)
                                          : APInt::getSignedMinValue(width);
      APInt maxVal = intType.isUnsigned() ? APInt::getMaxValue(width)
                                          : APInt::getSignedMaxValue(width);

      if (!isNoneValue(min)) {
        ElementsAttr minElems = getScalarConstantElementsAttr(min);
        if (!minElems)
          return rewriter.notifyMatchFailure(
              op, "min must be a constant for tosa.clamp");
        minVal = *minElems.getValues<APInt>().begin();
      }
      if (!isNoneValue(max)) {
        ElementsAttr maxElems = getScalarConstantElementsAttr(max);
        if (!maxElems)
          return rewriter.notifyMatchFailure(
              op, "max must be a constant for tosa.clamp");
        maxVal = *maxElems.getValues<APInt>().begin();
      }
      minAttr = rewriter.getIntegerAttr(elementType, minVal);
      maxAttr = rewriter.getIntegerAttr(elementType, maxVal);
    }

    rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(
        op, op.getType(), input, minAttr, maxAttr);
    return success();
  }
};

class ONNXDivOpLoweringToTOSA : public OpConversionPattern<ONNXDivOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    auto resultType = mlir::cast<TensorType>(op.getResult().getType());
    Type resultElementType = resultType.getElementType();

    TosaBuilder tosaBuilder(rewriter, op->getLoc());

    if (resultElementType.isSignlessInteger(32)) {
      // tosa::IntDivOp takes 32-but signless integers as inputs
      Value divOp = tosaBuilder.intdiv(lhs, rhs);
      rewriter.replaceOp(op, {divOp});
      return success();
    }
    // If it is not a 32-bit signless integer, decompose ONNXDivOp into
    // tosa::ReciprocalOp and tosa::MulOp
    Value reciprocalOp = tosaBuilder.reciprocal(rhs);
    Value mulOp = tosaBuilder.mul(lhs, reciprocalOp);
    rewriter.replaceOp(op, {mulOp});
    return success();
  }
};

class ONNXWhereOpLoweringToTOSA : public OpConversionPattern<ONNXWhereOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXWhereOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value cond = adaptor.getCondition();
    Value lhs = adaptor.getX();
    Value rhs = adaptor.getY();

    auto condType = mlir::dyn_cast<TensorType>(cond.getType());
    auto lhsType = mlir::dyn_cast<TensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<TensorType>(rhs.getType());
    auto resultType = mlir::dyn_cast<TensorType>(op.getResult().getType());
    if (!condType || !lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");

    Type resultElementType = resultType.getElementType();
    if (!resultElementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "only int and float are supported");

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value selectOp = tosaBuilder.select(cond, lhs, rhs);
    rewriter.replaceOp(op, {selectOp});
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXAddOp, mlir::tosa::AddOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXSubOp, mlir::tosa::SubOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXEqualOp, mlir::tosa::EqualOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXGreaterOp,
          mlir::tosa::GreaterOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXGreaterOrEqualOp,
          mlir::tosa::GreaterEqualOp>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXLessOp, mlir::tosa::GreaterOp,
          /*SwapOperands=*/true>,
      ONNXBinaryElementwiseOpLoweringToTOSA<ONNXLessOrEqualOp,
          mlir::tosa::GreaterEqualOp, /*SwapOperands=*/true>,
      ONNXSinOpLoweringToTOSA, ONNXCosOpLoweringToTOSA, ONNXErfOpLoweringToTOSA,
      ONNXTanhOpLoweringToTOSA, ONNXGeluOpLoweringToTOSA,
      ONNXAtanOpLoweringToTOSA, ONNXFloorOpLoweringToTOSA,
      ONNXReluOpLoweringToTOSA, ONNXClipOpLoweringToTOSA,
      ONNXDivOpLoweringToTOSA, ONNXWhereOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
