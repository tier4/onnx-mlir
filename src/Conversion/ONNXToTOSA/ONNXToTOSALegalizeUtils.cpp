/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.cpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h" // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"          // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp" // from @llvm-project
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;
namespace onnx_mlir {
namespace tosa {

mlir::RankedTensorType reduceAxisToOne(
    llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), 1), elementType, encoding);
}

Value buildZeroSplat(ConversionPatternRewriter &rewriter, Location loc,
    TosaBuilder &tosaBuilder, Type elementType) {
  Value zeroF32 = tosaBuilder.getSplattedConst(0.0f, {1});
  if (elementType.isF32())
    return zeroF32;
  return tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
      rewriter, loc, RankedTensorType::get({1}, elementType), zeroF32);
}

Value shiftAlongAxis(ConversionPatternRewriter &rewriter, Location loc,
    TosaBuilder &tosaBuilder, Value v, int64_t offset, int64_t axis,
    bool shiftRight, llvm::ArrayRef<int64_t> dataShape, Type elementType) {
  int64_t rank = (int64_t)dataShape.size();
  llvm::SmallVector<int64_t> sliceStart(rank, 0);
  llvm::SmallVector<int64_t> sliceSize(dataShape.begin(), dataShape.end());
  sliceSize[axis] -= offset;
  if (!shiftRight)
    sliceStart[axis] = offset;

  Value sliced = tosaBuilder.slice(v, sliceSize, sliceStart);

  // Pad shape format: flat [bef0, aft0, bef1, aft1, ...].
  llvm::SmallVector<int64_t> padding(rank * 2, 0);
  if (shiftRight)
    padding[axis * 2] = offset;
  else
    padding[axis * 2 + 1] = offset;
  Value padShape = mlir::tosa::getTosaConstShape(rewriter, loc, padding);

  Value zero = buildZeroSplat(rewriter, loc, tosaBuilder, elementType);

  Type padTy =
      RankedTensorType::get(llvm::SmallVector<int64_t>(dataShape), elementType);
  return tosa::CreateOpAndInfer<mlir::tosa::PadOp>(
      rewriter, loc, padTy, sliced, padShape, zero);
}

Value inclusiveScanAlongAxis(ConversionPatternRewriter &rewriter, Location loc,
    TosaBuilder &tosaBuilder, Value input, int64_t axis, bool forward,
    llvm::ArrayRef<int64_t> dataShape, Type elementType) {
  int64_t K = dataShape[axis];
  Value result = input;
  for (int64_t offset = 1; offset < K; offset *= 2) {
    Value shifted = shiftAlongAxis(rewriter, loc, tosaBuilder, result, offset,
        axis, forward, dataShape, elementType);
    result = tosaBuilder.binaryOp<mlir::tosa::AddOp>(result, shifted);
  }
  return result;
}

Value buildOnnxToTosaPaddingConstOp(mlir::PatternRewriter &rewriter,
    llvm::ArrayRef<int64_t> onnxPads, Location loc,
    const std::initializer_list<int64_t> &initialVals,
    const std::initializer_list<int64_t> &lastVals) {

  // Create a new pad vec in the right format
  // ONNX : [b1, b2, b3, b4, e1, e2, e3, e4]
  // TOSA :[b1, e1, b2, e2, b3, e3, b4, e4]

  // Adds any initial or last vals, not included in onnxPads.
  llvm::SmallVector<int64_t, 8> tosaPads{initialVals};

  const unsigned int dimSize = onnxPads.size() / 2;
  for (unsigned int i = 0; i < dimSize; i++) {
    tosaPads.push_back(onnxPads[i]);
    tosaPads.push_back(onnxPads[i + dimSize]);
  }
  tosaPads.insert(tosaPads.end(), lastVals.begin(), lastVals.end());
  TosaBuilder tosaBuilder(rewriter, loc);

  return mlir::tosa::getTosaConstShape(rewriter, loc, tosaPads);
}

} // namespace tosa
} // namespace onnx_mlir
