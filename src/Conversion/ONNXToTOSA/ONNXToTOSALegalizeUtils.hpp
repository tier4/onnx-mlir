/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.hpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020-2024 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect. It is also used by TensorFlow and torch-mlir.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
#define ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"     // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"         // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"                    // from @llvm-project
#include <src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp>

namespace onnx_mlir {
namespace tosa {

// Create a RankedTensorType with shape and all elements being 1
mlir::RankedTensorType reduceAxisToOne(llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType, mlir::Attribute encoding = {});

// Returns the value TOSA ConstOp
template <typename T>
T getValueFromTosaConst(mlir::Value &val) {
  return mlir::cast<T>(val.getDefiningOp<mlir::tosa::ConstOp>().getValues());
}

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Type result_ty, Args &&...args) {

  auto op = TosaOp::create(rewriter, loc, result_ty, args...);

  mlir::InferShapedTypeOpInterface shapeInterface =
      llvm::dyn_cast<mlir::InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  llvm::SmallVector<mlir::ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), op.getLoc(),
              op->getOperands(), op->getAttrDictionary(),
              op->getPropertiesStorage(), op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  assert(returnedShapes.size() >= 1 && "Expected at least one returned shape");
  auto predictedShape = returnedShapes[0];
  if (predictedShape.hasRank())
    updateType(nullptr, op, predictedShape.getDims(),
        mlir::cast<mlir::ShapedType>(result_ty).getElementType());
  return op;
}

template <typename TosaOp, typename... Args>
void CreateReplaceOpAndInfer(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, mlir::Type result_ty, Args &&...args) {
  auto result =
      CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

/// Create a padding tosa::ConstOp from ONNX to Tosa format.
/// The two formats are:
/// ONNX : [b1, b2, b3, b4, e1, e2, e3, e4]
/// TOSA :[[b1, e1], [b2, e2], [b3, e3], [b4, e4]]
mlir::Value buildOnnxToTosaPaddingConstOp(mlir::PatternRewriter &rewriter,
    llvm::ArrayRef<int64_t> onnxPads, mlir::Location loc,
    const std::initializer_list<int64_t> &initialVals = {},
    const std::initializer_list<int64_t> &lastVals = {});

/// Build a 1-D one-element splat constant of value 0 with element type
/// `elementType`. Used as the pad_const for tosa.pad.
mlir::Value buildZeroSplat(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, TosaBuilder &tosaBuilder, mlir::Type elementType);

/// Shift `v` along `axis` by `offset`, filling the vacated positions with
/// zeros of `elementType`. `shiftRight = true` produces
/// `out[..., i, ...] = v[..., i - offset, ...]` (with leading zeros);
/// `shiftRight = false` produces `out[..., i, ...] = v[..., i + offset, ...]`
/// (with trailing zeros).
mlir::Value shiftAlongAxis(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, TosaBuilder &tosaBuilder, mlir::Value v, int64_t offset,
    int64_t axis, bool shiftRight, llvm::ArrayRef<int64_t> dataShape,
    mlir::Type elementType);

/// Inclusive Hillis-Steele prefix-sum scan of `input` along `axis`.
/// `forward = true` for left-to-right scan, `forward = false` for reverse.
/// Uses O(K) memory and log2(K) shift+add steps, where K = dataShape[axis].
mlir::Value inclusiveScanAlongAxis(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, TosaBuilder &tosaBuilder, mlir::Value input,
    int64_t axis, bool forward, llvm::ArrayRef<int64_t> dataShape,
    mlir::Type elementType);

} // namespace tosa
} // namespace onnx_mlir

#endif // ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
