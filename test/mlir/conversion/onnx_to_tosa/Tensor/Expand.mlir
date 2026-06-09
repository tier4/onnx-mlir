// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// Broadcast a size-1 dimension, same rank.
func.func @test_expand_same_rank(%arg0: tensor<3x1xf32>) -> tensor<3x4xf32> {
  %shape = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<3x1xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_expand_same_rank
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x1xf32>) -> tensor<3x4xf32> {
// CHECK:           [[MUL_:%.+]] = tosa.const_shape {{.*}} -> !tosa.shape<2>
// CHECK:           [[RES_:%.+]] = tosa.tile [[PARAM_0_]], [[MUL_]] : (tensor<3x1xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           return [[RES_]] : tensor<3x4xf32>
}

// -----

// Output rank larger than input rank: input is reshaped, then tiled.
func.func @test_expand_higher_rank(%arg0: tensor<3x1xf32>) -> tensor<2x3x4xf32> {
  %shape = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_expand_higher_rank
// CHECK:           [[RESHAPE_:%.+]] = tosa.reshape [[PARAM_0_:%.+]], {{.*}} : (tensor<3x1xf32>, !tosa.shape<3>) -> tensor<1x3x1xf32>
// CHECK:           [[RES_:%.+]] = tosa.tile [[RESHAPE_]], {{.*}} : (tensor<1x3x1xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
// CHECK:           return [[RES_]] : tensor<2x3x4xf32>
}

// -----

// A scalar (rank 0) expanded to a full tensor.
func.func @test_expand_scalar(%arg0: tensor<f32>) -> tensor<2x3xf32> {
  %shape = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL:  func.func @test_expand_scalar
// CHECK:           [[RESHAPE_:%.+]] = tosa.reshape {{.*}} -> tensor<1x1xf32>
// CHECK:           [[RES_:%.+]] = tosa.tile [[RESHAPE_]], {{.*}} : (tensor<1x1xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
// CHECK:           return [[RES_]] : tensor<2x3xf32>
}

// -----

// Integer element type is handled the same way (tile is type-agnostic).
func.func @test_expand_int(%arg0: tensor<1x4xi32>) -> tensor<3x4xi32> {
  %shape = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x4xi32>, tensor<2xi64>) -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
// CHECK-LABEL:  func.func @test_expand_int
// CHECK:           tosa.tile {{.*}} : (tensor<1x4xi32>, !tosa.shape<2>) -> tensor<3x4xi32>
}

// -----

// When the input already matches the output, no tile is emitted (identity).
func.func @test_expand_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %shape = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL:  func.func @test_expand_identity
// CHECK-NOT:       tosa.tile
// CHECK:           return [[PARAM_0_:%.+]] : tensor<2x3xf32>
}
