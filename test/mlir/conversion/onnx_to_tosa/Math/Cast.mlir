// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_cast_f32_to_i32(%arg0 : tensor<3x4xf32>) -> tensor<3x4xi32> {
  %0 = "onnx.Cast"(%arg0) {to = i32, saturate = 1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xi32>
  "func.return"(%0) : (tensor<3x4xi32>) -> ()
// CHECK-LABEL:  func @test_cast_f32_to_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xi32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.cast [[PARAM_0_]] : (tensor<3x4xf32>) -> tensor<3x4xi32>
// CHECK:           return [[VAR_0_]] : tensor<3x4xi32>
}

// -----

func.func @test_cast_i32_to_f32(%arg0 : tensor<3x4xi32>) -> tensor<3x4xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32, saturate = 1 : si64} : (tensor<3x4xi32>) -> tensor<3x4xf32>
  "func.return"(%0) : (tensor<3x4xf32>) -> ()
// CHECK-LABEL:  func @test_cast_i32_to_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4xi32>) -> tensor<3x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.cast [[PARAM_0_]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_0_]] : tensor<3x4xf32>
}

// -----

func.func @test_cast_f32_to_f16(%arg0 : tensor<2xf32>) -> tensor<2xf16> {
  %0 = "onnx.Cast"(%arg0) {to = f16, saturate = 1 : si64} : (tensor<2xf32>) -> tensor<2xf16>
  "func.return"(%0) : (tensor<2xf16>) -> ()
// CHECK-LABEL:  func @test_cast_f32_to_f16
// CHECK:           [[VAR_0_:%.+]] = tosa.cast [[PARAM_0_:%.+]] : (tensor<2xf32>) -> tensor<2xf16>
// CHECK:           return [[VAR_0_]] : tensor<2xf16>
}

// -----

// A cast to the same element type folds away to the input.
func.func @test_cast_noop(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32, saturate = 1 : si64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()
// CHECK-LABEL:  func @test_cast_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NOT:       tosa.cast
// CHECK:           return [[PARAM_0_]] : tensor<2x2xf32>
}
