// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_transpose(%arg0 : tensor<1x2x3xf32>) -> tensor<2x3x1xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [1, 2, 0]} : (tensor<1x2x3xf32>) -> tensor<2x3x1xf32>
  "func.return"(%0) : (tensor<2x3x1xf32>) -> ()
// CHECK-LABEL:  func @test_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3xf32>) -> tensor<2x3x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 1, 2, 0>} : (tensor<1x2x3xf32>) -> tensor<2x3x1xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<2x3x1xf32>
}

// -----

func.func @test_transpose_default(%arg0 : tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  "func.return"(%0) : (tensor<3x2x1xf32>) -> ()
// CHECK-LABEL:  func @test_transpose_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 2, 1, 0>} : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<3x2x1xf32>
}
