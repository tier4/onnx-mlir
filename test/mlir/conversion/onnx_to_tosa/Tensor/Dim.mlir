// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_dim(%arg0 : tensor<2x3x5xf32>) -> tensor<1xi64> {
  %0 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<2x3x5xf32>) -> tensor<1xi64>
  "func.return"(%0) : (tensor<1xi64>) -> ()
// CHECK-LABEL:  func @test_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x5xf32>) -> tensor<1xi64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{values = dense<3> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK:           return [[VAR_0_]] : tensor<1xi64>
}

// -----

func.func @test_dim_dynamic(%arg0 : tensor<?x3xf32>) -> tensor<1xi64> {
  %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x3xf32>) -> tensor<1xi64>
  "func.return"(%0) : (tensor<1xi64>) -> ()
// CHECK-LABEL:  func @test_dim_dynamic
// CHECK:           onnx.Dim
}
