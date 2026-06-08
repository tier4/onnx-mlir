// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_slice(%arg0 : tensor<3x4xf32>) -> tensor<2x2xf32> {
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %ends   = "onnx.Constant"() {value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %axes   = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %steps  = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<3x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()
// CHECK-LABEL:  func @test_slice
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4xf32>) -> tensor<2x2xf32> {
// CHECK-DAG:       [[START:%.+]] = tosa.const_shape  {values = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[SIZE:%.+]] = tosa.const_shape  {values = dense<2> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[SLICE:%.+]] = tosa.slice [[PARAM_0_]], [[START]], [[SIZE]] : (tensor<3x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK:           return [[SLICE]] : tensor<2x2xf32>
}

// -----

func.func @test_slice_step(%arg0 : tensor<4xf32>) -> tensor<2xf32> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps  = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
// CHECK-LABEL:  func @test_slice_step
// CHECK:           onnx.Slice
}
