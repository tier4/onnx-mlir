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

// A step of 2 over an evenly divisible span: slice the span, then
// reshape/slice/reshape to pick every other element.
func.func @test_slice_step2(%arg0 : tensor<4xf32>) -> tensor<2xf32> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps  = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
// CHECK-LABEL:  func @test_slice_step2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xf32>) -> tensor<2xf32> {
// CHECK:           [[SLICE0:%.+]] = tosa.slice [[PARAM_0_]], {{.*}} -> tensor<3xf32>
// CHECK:           [[PAD:%.+]] = tosa.pad [[SLICE0]], {{.*}} -> tensor<4xf32>
// CHECK:           [[RESHAPE0:%.+]] = tosa.reshape [[PAD]], {{.*}} -> tensor<2x2xf32>
// CHECK:           [[SLICE1:%.+]] = tosa.slice [[RESHAPE0]], {{.*}} -> tensor<2x1xf32>
// CHECK:           [[RESHAPE1:%.+]] = tosa.reshape [[SLICE1]], {{.*}} -> tensor<2xf32>
// CHECK:           return [[RESHAPE1]] : tensor<2xf32>
}

// -----

// A step of 2 over a span that is not a multiple of the step: the span is
// padded up before the reshape so the last element is still selected.
func.func @test_slice_step2_odd(%arg0 : tensor<5xf32>) -> tensor<3xf32> {
  %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<5> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps  = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL:  func @test_slice_step2_odd
// CHECK:           [[SLICE0:%.+]] = tosa.slice [[PARAM_0_:%.+]], {{.*}} -> tensor<5xf32>
// CHECK:           [[PAD:%.+]] = tosa.pad [[SLICE0]], {{.*}} -> tensor<6xf32>
// CHECK:           [[RESHAPE0:%.+]] = tosa.reshape [[PAD]], {{.*}} -> tensor<3x2xf32>
// CHECK:           [[SLICE1:%.+]] = tosa.slice [[RESHAPE0]], {{.*}} -> tensor<3x1xf32>
// CHECK:           [[RESHAPE1:%.+]] = tosa.reshape [[SLICE1]], {{.*}} -> tensor<3xf32>
// CHECK:           return [[RESHAPE1]] : tensor<3xf32>
}

// -----

// A negative step reverses the axis first, then performs a contiguous slice.
func.func @test_slice_negative_step(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %starts = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<-5> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps  = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  "func.return"(%0) : (tensor<4xf32>) -> ()
// CHECK-LABEL:  func @test_slice_negative_step
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           [[REVERSE:%.+]] = tosa.reverse [[PARAM_0_]] {axis = 0 : i32} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK:           [[SLICE:%.+]] = tosa.slice [[REVERSE]], {{.*}} -> tensor<4xf32>
// CHECK:           return [[SLICE]] : tensor<4xf32>
}
