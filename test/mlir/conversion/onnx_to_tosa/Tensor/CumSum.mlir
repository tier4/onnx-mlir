// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @cumsum_incl_fwd(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %axis = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.CumSum"(%arg0, %axis) : (tensor<5xf32>, tensor<i64>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
// CHECK-LABEL:  func @cumsum_incl_fwd
// CHECK:       tosa.slice %arg0, %{{.*}}, %{{.*}} : (tensor<5xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xf32>
// CHECK:       tosa.const_shape  {values = dense<[1, 0]>
// CHECK:       tosa.pad %{{.*}} : (tensor<4xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<5xf32>
// CHECK:       tosa.add %arg0, %{{.*}} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:       tosa.const_shape  {values = dense<[2, 0]>
// CHECK:       tosa.const_shape  {values = dense<[4, 0]>
}

// -----

func.func @cumsum_excl_fwd(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %axis = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64} : (tensor<5xf32>, tensor<i64>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
// CHECK-LABEL:  func @cumsum_excl_fwd
// CHECK:       tosa.add
// CHECK:       tosa.add
// CHECK:       tosa.add
// CHECK:       tosa.slice %{{.*}}: (tensor<5xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xf32>
// CHECK:       tosa.const_shape  {values = dense<[1, 0]>
// CHECK:       tosa.pad %{{.*}}: (tensor<4xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<5xf32>
// CHECK-NOT:   tosa.add
}

// -----

func.func @cumsum_incl_rev(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %axis = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.CumSum"(%arg0, %axis) {reverse = 1 : si64} : (tensor<5xf32>, tensor<i64>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
// CHECK-LABEL:  func @cumsum_incl_rev
// CHECK:       tosa.const_shape  {values = dense<1> : tensor<1xindex>}
// CHECK:       tosa.slice %arg0, %{{.*}} : (tensor<5xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xf32>
// CHECK:       tosa.const_shape  {values = dense<[0, 1]> : tensor<2xindex>}
// CHECK:       tosa.pad %{{.*}}: (tensor<4xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<5xf32>
// CHECK:       tosa.const_shape  {values = dense<[0, 2]>
// CHECK:       tosa.const_shape  {values = dense<[0, 4]>
}

// -----

func.func @cumsum_2d_excl_rev_neg(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %axis = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf32>, tensor<i64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL:  func @cumsum_2d_excl_rev_neg
// CHECK:       tosa.slice %arg0, %{{.*}}: (tensor<2x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK:       tosa.const_shape  {values = dense<[0, 0, 0, 1]> : tensor<4xindex>}
// CHECK:       tosa.pad %{{.*}}: (tensor<2x2xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<2x3xf32>
// CHECK:       tosa.add
// CHECK:       tosa.const_shape  {values = dense<[0, 0, 0, 2]> : tensor<4xindex>}
// CHECK:       tosa.add
// CHECK:       tosa.slice %{{.*}}: (tensor<2x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
}
