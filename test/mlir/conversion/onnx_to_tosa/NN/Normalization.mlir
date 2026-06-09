// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_layer_norm(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x3x4xf32>, none, none)
  return %Y : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_layer_norm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<4xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 2 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.mul [[VAR_0_]], {{.*}} : (tensor<2x3x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<2x3x1xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_5_]] : (tensor<2x3x4xf32>, tensor<2x3x1xf32>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_8_:%.+]] = tosa.mul [[VAR_6_]], [[VAR_6_]], {{.*}} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_9_:%.+]] = tosa.reduce_sum [[VAR_8_]] {axis = 2 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK:           [[VAR_18_:%.+]] = tosa.add {{.*}} : (tensor<2x3x1xf32>, tensor<1x1x1xf32>) -> tensor<2x3x1xf32>
// CHECK:           [[VAR_19_:%.+]] = tosa.rsqrt [[VAR_18_]] : (tensor<2x3x1xf32>) -> tensor<2x3x1xf32>
// CHECK:           [[VAR_21_:%.+]] = tosa.mul [[VAR_6_]], [[VAR_19_]], {{.*}} : (tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_25_:%.+]] = tosa.mul [[VAR_21_]], {{.*}} : (tensor<2x3x4xf32>, tensor<1x1x4xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_28_:%.+]] = tosa.add [[VAR_25_]], {{.*}} : (tensor<2x3x4xf32>, tensor<1x1x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           return [[VAR_28_]] : tensor<2x3x4xf32>
}

// -----

// Mean and InvStdDev outputs are produced when requested.
func.func @test_layer_norm_all_outputs(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<2x3x1xf32>) {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<2x3x1xf32>)
  return %Y, %Mean, %InvStdDev : tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<2x3x1xf32>
// CHECK-LABEL:  func.func @test_layer_norm_all_outputs
// CHECK:           [[MEAN_:%.+]] = tosa.mul {{.*}} : (tensor<2x3x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<2x3x1xf32>
// CHECK:           [[INVSTD_:%.+]] = tosa.rsqrt {{.*}} : (tensor<2x3x1xf32>) -> tensor<2x3x1xf32>
// CHECK:           return {{.*}}, [[MEAN_]], [[INVSTD_]] : tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<2x3x1xf32>
}

// -----

// Bias B is optional.
func.func @test_layer_norm_no_bias(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %none) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x3x4xf32>, tensor<4xf32>, none) -> (tensor<2x3x4xf32>, none, none)
  return %Y : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_layer_norm_no_bias
// CHECK:           [[RES_:%.+]] = tosa.mul {{.*}} : (tensor<2x3x4xf32>, tensor<1x1x4xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
// CHECK-NOT:       tosa.add {{.*}} : (tensor<2x3x4xf32>, tensor<1x1x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           return [[RES_]] : tensor<2x3x4xf32>
}
