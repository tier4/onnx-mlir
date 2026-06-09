// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_gather_axis0(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xi64>) -> tensor<2x2x2xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  return %0 : tensor<2x2x2xf32>
// CHECK-LABEL:  func.func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<2x2xi64>) -> tensor<2x2x2xf32> {
// CHECK:           [[VAL_:%.+]] = tosa.reshape [[PARAM_0_]], {{.*}} : (tensor<3x2xf32>, !tosa.shape<3>) -> tensor<1x3x2xf32>
// CHECK:           [[CAST_:%.+]] = tosa.cast [[PARAM_1_]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK:           [[IDX_:%.+]] = tosa.reshape [[CAST_]], {{.*}} : (tensor<2x2xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
// CHECK:           tosa.greater
// CHECK:           tosa.add
// CHECK:           tosa.select
// CHECK:           [[GATHER_:%.+]] = tosa.gather [[VAL_]], {{.*}} : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[GATHER_]], {{.*}} : (tensor<1x4x2xf32>, !tosa.shape<3>) -> tensor<2x2x2xf32>
// CHECK:           return [[RES_]] : tensor<2x2x2xf32>
}

// -----

// Gathering along a non-zero axis requires transposing the gathered axis to the
// front and transposing the result back to the ONNX layout.
func.func @test_gather_axis1(%arg0: tensor<3x3xf32>, %arg1: tensor<1x2xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:  func.func @test_gather_axis1
// CHECK:           [[T0_:%.+]] = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK:           tosa.reshape [[T0_]], {{.*}} -> tensor<1x3x3xf32>
// CHECK:           [[GATHER_:%.+]] = tosa.gather {{.*}} -> tensor<1x2x3xf32>
// CHECK:           [[RESHAPE_:%.+]] = tosa.reshape [[GATHER_]], {{.*}} -> tensor<1x2x3xf32>
// CHECK:           [[RES_:%.+]] = tosa.transpose [[RESHAPE_]] {perms = array<i32: 2, 0, 1>} : (tensor<1x2x3xf32>) -> tensor<3x1x2xf32>
// CHECK:           return [[RES_]] : tensor<3x1x2xf32>
}

// -----

// GatherND, batch_dims = 0, indices[-1] == rank (gathers scalars).
func.func @test_gathernd(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// CHECK-LABEL:  func.func @test_gathernd
// CHECK:           [[VAL_:%.+]] = tosa.reshape %arg0, {{.*}} : (tensor<2x2x2xf32>, !tosa.shape<3>) -> tensor<1x4x2xf32>
// CHECK:           [[IDX_:%.+]] = tosa.reshape {{.*}} -> tensor<1x2x2xi32>
// CHECK:           [[FLAT_:%.+]] = tosa.reduce_sum {{.*}} {axis = 2 : i32} : (tensor<1x2x2xi32>) -> tensor<1x2x1xi32>
// CHECK:           [[IDX2_:%.+]] = tosa.reshape [[FLAT_]], {{.*}} -> tensor<1x2xi32>
// CHECK:           [[GATHER_:%.+]] = tosa.gather [[VAL_]], [[IDX2_]] : (tensor<1x4x2xf32>, tensor<1x2xi32>) -> tensor<1x2x2xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[GATHER_]], {{.*}} : (tensor<1x2x2xf32>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK:           return [[RES_]] : tensor<2x2xf32>
}

// -----

// GatherND with batch_dims = 1.
func.func @test_gathernd_batch(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64} : (tensor<2x2x2xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// CHECK-LABEL:  func.func @test_gathernd_batch
// CHECK:           [[VAL_:%.+]] = tosa.reshape %arg0, {{.*}} : (tensor<2x2x2xf32>, !tosa.shape<3>) -> tensor<2x2x2xf32>
// CHECK:           [[GATHER_:%.+]] = tosa.gather [[VAL_]], {{.*}} : (tensor<2x2x2xf32>, tensor<2x1xi32>) -> tensor<2x1x2xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[GATHER_]], {{.*}} : (tensor<2x1x2xf32>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK:           return [[RES_]] : tensor<2x2xf32>
}

// -----

// Indices that are already i32 do not need a cast.
func.func @test_gather_i32_indices(%arg0: tensor<4x8xf32>, %arg1: tensor<3xi32>) -> tensor<3x8xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<4x8xf32>, tensor<3xi32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
// CHECK-LABEL:  func.func @test_gather_i32_indices
// CHECK-NOT:       tosa.cast
// CHECK:           tosa.gather {{.*}} : (tensor<1x4x8xf32>, tensor<1x3xi32>) -> tensor<1x3x8xf32>
}
