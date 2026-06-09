// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// ScatterElements along axis 0: data[indices[i][j]][j] = updates[i][j].
func.func @test_scatter_elements_axis0(%arg0: tensor<3x3xf32>, %arg1: tensor<2x3xi64>, %arg2: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<2x3xi64>, tensor<2x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
// CHECK-LABEL:  func.func @test_scatter_elements_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xi64>, [[PARAM_2_:%.+]]: tensor<2x3xf32>) -> tensor<3x3xf32> {
// CHECK:           [[VAL_:%.+]] = tosa.reshape [[PARAM_0_]], {{.*}} : (tensor<3x3xf32>, !tosa.shape<3>) -> tensor<1x9x1xf32>
// CHECK:           [[CAST_:%.+]] = tosa.cast [[PARAM_1_]] : (tensor<2x3xi64>) -> tensor<2x3xi32>
// CHECK:           tosa.greater
// CHECK:           tosa.add
// CHECK:           tosa.select
// CHECK:           tosa.mul
// CHECK:           tosa.add
// CHECK:           [[IDX_:%.+]] = tosa.reshape {{.*}} -> tensor<1x6xi32>
// CHECK:           [[UPD_:%.+]] = tosa.reshape [[PARAM_2_]], {{.*}} : (tensor<2x3xf32>, !tosa.shape<3>) -> tensor<1x6x1xf32>
// CHECK:           [[SCATTER_:%.+]] = tosa.scatter [[VAL_]], [[IDX_]], [[UPD_]] : (tensor<1x9x1xf32>, tensor<1x6xi32>, tensor<1x6x1xf32>) -> tensor<1x9x1xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[SCATTER_]], {{.*}} : (tensor<1x9x1xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
// CHECK:           return [[RES_]] : tensor<3x3xf32>
}

// -----

// ScatterElements along axis 1.
func.func @test_scatter_elements_axis1(%arg0: tensor<1x5xf32>, %arg1: tensor<1x2xi64>, %arg2: tensor<1x2xf32>) -> tensor<1x5xf32> {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<1x5xf32>, tensor<1x2xi64>, tensor<1x2xf32>) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
// CHECK-LABEL:  func.func @test_scatter_elements_axis1
// CHECK:           [[VAL_:%.+]] = tosa.reshape {{.*}} -> tensor<1x5x1xf32>
// CHECK:           [[SCATTER_:%.+]] = tosa.scatter {{.*}} : (tensor<1x5x1xf32>, tensor<1x2xi32>, tensor<1x2x1xf32>) -> tensor<1x5x1xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[SCATTER_]], {{.*}} -> tensor<1x5xf32>
// CHECK:           return [[RES_]] : tensor<1x5xf32>
}

// -----

// ScatterND, indices[-1] == rank (scatters scalars).
func.func @test_scatternd(%arg0: tensor<8xf32>, %arg1: tensor<4x1xi64>, %arg2: tensor<4xf32>) -> tensor<8xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
// CHECK-LABEL:  func.func @test_scatternd
// CHECK:           [[VAL_:%.+]] = tosa.reshape %arg0, {{.*}} : (tensor<8xf32>, !tosa.shape<3>) -> tensor<1x8x1xf32>
// CHECK:           [[IDX_:%.+]] = tosa.reshape {{.*}} -> tensor<1x4x1xi32>
// CHECK:           [[FLAT_:%.+]] = tosa.reduce_sum {{.*}} {axis = 2 : i32} : (tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
// CHECK:           [[IDX2_:%.+]] = tosa.reshape [[FLAT_]], {{.*}} -> tensor<1x4xi32>
// CHECK:           [[UPD_:%.+]] = tosa.reshape %arg2, {{.*}} : (tensor<4xf32>, !tosa.shape<3>) -> tensor<1x4x1xf32>
// CHECK:           [[SCATTER_:%.+]] = tosa.scatter [[VAL_]], [[IDX2_]], [[UPD_]] : (tensor<1x8x1xf32>, tensor<1x4xi32>, tensor<1x4x1xf32>) -> tensor<1x8x1xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[SCATTER_]], {{.*}} : (tensor<1x8x1xf32>, !tosa.shape<1>) -> tensor<8xf32>
// CHECK:           return [[RES_]] : tensor<8xf32>
}

// -----

// ScatterND that scatters slices (indices[-1] < rank).
func.func @test_scatternd_slices(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
// CHECK-LABEL:  func.func @test_scatternd_slices
// CHECK:           [[VAL_:%.+]] = tosa.reshape %arg0, {{.*}} : (tensor<4x4x4xf32>, !tosa.shape<3>) -> tensor<1x4x16xf32>
// CHECK:           [[UPD_:%.+]] = tosa.reshape %arg2, {{.*}} : (tensor<2x4x4xf32>, !tosa.shape<3>) -> tensor<1x2x16xf32>
// CHECK:           [[SCATTER_:%.+]] = tosa.scatter [[VAL_]], {{.*}}, [[UPD_]] : (tensor<1x4x16xf32>, tensor<1x2xi32>, tensor<1x2x16xf32>) -> tensor<1x4x16xf32>
// CHECK:           [[RES_:%.+]] = tosa.reshape [[SCATTER_]], {{.*}} : (tensor<1x4x16xf32>, !tosa.shape<3>) -> tensor<4x4x4xf32>
// CHECK:           return [[RES_]] : tensor<4x4x4xf32>
}

// -----

// Indices that are already i32 do not need a cast.
func.func @test_scatter_elements_i32_indices(%arg0: tensor<4x8xf32>, %arg1: tensor<2x8xi32>, %arg2: tensor<2x8xf32>) -> tensor<4x8xf32> {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<4x8xf32>, tensor<2x8xi32>, tensor<2x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
// CHECK-LABEL:  func.func @test_scatter_elements_i32_indices
// CHECK-NOT:       tosa.cast
// CHECK:           tosa.scatter {{.*}} : (tensor<1x32x1xf32>, tensor<1x16xi32>, tensor<1x16x1xf32>) -> tensor<1x32x1xf32>
}

// -----

// reduction != "none" is not supported and must not be converted.
func.func @test_scatternd_reduction(%arg0: tensor<8xf32>, %arg1: tensor<4x1xi64>, %arg2: tensor<4xf32>) -> tensor<8xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
// CHECK-LABEL:  func.func @test_scatternd_reduction
// CHECK:           "onnx.ScatterND"
// CHECK-NOT:       tosa.scatter
}
