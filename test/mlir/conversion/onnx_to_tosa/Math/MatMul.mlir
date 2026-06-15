// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// -----

func.func @test_matmul_2d(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
// CHECK-LABEL: func.func @test_matmul_2d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<3x4xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], [[ZP_A]], [[ZP_B]] : (tensor<1x3x4xf32>, tensor<1x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x5xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[3, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<1x3x5xf32>, !tosa.shape<2>) -> tensor<3x5xf32>
// CHECK:           return [[RESULT]] : tensor<3x5xf32>
}

// -----

func.func @test_matmul_3d(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_3d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[PARAM_0_]], [[PARAM_1_]], [[ZP_A]], [[ZP_B]] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x3x5xf32>
// CHECK:           return
}

// -----

func.func @test_matmul_nd(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  return %0 : tensor<2x3x4x6xf32>
// CHECK-LABEL: func.func @test_matmul_nd
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[6, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<2x3x4x5xf32>, !tosa.shape<3>) -> tensor<6x4x5xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[6, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<2x3x5x6xf32>, !tosa.shape<3>) -> tensor<6x5x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], [[ZP_A]], [[ZP_B]] : (tensor<6x4x5xf32>, tensor<6x5x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<6x4x6xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[2, 3, 4, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<6x4x6xf32>, !tosa.shape<4>) -> tensor<2x3x4x6xf32>
// CHECK:           return [[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @test_matmul_3d_x_2d(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_3d_x_2d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, 6, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<2x3x4xf32>, !tosa.shape<3>) -> tensor<1x6x4xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], [[ZP_A]], [[ZP_B]] : (tensor<1x6x4xf32>, tensor<1x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x6x5xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[2, 3, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<1x6x5xf32>, !tosa.shape<3>) -> tensor<2x3x5xf32>
// CHECK:           return [[RESULT]] : tensor<2x3x5xf32>
}

// -----

func.func @test_matmul_2d_x_3d(%arg0: tensor<3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_2d_x_3d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[TILE_MULT:%.+]] = tosa.const_shape {values = dense<[2, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A_TILED:%.+]] = tosa.tile [[A3D]], [[TILE_MULT]] : (tensor<1x3x4xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A_TILED]], [[PARAM_1_]], [[ZP_A]], [[ZP_B]] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x3x5xf32>
// CHECK:           return
}

// -----

func.func @test_matmul_2d_x_4d(%arg0: tensor<3x4xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32>
  return %0 : tensor<2x3x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_2d_x_4d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[6, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<2x3x4x5xf32>, !tosa.shape<3>) -> tensor<6x4x5xf32>
// CHECK-DAG:       [[TILE_MULT:%.+]] = tosa.const_shape {values = dense<[6, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A_TILED:%.+]] = tosa.tile [[A3D]], [[TILE_MULT]] : (tensor<1x3x4xf32>, !tosa.shape<3>) -> tensor<6x3x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A_TILED]], [[B3D]], [[ZP_A]], [[ZP_B]] : (tensor<6x3x4xf32>, tensor<6x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<6x3x5xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[2, 3, 3, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<6x3x5xf32>, !tosa.shape<4>) -> tensor<2x3x3x5xf32>
// CHECK:           return [[RESULT]] : tensor<2x3x3x5xf32>
}

// -----

// Dynamic shapes are not supported: the op must stay as onnx.MatMul.
func.func @test_matmul_dynamic(%arg0: tensor<?x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<?x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x5xf32>) -> tensor<?x5xf32>
  return %0 : tensor<?x5xf32>
// CHECK-LABEL: func.func @test_matmul_dynamic
// CHECK:           "onnx.MatMul"
// CHECK-NOT:       tosa.matmul
}
