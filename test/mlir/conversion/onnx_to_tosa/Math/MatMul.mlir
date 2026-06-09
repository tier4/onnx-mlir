// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// -----
// 2D x 2D: [M,K] x [K,N] -> [M,N]
// Inputs are reshaped to 3D (batch=1) for tosa.matmul, then the result
// is reshaped back to 2D.

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
// 3D batched: [B,M,K] x [B,K,N] -> [B,M,N]
// tosa.matmul natively handles 3D; inputs are passed through directly.

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
// N-D batched (rank=4): [D0,D1,M,K] x [D0,D1,K,N] -> [D0,D1,M,N]
// Leading batch dims are flattened (D0*D1 = 2*3 = 6), matmul is run on
// 3D tensors, then the result is restored to the original 4D shape.

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
// 3D x 2D: A has batch dims, B is a 2-D weight matrix.
//   A: [B,M,K] x B: [K,N] -> [B,M,N]
// All of A's dims except K are folded into a flat row dim: [1, B*M, K] x [1, K, N].
// The batch*M product (2*3 = 6) must be statically known.

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
// 2D x 3D: A is a 2-D matrix broadcast over B's batch dimension.
//   A: [M,K] x B: [batch,K,N] -> [batch,M,N]
// A is expanded to [1,M,K] and tiled to [batch,M,K] so both inputs share
// the same batch dimension.  B's batch (2) must be statically known.

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
// 2D x N-D (N >= 4): A is a 2-D matrix broadcast over B's leading dims.
//   A: [M,K] x B: [d0,d1,K,N] -> [d0,d1,M,N]
// A's batch dims after alignment are all 1, so A is folded to [1,M,K] and
// tiled by the full broadcast batch product (6 = 2*3).  B's leading dims
// already match the broadcast batch shape and just get folded into one
// batch dim.  This is the "all-ones" fast path of broadcastAndFold.

func.func @test_matmul_2d_x_4d(%arg0: tensor<3x4xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32>
  return %0 : tensor<2x3x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_2d_x_4d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[TILE_MULT:%.+]] = tosa.const_shape {values = dense<[6, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A_TILED:%.+]] = tosa.tile [[A3D]], [[TILE_MULT]] : (tensor<1x3x4xf32>, !tosa.shape<3>) -> tensor<6x3x4xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[6, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<2x3x4x5xf32>, !tosa.shape<3>) -> tensor<6x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[ZP_A:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[ZP_B:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A_TILED]], [[B3D]], [[ZP_A]], [[ZP_B]] : (tensor<6x3x4xf32>, tensor<6x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<6x3x5xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[2, 3, 3, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<6x3x5xf32>, !tosa.shape<4>) -> tensor<2x3x3x5xf32>
// CHECK:           return [[RESULT]] : tensor<2x3x3x5xf32>
}

// -----
// N-D x N-D with broadcast: A has more leading batch dims than B.
//   A: [4,3,M,K] x B: [3,K,N] -> [4,3,M,N]
// Result batch shape is [4,3].  A's leading dims already match, so it is
// folded directly to [12,M,K].  B is aligned to [1,3,K,N], tiled along
// axis 0 by 4 to [4,3,K,N], then folded to [12,K,N].

func.func @test_matmul_bcast_a_more(%arg0: tensor<4x3x2x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<4x3x2x6xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x3x2x5xf32>, tensor<3x5x6xf32>) -> tensor<4x3x2x6xf32>
  return %0 : tensor<4x3x2x6xf32>
// CHECK-LABEL: func.func @test_matmul_bcast_a_more
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<4x3x2x5xf32>, [[PARAM_1_:%.+]]: tensor<3x5x6xf32>) -> tensor<4x3x2x6xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[12, 2, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<4x3x2x5xf32>, !tosa.shape<3>) -> tensor<12x2x5xf32>
// CHECK-DAG:       [[ALIGN_B:%.+]] = tosa.const_shape {values = dense<[1, 3, 5, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[B_ALIGNED:%.+]] = tosa.reshape [[PARAM_1_]], [[ALIGN_B]] : (tensor<3x5x6xf32>, !tosa.shape<4>) -> tensor<1x3x5x6xf32>
// CHECK-DAG:       [[TILE_B:%.+]] = tosa.const_shape {values = dense<[4, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[B_TILED:%.+]] = tosa.tile [[B_ALIGNED]], [[TILE_B]] : (tensor<1x3x5x6xf32>, !tosa.shape<4>) -> tensor<4x3x5x6xf32>
// CHECK-DAG:       [[FOLD_B:%.+]] = tosa.const_shape {values = dense<[12, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[B_TILED]], [[FOLD_B]] : (tensor<4x3x5x6xf32>, !tosa.shape<3>) -> tensor<12x5x6xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], {{.*}} : (tensor<12x2x5xf32>, tensor<12x5x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x2x6xf32>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], {{.*}} : (tensor<12x2x6xf32>, !tosa.shape<4>) -> tensor<4x3x2x6xf32>
// CHECK:           return [[RESULT]] : tensor<4x3x2x6xf32>
}

// -----
// N-D x N-D with broadcast: B has more leading batch dims than A.
//   A: [3,M,K] x B: [4,3,K,N] -> [4,3,M,N]
// A is aligned to [1,3,M,K] and tiled along axis 0 by 4, then folded to
// [12,M,K].  B's leading dims already match and are folded directly.

func.func @test_matmul_bcast_b_more(%arg0: tensor<3x2x5xf32>, %arg1: tensor<4x3x5x6xf32>) -> tensor<4x3x2x6xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x2x5xf32>, tensor<4x3x5x6xf32>) -> tensor<4x3x2x6xf32>
  return %0 : tensor<4x3x2x6xf32>
// CHECK-LABEL: func.func @test_matmul_bcast_b_more
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<3x2x5xf32>, [[PARAM_1_:%.+]]: tensor<4x3x5x6xf32>) -> tensor<4x3x2x6xf32> {
// CHECK-DAG:       [[ALIGN_A:%.+]] = tosa.const_shape {values = dense<[1, 3, 2, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[A_ALIGNED:%.+]] = tosa.reshape [[PARAM_0_]], [[ALIGN_A]] : (tensor<3x2x5xf32>, !tosa.shape<4>) -> tensor<1x3x2x5xf32>
// CHECK-DAG:       [[TILE_A:%.+]] = tosa.const_shape {values = dense<[4, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[A_TILED:%.+]] = tosa.tile [[A_ALIGNED]], [[TILE_A]] : (tensor<1x3x2x5xf32>, !tosa.shape<4>) -> tensor<4x3x2x5xf32>
// CHECK-DAG:       [[FOLD_A:%.+]] = tosa.const_shape {values = dense<[12, 2, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[A_TILED]], [[FOLD_A]] : (tensor<4x3x2x5xf32>, !tosa.shape<3>) -> tensor<12x2x5xf32>
// CHECK-DAG:       [[FOLD_B:%.+]] = tosa.const_shape {values = dense<[12, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[FOLD_B]] : (tensor<4x3x5x6xf32>, !tosa.shape<3>) -> tensor<12x5x6xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], {{.*}} : (tensor<12x2x5xf32>, tensor<12x5x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x2x6xf32>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], {{.*}} : (tensor<12x2x6xf32>, !tosa.shape<4>) -> tensor<4x3x2x6xf32>
// CHECK:           return [[RESULT]] : tensor<4x3x2x6xf32>
}

// -----
// Per-axis broadcast: each operand has a 1 where the other has a non-1
// batch dim.
//   A: [4,1,M,K] x B: [1,5,K,N] -> [4,5,M,N]
// Both operands take the general per-axis tile path: A is tiled along
// axis 1 by 5, B along axis 0 by 4, then each is folded to a 3-D tensor
// with batch = 4*5 = 20.

func.func @test_matmul_bcast_per_axis(%arg0: tensor<4x1x2x5xf32>, %arg1: tensor<1x5x5x6xf32>) -> tensor<4x5x2x6xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x1x2x5xf32>, tensor<1x5x5x6xf32>) -> tensor<4x5x2x6xf32>
  return %0 : tensor<4x5x2x6xf32>
// CHECK-LABEL: func.func @test_matmul_bcast_per_axis
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<4x1x2x5xf32>, [[PARAM_1_:%.+]]: tensor<1x5x5x6xf32>) -> tensor<4x5x2x6xf32> {
// CHECK-DAG:       [[TILE_A:%.+]] = tosa.const_shape {values = dense<[1, 5, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[A_TILED:%.+]] = tosa.tile [[PARAM_0_]], [[TILE_A]] : (tensor<4x1x2x5xf32>, !tosa.shape<4>) -> tensor<4x5x2x5xf32>
// CHECK-DAG:       [[FOLD_A:%.+]] = tosa.const_shape {values = dense<[20, 2, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[A_TILED]], [[FOLD_A]] : (tensor<4x5x2x5xf32>, !tosa.shape<3>) -> tensor<20x2x5xf32>
// CHECK-DAG:       [[TILE_B:%.+]] = tosa.const_shape {values = dense<[4, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[B_TILED:%.+]] = tosa.tile [[PARAM_1_]], [[TILE_B]] : (tensor<1x5x5x6xf32>, !tosa.shape<4>) -> tensor<4x5x5x6xf32>
// CHECK-DAG:       [[FOLD_B:%.+]] = tosa.const_shape {values = dense<[20, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[B_TILED]], [[FOLD_B]] : (tensor<4x5x5x6xf32>, !tosa.shape<3>) -> tensor<20x5x6xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], {{.*}} : (tensor<20x2x5xf32>, tensor<20x5x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<20x2x6xf32>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], {{.*}} : (tensor<20x2x6xf32>, !tosa.shape<4>) -> tensor<4x5x2x6xf32>
// CHECK:           return [[RESULT]] : tensor<4x5x2x6xf32>
}

// -----
// Dynamic batch dim shared by both 3D operands.  Each operand already has
// shape [?, R, C], which is the fold target [batchProduct, R, C], so no
// input reshape is emitted.  The matmul output also matches the result type,
// so the trailing unfold reshape is elided.

func.func @test_matmul_dyn_batch(%arg0: tensor<?x3x4xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<?x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x3x4xf32>, tensor<?x4x5xf32>) -> tensor<?x3x5xf32>
  return %0 : tensor<?x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_dyn_batch
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<?x3x4xf32>, [[PARAM_1_:%.+]]: tensor<?x4x5xf32>) -> tensor<?x3x5xf32> {
// CHECK-NOT:   tosa.reshape
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[PARAM_0_]], [[PARAM_1_]], {{.*}} : (tensor<?x3x4xf32>, tensor<?x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3x5xf32>
// CHECK-NEXT:      return [[MATMUL]] : tensor<?x3x5xf32>
}

// -----
// 3D x 2D fast path with a dynamic leading batch dim on A.  A's leading
// dims (?, 3) fold into a single dynamic dim, encoded as -1 in the
// const_shape so tosa.reshape infers it from the input's total elements.
// The result type's leading dim is also dynamic and likewise becomes -1.

func.func @test_matmul_dyn_lead_3d_x_2d(%arg0: tensor<?x3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<?x3x5xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x3x4xf32>, tensor<4x5xf32>) -> tensor<?x3x5xf32>
  return %0 : tensor<?x3x5xf32>
// CHECK-LABEL: func.func @test_matmul_dyn_lead_3d_x_2d
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<?x3x4xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<?x3x5xf32> {
// CHECK-DAG:       [[SHAPE_A:%.+]] = tosa.const_shape {values = dense<[1, -1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[A3D:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_A]] : (tensor<?x3x4xf32>, !tosa.shape<3>) -> tensor<1x?x4xf32>
// CHECK-DAG:       [[SHAPE_B:%.+]] = tosa.const_shape {values = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[B3D:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_B]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK:           [[MATMUL:%.+]] = tosa.matmul [[A3D]], [[B3D]], {{.*}} : (tensor<1x?x4xf32>, tensor<1x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x?x5xf32>
// CHECK-DAG:       [[SHAPE_OUT:%.+]] = tosa.const_shape {values = dense<[-1, 3, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[RESULT:%.+]] = tosa.reshape [[MATMUL]], [[SHAPE_OUT]] : (tensor<1x?x5xf32>, !tosa.shape<3>) -> tensor<?x3x5xf32>
// CHECK:           return [[RESULT]] : tensor<?x3x5xf32>
}
