// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// The NonZero compress/compute/scatter idiom is rewritten to compute on all
// rows and select the result with the mask, so that no op with a
// data-dependent result shape remains.
func.func @test_nonzero_compress_scatter(%mask: tensor<4xi1>, %data: tensor<4x3xf32>, %w: tensor<3x8xf32>) -> tensor<4x8xf32> {
  %init = onnx.Constant dense<0.0> : tensor<4x8xf32>
  %shape2 = onnx.Constant dense<[4, 8]> : tensor<2xi64>
  %axes1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %zero = onnx.Constant dense<[0]> : tensor<1xi64>
  %one = onnx.Constant dense<[1]> : tensor<1xi64>
  %minus1 = onnx.Constant dense<[-1]> : tensor<1xi64>
  %nz1 = "onnx.NonZero"(%mask) : (tensor<4xi1>) -> tensor<1x?xi64>
  %idx1 = "onnx.Transpose"(%nz1) {perm = [1, 0]} : (tensor<1x?xi64>) -> tensor<?x1xi64>
  %rows = "onnx.GatherND"(%data, %idx1) {batch_dims = 0 : si64} : (tensor<4x3xf32>, tensor<?x1xi64>) -> tensor<?x3xf32>
  %x = "onnx.MatMul"(%rows, %w) : (tensor<?x3xf32>, tensor<3x8xf32>) -> tensor<?x8xf32>
  %u = "onnx.Unsqueeze"(%mask, %axes1) : (tensor<4xi1>, tensor<1xi64>) -> tensor<4x1xi1>
  %m2 = "onnx.Expand"(%u, %shape2) : (tensor<4x1xi1>, tensor<2xi64>) -> tensor<4x8xi1>
  %nz2 = "onnx.NonZero"(%m2) : (tensor<4x8xi1>) -> tensor<2x?xi64>
  %idx2 = "onnx.Transpose"(%nz2) {perm = [1, 0]} : (tensor<2x?xi64>) -> tensor<?x2xi64>
  %flat = "onnx.Reshape"(%x, %minus1) {allowzero = 0 : si64} : (tensor<?x8xf32>, tensor<1xi64>) -> tensor<?xf32>
  %n = "onnx.Dim"(%nz2) {axis = 1 : si64} : (tensor<2x?xi64>) -> tensor<1xi64>
  %upd = "onnx.Slice"(%flat, %zero, %n, %zero, %one) : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  %out = "onnx.ScatterND"(%init, %idx2, %upd) {reduction = "none"} : (tensor<4x8xf32>, tensor<?x2xi64>, tensor<?xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
// CHECK-LABEL:  func.func @test_nonzero_compress_scatter
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xi1>, [[PARAM_1_:%.+]]: tensor<4x3xf32>, [[PARAM_2_:%.+]]: tensor<3x8xf32>) -> tensor<4x8xf32> {
// CHECK-NOT:       onnx.NonZero
// CHECK-NOT:       onnx.GatherND
// CHECK-NOT:       onnx.ScatterND
// CHECK-DAG:       [[INIT_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4x8xf32>}> : () -> tensor<4x8xf32>
// CHECK:           [[MATMUL_:%.+]] = tosa.matmul {{.*}} -> tensor<1x4x8xf32>
// CHECK:           [[TILE_:%.+]] = tosa.tile {{.*}} -> tensor<4x8xi1>
// CHECK:           [[SELECT_:%.+]] = tosa.select [[TILE_]], {{.*}} : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:           return [[SELECT_]] : tensor<4x8xf32>
}
