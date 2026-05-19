// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// -----
// Negative axis is normalized. With before=3 (axis=1), indices are tiled.
func.func @compress_const_neg_axis(%arg0: tensor<3x4xf32>) -> tensor<3x?xf32> {
  %cond = "onnx.Constant"() {value = dense<[false, true, true, false]> : tensor<4xi1>} : () -> tensor<4xi1>
  %0 = "onnx.Compress"(%arg0, %cond) {axis = -1 : si64} : (tensor<3x4xf32>, tensor<4xi1>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
// CHECK-LABEL:  func @compress_const_neg_axis
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x8x1xi32>, tensor<1x4xi32>, tensor<1x4x1xi32>) -> tensor<1x8x1xi32>
// CHECK:       tosa.tile %{{.*}} : (tensor<1x4xi32>, !tosa.shape<2>) -> tensor<3x4xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<3x4x1xf32>, tensor<3x4xi32>) -> tensor<3x4x1xf32>
// CHECK:       tosa.mul
// CHECK:       tensor.cast %{{.*}} : tensor<3x4xf32> to tensor<3x?xf32>
}

// -----
// No axis attribute => input flattened first.
func.func @compress_const_noaxis(%arg0: tensor<3x2xf32>) -> tensor<?xf32> {
  %cond = "onnx.Constant"() {value = dense<[true, false, false, true, true, false]> : tensor<6xi1>} : () -> tensor<6xi1>
  %0 = "onnx.Compress"(%arg0, %cond) : (tensor<3x2xf32>, tensor<6xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
// CHECK-LABEL:  func @compress_const_noaxis
// CHECK:       tosa.reshape{{.*}}: (tensor<3x2xf32>, !tosa.shape<1>) -> tensor<6xf32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x12x1xi32>, tensor<1x6xi32>, tensor<1x6x1xi32>) -> tensor<1x12x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x6x1xf32>, tensor<1x6xi32>) -> tensor<1x6x1xf32>
// CHECK:       tensor.cast %{{.*}} : tensor<6xf32> to tensor<?xf32>
}

// -----
// Condition length < axis dim: short condition is zero-padded so the
// Hillis-Steele scan operates over the full K (the padded entries contribute
// nothing) and the post-mask multiply zeroes out the corresponding slots.
func.func @compress_const_short_cond(%arg0: tensor<5x2xf32>) -> tensor<?x2xf32> {
  %cond = "onnx.Constant"() {value = dense<[true, false, true]> : tensor<3xi1>} : () -> tensor<3xi1>
  %0 = "onnx.Compress"(%arg0, %cond) {axis = 0 : si64} : (tensor<5x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// CHECK-LABEL:  func @compress_const_short_cond
// CHECK:       tosa.cast %{{.*}} : (tensor<3xi1>) -> tensor<3xf32>
// CHECK:       tosa.pad %{{.*}} : (tensor<3xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<5xf32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x10x1xi32>, tensor<1x5xi32>, tensor<1x5x1xi32>) -> tensor<1x10x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x5x2xf32>, tensor<1x5xi32>) -> tensor<1x5x2xf32>
// CHECK:       tensor.cast %{{.*}} : tensor<5x2xf32> to tensor<?x2xf32>
}

// -----
// Dynamic condition with statically known length. Lowering uses Hillis-
// Steele cumsum, builds an inverse permutation via tosa.scatter into a
// double-sized buffer (sentinel slots absorb the false-entry sentinels),
// gathers from input, and zero-masks the padded slots.
func.func @compress_dynamic_axis0(%arg0: tensor<3x2xf32>, %cond: tensor<3xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %cond) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// CHECK-LABEL:  func @compress_dynamic_axis0
// CHECK:       tosa.cast %{{.*}} : (tensor<3xi1>) -> tensor<3xf32>
// CHECK:       tosa.add
// CHECK:       tosa.sub
// CHECK:       tosa.cast %{{.*}} : (tensor<3xf32>) -> tensor<3xi32>
// CHECK:       tosa.logical_not
// CHECK:       "tosa.const"(){{.*}}dense<[3, 4, 5]> : tensor<3xi32>
// CHECK:       tosa.select %{{.*}} : (tensor<3xi1>, tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x6x1xi32>, tensor<1x3xi32>, tensor<1x3x1xi32>) -> tensor<1x6x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x3x2xf32>, tensor<1x3xi32>) -> tensor<1x3x2xf32>
// CHECK:       tosa.reduce_sum
// CHECK:       tosa.greater
// CHECK:       tosa.mul
// CHECK:       tensor.cast %{{.*}} : tensor<3x2xf32> to tensor<?x2xf32>
}

// -----
// Dynamic condition with the condition length itself dynamic: not
// supported, op is left as-is.
func.func @compress_unsupported_dyn_cond_dim(%arg0: tensor<3x2xf32>, %cond: tensor<?xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %cond) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<?xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// CHECK-LABEL:  func @compress_unsupported_dyn_cond_dim
// CHECK:       "onnx.Compress"
// CHECK-NOT:   tosa.gather
// CHECK-NOT:   tosa.scatter
}
