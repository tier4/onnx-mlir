// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// -----
// 1D NonZero. Mask -> Hillis-Steele cumsum -> sentinel select -> scatter to
// a 2N buffer -> slice first N -> gather rows of the flat-to-multi-index
// table I[N, rank]. Padded slots gather I[0, :] = zeros, matching the prior
// behavior of the selection-matrix lowering.
func.func @nonzero_1d(%arg0: tensor<5xf32>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<5xf32>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
// CHECK-LABEL:  func @nonzero_1d
// CHECK:       tosa.equal %{{.*}} : (tensor<5xf32>, tensor<1xf32>) -> tensor<5xi1>
// CHECK:       tosa.logical_not
// CHECK:       tosa.cast %{{.*}} : (tensor<5xi1>) -> tensor<5xf32>
// CHECK:       tosa.add
// CHECK:       tosa.sub
// CHECK:       tosa.cast %{{.*}} : (tensor<5xf32>) -> tensor<5xi32>
// CHECK:       "tosa.const"(){{.*}}dense<[5, 6, 7, 8, 9]> : tensor<5xi32>
// CHECK:       tosa.select %{{.*}} : (tensor<5xi1>, tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x10x1xi32>, tensor<1x5xi32>, tensor<1x5x1xi32>) -> tensor<1x10x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x5x1xf32>, tensor<1x5xi32>) -> tensor<1x5x1xf32>
// CHECK:       tosa.transpose %{{.*}} {perms = array<i32: 1, 0>} : (tensor<5x1xf32>) -> tensor<1x5xf32>
// CHECK:       tosa.cast %{{.*}} : (tensor<1x5xf32>) -> tensor<1x5xi64>
// CHECK:       tensor.cast %{{.*}} : tensor<1x5xi64> to tensor<1x?xi64>
}

// -----
// 2D NonZero. Index table I has 2 columns; gather returns [1, N, 2] which
// is reshaped to [N, 2] and transposed to [2, N].
func.func @nonzero_2d(%arg0: tensor<3x4xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<3x4xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
// CHECK-LABEL:  func @nonzero_2d
// CHECK:       tosa.reshape %arg0, %{{.*}}: (tensor<3x4xf32>, !tosa.shape<1>) -> tensor<12xf32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x24x1xi32>, tensor<1x12xi32>, tensor<1x12x1xi32>) -> tensor<1x24x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x12x2xf32>, tensor<1x12xi32>) -> tensor<1x12x2xf32>
// CHECK:       tosa.transpose %{{.*}} : (tensor<12x2xf32>) -> tensor<2x12xf32>
// CHECK:       tosa.cast %{{.*}} : (tensor<2x12xf32>) -> tensor<2x12xi64>
}

// -----
// Integer input is cast to f32 first so the same scan path applies.
func.func @nonzero_int(%arg0: tensor<2x3xi32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<2x3xi32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
// CHECK-LABEL:  func @nonzero_int
// CHECK:       tosa.cast %{{.*}} : (tensor<6xi32>) -> tensor<6xf32>
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x12x1xi32>, tensor<1x6xi32>, tensor<1x6x1xi32>) -> tensor<1x12x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x6x2xf32>, tensor<1x6xi32>) -> tensor<1x6x2xf32>
// CHECK:       tensor.cast %{{.*}} : tensor<2x6xi64> to tensor<2x?xi64>
}

// -----
// Large input that previously hit the N^2 SmallVector overflow guard.
func.func @nonzero_large(%arg0: tensor<256x256xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<256x256xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
// CHECK-LABEL:  func @nonzero_large
// CHECK:       tosa.scatter %{{.*}} : (tensor<1x131072x1xi32>, tensor<1x65536xi32>, tensor<1x65536x1xi32>) -> tensor<1x131072x1xi32>
// CHECK:       tosa.gather %{{.*}} : (tensor<1x65536x2xf32>, tensor<1x65536xi32>) -> tensor<1x65536x2xf32>
}

// -----
// Dynamic input shape: not supported, op left as-is.
func.func @nonzero_dynamic(%arg0: tensor<?x4xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<?x4xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
// CHECK-LABEL:  func @nonzero_dynamic
// CHECK:       "onnx.NonZero"
// CHECK-NOT:   tosa.scatter
}
