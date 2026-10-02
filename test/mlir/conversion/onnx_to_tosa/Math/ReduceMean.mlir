// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[V0:.*]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[V1:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[V2:.*]] = tosa.reshape %[[V0]], %[[V1]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[V3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[V4:.*]] = tosa.mul %[[ARG0]], %[[V2]], %[[V3]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[V5:.*]] = tosa.reduce_sum %[[V4]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[V6:.*]] = tosa.reduce_sum %[[V5]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[V6]] : tensor<2x5x1x1xf32>
}
// -----

func.func @reduce_mean_no_axes_attr(%arg0: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) : (tensor<2x5x9x11xf32>, none) -> tensor<1x1x1x1xf32>
return %0 : tensor<1x1x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean_no_axes_attr(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:           %[[V0:.*]] = "tosa.const"() <{values = dense<0.00101010106> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[V1:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[V2:.*]] = tosa.reshape %[[V0]], %[[V1]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[V3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[V4:.*]] = tosa.mul %[[ARG0]], %[[V2]], %[[V3]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[V5:.*]] = tosa.reduce_sum %[[V4]] {axis = 0 : i32} : (tensor<2x5x9x11xf32>) -> tensor<1x5x9x11xf32>
// CHECK:           %[[V6:.*]] = tosa.reduce_sum %[[V5]] {axis = 1 : i32} : (tensor<1x5x9x11xf32>) -> tensor<1x1x9x11xf32>
// CHECK:           %[[V7:.*]] = tosa.reduce_sum %[[V6]] {axis = 2 : i32} : (tensor<1x1x9x11xf32>) -> tensor<1x1x1x11xf32>
// CHECK:           %[[V8:.*]] = tosa.reduce_sum %[[V7]] {axis = 3 : i32} : (tensor<1x1x1x11xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           return %[[V8]] : tensor<1x1x1x1xf32>
}
// -----

func.func @reduce_mean_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
// CHECK-LABEL:   func.func @reduce_mean_keepdims_false(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
// CHECK:           %[[V0:.*]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[V1:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[V2:.*]] = tosa.reshape %[[V0]], %[[V1]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[V3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[V4:.*]] = tosa.mul %[[ARG0]], %[[V2]], %[[V3]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[V5:.*]] = tosa.reduce_sum %[[V4]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[V6:.*]] = tosa.reduce_sum %[[V5]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[V7:.*]] = tosa.const_shape  {values = dense<[2, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[V8:.*]] = tosa.reshape %[[V6]], %[[V7]] : (tensor<2x5x1x1xf32>, !tosa.shape<2>) -> tensor<2x5xf32>
// CHECK:           return %[[V8]] : tensor<2x5xf32>
}
// -----

func.func @reduce_mean_noop_with_emtpy_axes_one(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean_noop_with_emtpy_axes_one(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[V0:.*]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[V1:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[V2:.*]] = tosa.reshape %[[V0]], %[[V1]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[V3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[V4:.*]] = tosa.mul %[[ARG0]], %[[V2]], %[[V3]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[V5:.*]] = tosa.reduce_sum %[[V4]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[V6:.*]] = tosa.reduce_sum %[[V5]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[V6]] : tensor<2x5x1x1xf32>
}
// -----

func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, none) ->  tensor<2x5x9x11xf32>
return %0 : tensor<2x5x9x11xf32>
// CHECK-LABEL:   func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
// CHECK:           %[[V0:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[V1:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[V2:.*]] = tosa.reshape %[[V0]], %[[V1]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[V3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[V4:.*]] = tosa.mul %[[ARG0]], %[[V2]], %[[V3]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           %[[V5:.*]] = tosa.identity %[[V4]] : (tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32>
// CHECK:           return %[[V5]] : tensor<2x5x9x11xf32>
}