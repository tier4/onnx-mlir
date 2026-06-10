# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  DialectBuilder.cpp
  ONNXToTOSALegalizeUtils.cpp

  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/MatMul.cpp
  Math/Softmax.cpp
  Math/ReduceMean.cpp
  Math/Reduce.cpp
  Math/Conv2D.cpp
  NN/MaxPoolSingleOut.cpp
  NN/AveragePool.cpp
  NN/Normalization.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/Expand.cpp
  Tensor/Gather.cpp
  Tensor/Scatter.cpp
  Tensor/Reshape.cpp
  Tensor/Resize.cpp
  Tensor/Slice.cpp
  Tensor/Split.cpp
  Tensor/SqueezeUnsqueeze.cpp
  Tensor/Transpose.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
