/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

TfLiteStatus DepthwiseConvReferenceEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

   reference_integer_ops::DepthwiseConvPerChannel(
       DepthwiseConvParamsQuantized(params, data),
       data.per_channel_output_multiplier, data.per_channel_output_shift,
       tflite::micro::GetTensorShape(input),
       tflite::micro::GetTensorData<int16_t>(input),
       tflite::micro::GetTensorShape(filter),
       tflite::micro::GetTensorData<int8_t>(filter),
       tflite::micro::GetTensorShape(bias),
       tflite::micro::GetOptionalTensorData<int64_t>(bias),
       tflite::micro::GetTensorShape(output),
       tflite::micro::GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

// TODO(b/189981943): This variant can be used for a smaller binary
// since the optimized conv implementation currently adds a lot to
// the binary size (~30KB to text section).
TFLMRegistration Register_DEPTHWISE_CONV_2D_INT16REF() {
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare,
                                   DepthwiseConvReferenceEvalInt16);
}

}  // namespace tflite
