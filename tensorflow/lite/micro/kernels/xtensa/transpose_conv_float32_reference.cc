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
#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_transpose_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

TfLiteStatus TransposeConvReferenceEvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));  
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  ConvParams op_params = data.params;
  CalculateActivationRange(params.activation,
                            &op_params.float_activation_min,
                            &op_params.float_activation_max);

  reference_ops::TransposeConv(
     op_params, tflite::micro::GetTensorShape(input),
     tflite::micro::GetTensorData<float>(input),
     tflite::micro::GetTensorShape(filter),
     tflite::micro::GetTensorData<float>(filter),
     tflite::micro::GetTensorShape(bias),
     tflite::micro::GetOptionalTensorData<float>(bias),
     tflite::micro::GetTensorShape(output),
     tflite::micro::GetTensorData<float>(output),
     tflite::micro::GetTensorShape(nullptr), nullptr);

  return kTfLiteOk;
}

TFLMRegistration Register_TRANSPOSE_CONV_FLOAT32REF() {
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepare, TransposeConvReferenceEvalFloat32);
}

}  // namespace tflite
