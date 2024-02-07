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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_transpose_conv.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& op_data = *(reinterpret_cast<OpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data));
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
          
  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  return TransposeConvEvalHifiInt8(context, node, params, op_data, input, filter, bias,
                           output); 
}

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& op_data = *(reinterpret_cast<OpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data));
  
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  return TransposeConvEvalHifiInt16(context, node, params, op_data, input, filter, bias,
                           output);      
}

#if defined(INCLUDE_FLOAT_OPT)
TfLiteStatus EvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& op_data = *(reinterpret_cast<OpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data));
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  return TransposeConvEvalHifiFloat32(context, node, params, op_data, input, filter, bias,
                           output);
}
#endif // #if defined(INCLUDE_FLOAT_OPT)
#endif // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}  // namespace

TFLMRegistration Register_TRANSPOSE_CONV_INT8() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepareXtensa, EvalInt8);
#else
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepare, TransposeConvReferenceEvalInt8);
#endif  
}

TFLMRegistration Register_TRANSPOSE_CONV_INT16() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepareXtensa, EvalInt16);
#else
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepare, TransposeConvReferenceEvalInt8);
#endif    
}

TFLMRegistration Register_TRANSPOSE_CONV_FLOAT32() {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))    
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepareXtensa, EvalFloat32);
#else
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepare, TransposeConvReferenceEvalFloat32);
#endif  
}

}  // namespace tflite
