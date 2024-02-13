/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/mul.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_mul.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

TfLiteStatus MulEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      EvalMulQuantizedHiFiInt8(context, node, data, input1, input2, output);
#else
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
#endif
  return kTfLiteOk;
}

TfLiteStatus MulEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      EvalMulQuantizedHiFiInt16(context, node, data, input1, input2, output);
#else
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
#endif
  return kTfLiteOk;
}

TfLiteStatus MulEvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

  tflite::ArithmeticParams op_params = {};
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = data->output_activation_max_f32;
  op_params.input1_offset = -data->input1_zero_point;
  op_params.input2_offset = -data->input2_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  bool need_broadcast;
  need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
  if (!need_broadcast) 
    EvalMulFloatHiFi(context, node, params, data, input1, input2, output);
  else
    EvalMulFloatReference(context, node, params, data, input1, input2, output);
#else
    EvalMulFloatReference(context, node, params, data, input1, input2, output);
#endif
  return kTfLiteOk;
}

TFLMRegistration Register_MUL_INT8() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)  
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalInt8);
#else
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalQuantizedReference);
#endif  
}

TFLMRegistration Register_MUL_INT16() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)  
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalInt16);
#else
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalQuantizedReference);
#endif   
}

TFLMRegistration Register_MUL_FLOAT32() {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))  
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalFloat32);
#else
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEvalFloatReference);
#endif   
}

}  // namespace tflite
