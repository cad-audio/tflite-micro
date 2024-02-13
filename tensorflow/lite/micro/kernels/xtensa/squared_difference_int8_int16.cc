/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_squared_difference.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
TfLiteStatus SquaredDifferenceEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  EvalQuantizedSquaredDifferenceInt8Hifi(context, node, data, input1, input2,
                                        output);

  return kTfLiteOk;
}

TfLiteStatus SquaredDifferenceEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  EvalQuantizedSquaredDifferenceInt16Hifi(context, node, data, input1, input2,
                                        output);
  return kTfLiteOk;
}
#endif
}  // namespace

TFLMRegistration Register_SQUARED_DIFFERENCE_INT8() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)  
  return tflite::micro::RegisterOp(
      SquaredDifferenceInit, SquaredDifferencePrepare, SquaredDifferenceEvalInt8);
#else
  return tflite::micro::RegisterOp(
      SquaredDifferenceInit, SquaredDifferencePrepare, SquaredDifferenceEvalInt8Reference);
#endif      
}

TFLMRegistration Register_SQUARED_DIFFERENCE_INT16() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)  
  return tflite::micro::RegisterOp(
      SquaredDifferenceInit, SquaredDifferencePrepare, SquaredDifferenceEvalInt16);
#else
  return tflite::micro::RegisterOp(
      SquaredDifferenceInit, SquaredDifferencePrepare, SquaredDifferenceEvalInt16Reference);
#endif      
}

}  // namespace tflite
