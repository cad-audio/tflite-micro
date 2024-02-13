/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SQUARED_DIFFERENCE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SQUARED_DIFFERENCE_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/squared_difference.h"

namespace tflite {
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;
  ArithmeticParams arithmetic_params;
};

template <typename T>
T SquaredDifference(T input1, T input2) {
  const T difference = input1 - input2;
  return difference * difference;
}

template <typename T>
T SquaredDifference(T x, T y, const ArithmeticParams& params) {
  const int32_t input1_val = params.input1_offset + x;
  const int32_t input2_val = params.input2_offset + y;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  const int32_t scaled_input2_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input2_val, params.input2_multiplier, params.input2_shift);
  const int32_t raw_diff = scaled_input1_val - scaled_input2_val;

  // Max of this is 32767^2 * (1 << 0), so won't overflow 32 bits.
  const int32_t squared_raw_diff = raw_diff * raw_diff;
  const int32_t raw_output =
      MultiplyByQuantizedMultiplier(squared_raw_diff, params.output_multiplier,
                                    params.output_shift) +
      params.output_offset;
  const int32_t clamped_output =
      std::min(params.quantized_activation_max,
               std::max(params.quantized_activation_min, raw_output));
  return static_cast<T>(clamped_output);
}

template <typename T>
void EvalQuantizedSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                                    const OpData* data,
                                    const TfLiteEvalTensor* input1,
                                    const TfLiteEvalTensor* input2,
                                    TfLiteEvalTensor* output) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  if (data->requires_broadcast) {
    reference_integer_ops::BroadcastBinaryFunction4DSlow(
        op_data->arithmetic_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<T>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<T>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<T>(output),
        reference_integer_ops::CheckArithmeticParams, SquaredDifference);
  } else {
    const int flat_size = tflite::micro::GetTensorShape(input1).FlatSize();
    reference_integer_ops::ElementWise(
        flat_size, op_data->arithmetic_params,
        tflite::micro::GetTensorData<T>(input1),
        tflite::micro::GetTensorData<T>(input2),
        tflite::micro::GetTensorData<T>(output),
        reference_integer_ops::CheckArithmeticParams, SquaredDifference);
  }
}

template <typename T>
void EvalSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                           const OpData* data, const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output) {
  if (data->requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<T>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<T>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<T>(output), SquaredDifference<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<T>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<T>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<T>(output), SquaredDifference<T>);
  }
}

TfLiteStatus SquaredDifferenceEvalInt8Reference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SquaredDifferenceEvalInt16Reference(TfLiteContext* context, TfLiteNode* node);
void* SquaredDifferenceInit(TfLiteContext* context, const char* buffer, size_t length);
TfLiteStatus SquaredDifferencePrepare(TfLiteContext* context, TfLiteNode* node);

#if defined(XTENSA)
void EvalQuantizedSquaredDifferenceInt8Hifi(TfLiteContext* context,
                                            TfLiteNode* node,
                                            const OpData* data,
                                            const TfLiteEvalTensor* input1,
                                            const TfLiteEvalTensor* input2,
                                            TfLiteEvalTensor* output);

void EvalQuantizedSquaredDifferenceInt16Hifi(TfLiteContext* context,
                                            TfLiteNode* node,
                                            const OpData* data,
                                            const TfLiteEvalTensor* input1,
                                            const TfLiteEvalTensor* input2,
                                            TfLiteEvalTensor* output);
#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SQUARED_DIFFERENCE_H_
