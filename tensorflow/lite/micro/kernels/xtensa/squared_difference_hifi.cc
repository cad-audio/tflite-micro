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

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

void* SquaredDifferenceInit(TfLiteContext* context, const char* buffer,
                            size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

void PrepareQuantized(
    const TfLiteQuantizationParams& input1_quantization_params,
    const TfLiteQuantizationParams& input2_quantization_params,
    const TfLiteQuantizationParams& output_quantization_params,
    const int left_shift, const int32_t quantized_activation_min,
    const int32_t quantized_activation_max, OpData* data) {
  data->arithmetic_params.input1_offset =
      -input1_quantization_params.zero_point;
  data->arithmetic_params.input2_offset =
      -input2_quantization_params.zero_point;
  data->arithmetic_params.output_offset = output_quantization_params.zero_point;
  data->arithmetic_params.left_shift = left_shift;
  const double twice_max_input_scale =
      2.0 * static_cast<double>(std::max(input1_quantization_params.scale,
                                         input2_quantization_params.scale));
  const double real_input1_multiplier =
      static_cast<double>(input1_quantization_params.scale) /
      twice_max_input_scale;
  double real_input2_multiplier =
      static_cast<double>(input2_quantization_params.scale) /
      twice_max_input_scale;
  const double real_output_multiplier =
      (twice_max_input_scale * twice_max_input_scale) /
      static_cast<double>((1 << data->arithmetic_params.left_shift * 2) *
                          output_quantization_params.scale);
  QuantizeMultiplierSmallerThanOneExp(
      real_input1_multiplier, &data->arithmetic_params.input1_multiplier,
      &data->arithmetic_params.input1_shift);
  QuantizeMultiplierSmallerThanOneExp(
      real_input2_multiplier, &data->arithmetic_params.input2_multiplier,
      &data->arithmetic_params.input2_shift);
  QuantizeMultiplier(real_output_multiplier,
                     &data->arithmetic_params.output_multiplier,
                     &data->arithmetic_params.output_shift);
  data->arithmetic_params.quantized_activation_min = quantized_activation_min;
  data->arithmetic_params.quantized_activation_max = quantized_activation_max;
}


TfLiteStatus SquaredDifferencePrepare(TfLiteContext* context,
                                      TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  data->requires_broadcast = false;

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  const TfLiteQuantizationParams& input1_quantization_params = input1->params;
  const TfLiteQuantizationParams& input2_quantization_params = input2->params;
  const TfLiteQuantizationParams& output_quantization_params = output->params;
  if (input1->type == kTfLiteInt8) {
    const int32_t integer_type_min = std::numeric_limits<int8_t>::min();
    const int32_t integer_type_max = std::numeric_limits<int8_t>::max();
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point <= integer_type_max);
    // leftshift = 7 is selected so that maximum shifted result 255^2 * (1 << (7
    // * 2 )) does not overflow signed 32-bit integer
    PrepareQuantized(input1_quantization_params, input2_quantization_params,
                     output_quantization_params, /*left_shift=*/7,
                     /*quantized_activation_min*/ integer_type_min,
                     /*quantized_activation_max*/ integer_type_max, data);
  } else if (input1->type == kTfLiteInt16) {
    const int32_t integer_type_min = std::numeric_limits<int16_t>::min();
    const int32_t integer_type_max = std::numeric_limits<int16_t>::max();
    TF_LITE_ENSURE(context, input1_quantization_params.zero_point == 0);
    TF_LITE_ENSURE(context, input2_quantization_params.zero_point == 0);
    TF_LITE_ENSURE(context, output_quantization_params.zero_point == 0);

    // leftshift = 0 as number is already 16-bit. so that maximum shifted result
    // 32767^2 * (1 << (0 * 2 ))
    PrepareQuantized(input1_quantization_params, input2_quantization_params,
                     output_quantization_params, /*left_shift=*/0,
                     /*quantized_activation_min*/ integer_type_min,
                     /*quantized_activation_max*/ integer_type_max, data);
  }

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

void EvalQuantizedSquaredDifferenceInt8Hifi(TfLiteContext* context,
                                            TfLiteNode* node,
                                            const OpData* data,
                                            const TfLiteEvalTensor* input1,
                                            const TfLiteEvalTensor* input2,
                                            TfLiteEvalTensor* output) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const ArithmeticParams& params = op_data->arithmetic_params;
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_squared_diff_broadcast_4D_asym8sxasym8s_asym8s(
      tflite::micro::GetTensorData<int8_t>(output),
      extended_output_shape.DimsData(), params.output_offset,
      params.output_shift, params.output_multiplier,
      params.quantized_activation_min,
      params.quantized_activation_max,
      tflite::micro::GetTensorData<int8_t>(input1),
      extended_input1_shape.DimsData(), params.input1_offset,
      params.input1_shift, params.input1_multiplier,
      tflite::micro::GetTensorData<int8_t>(input2),
      extended_input2_shape.DimsData(),
      params.input2_offset, params.input2_shift,
      params.input2_multiplier, params.left_shift);
  (void)err;
}

void EvalQuantizedSquaredDifferenceInt16Hifi(TfLiteContext* context,
                                            TfLiteNode* node,
                                            const OpData* data,
                                            const TfLiteEvalTensor* input1,
                                            const TfLiteEvalTensor* input2,
                                            TfLiteEvalTensor* output) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const ArithmeticParams& params = op_data->arithmetic_params;
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_squared_diff_broadcast_4D_sym16sxsym16s_sym16s(
      tflite::micro::GetTensorData<int16_t>(output),
      extended_output_shape.DimsData(),
      params.output_shift, params.output_multiplier,
      params.quantized_activation_min,
      params.quantized_activation_max,
      tflite::micro::GetTensorData<int16_t>(input1),
      extended_input1_shape.DimsData(),
      params.input1_shift, params.input1_multiplier,
      tflite::micro::GetTensorData<int16_t>(input2),
      extended_input2_shape.DimsData(),
      params.input2_shift, params.input2_multiplier,
      params.left_shift);
  (void)err;    
}
#endif // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

}  // namespace tflite
