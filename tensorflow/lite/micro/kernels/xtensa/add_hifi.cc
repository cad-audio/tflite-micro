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
#include "tensorflow/lite/kernels/internal/reference/add.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_add.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) 
#if defined(INCLUDE_FLOAT_OPT)
TfLiteStatus AddEvalHifiFloat32(TfLiteContext* context, TfLiteNode* node,
                     TfLiteAddParams* params, const OpDataAdd* data,
                     const TfLiteEvalTensor* input1,
                     const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(data->output_activation_min_f32,
                      data->output_activation_max_f32, &op_params);
  if (data->requires_broadcast) {
    reference_ops::BroadcastAdd4DSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  } else {
  int err;
  const RuntimeShape& input1_shape = tflite::micro::GetTensorShape(input1);
  const RuntimeShape& input2_shape = tflite::micro::GetTensorShape(input2);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  err = xa_nn_elm_add_f32xf32_f32(tflite::micro::GetTensorData<float>(output),
                                  tflite::micro::GetTensorData<float>(input1),
                                  tflite::micro::GetTensorData<float>(input2),
                                  flat_size);

  TF_LITE_ENSURE(context, err == 0);

  err = xa_nn_vec_activation_min_max_f32_f32(
      tflite::micro::GetTensorData<float>(output),
      tflite::micro::GetTensorData<float>(output),
      data->output_activation_min_f32, data->output_activation_max_f32,
      flat_size);

  TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}
#endif

TfLiteStatus AddEvalHifiInt8(TfLiteContext* context, TfLiteNode* node,
                     TfLiteAddParams* params, const OpDataAdd* data,
                     const TfLiteEvalTensor* input1,
                     const TfLiteEvalTensor* input2, TfLiteEvalTensor* output, tflite::ArithmeticParams op_params) {                   
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_add_broadcast_4D_asym8sxasym8s_asym8s(
      tflite::micro::GetTensorData<int8_t>(output),
      extended_output_shape.DimsData(), op_params.output_offset,
      op_params.output_shift, op_params.output_multiplier,
      op_params.quantized_activation_min,
      op_params.quantized_activation_max,
      tflite::micro::GetTensorData<int8_t>(input1),
      extended_input1_shape.DimsData(), op_params.input1_offset,
      op_params.input1_shift, op_params.input1_multiplier,
      tflite::micro::GetTensorData<int8_t>(input2),
      extended_input2_shape.DimsData(),
      op_params.input2_offset, op_params.input2_shift,
      op_params.input2_multiplier, op_params.left_shift);

  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}

TfLiteStatus AddEvalHifiInt16(TfLiteContext* context, TfLiteNode* node,
                     TfLiteAddParams* params, const OpDataAdd* data,
                     const TfLiteEvalTensor* input1,
                     const TfLiteEvalTensor* input2, TfLiteEvalTensor* output, tflite::ArithmeticParams op_params) {
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_add_broadcast_4D_asym16sxasym16s_asym16s(
      tflite::micro::GetTensorData<int16_t>(output),
      extended_output_shape.DimsData(), op_params.output_offset,
      op_params.output_shift, op_params.output_multiplier,
      op_params.quantized_activation_min,
      op_params.quantized_activation_max,
      tflite::micro::GetTensorData<int16_t>(input1),
      extended_input1_shape.DimsData(), op_params.input1_offset,
      op_params.input1_shift, op_params.input1_multiplier,
      tflite::micro::GetTensorData<int16_t>(input2),
      extended_input2_shape.DimsData(), op_params.input2_offset,
      op_params.input2_shift, op_params.input2_multiplier,
      op_params.left_shift);

  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}
#endif
}  // namespace tflite
