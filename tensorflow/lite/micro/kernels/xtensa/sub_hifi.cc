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
#include "tensorflow/lite/micro/kernels/sub.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)  
TfLiteStatus EvalSubQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(output));
  const int* input1_dims = extended_input1_shape.DimsData();
  const int* input2_dims = extended_input2_shape.DimsData();
  const int* output_dims = extended_output_shape.DimsData();
  // TODO(b/259724572): Refactor the following block of code.
  int b;
  int inp1_off = 0;
  int inp2_off = 0;
  int out_off;
  out_off =
      output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];
  if (input1_dims[0] > 1) {
    inp1_off =
        input1_dims[1] * input1_dims[2] * input1_dims[3] * input1_dims[4];
  }
  if (input2_dims[0] > 1) {
    inp2_off =
        input2_dims[1] * input2_dims[2] * input2_dims[3] * input2_dims[4];
  }

  for (b = 0; b < output_dims[0]; b++) {
    err = xa_nn_elm_sub_broadcast_4D_asym8sxasym8s_asym8s(
        tflite::micro::GetTensorData<int8_t>(output) + b * out_off,
        output_dims + 1, op_params.output_offset, op_params.output_shift,
        op_params.output_multiplier, op_params.quantized_activation_min,
        op_params.quantized_activation_max,
        tflite::micro::GetTensorData<int8_t>(input1) + b * inp1_off,
        input1_dims + 1, op_params.input1_offset, op_params.input1_shift,
        op_params.input1_multiplier,
        tflite::micro::GetTensorData<int8_t>(input2) + b * inp2_off,
        input2_dims + 1, op_params.input2_offset, op_params.input2_shift,
        op_params.input2_multiplier, op_params.left_shift);

    TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalSubQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(output));
  const int* input1_dims = extended_input1_shape.DimsData();
  const int* input2_dims = extended_input2_shape.DimsData();
  const int* output_dims = extended_output_shape.DimsData();
  int b;
  int inp1_off = 0;
  int inp2_off = 0;
  int out_off;
  out_off =
      output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];
  if (input1_dims[0] > 1) {
    inp1_off =
        input1_dims[1] * input1_dims[2] * input1_dims[3] * input1_dims[4];
  }
  if (input2_dims[0] > 1) {
    inp2_off =
        input2_dims[1] * input2_dims[2] * input2_dims[3] * input2_dims[4];
  }

  for (b = 0; b < output_dims[0]; b++) {
    err = xa_nn_elm_sub_broadcast_4D_asym16sxasym16s_asym16s(
        tflite::micro::GetTensorData<int16_t>(output) + b * out_off,
        output_dims + 1, op_params.output_offset, op_params.output_shift,
        op_params.output_multiplier, op_params.quantized_activation_min,
        op_params.quantized_activation_max,
        tflite::micro::GetTensorData<int16_t>(input1) + b * inp1_off,
        input1_dims + 1, op_params.input1_offset, op_params.input1_shift,
        op_params.input1_multiplier,
        tflite::micro::GetTensorData<int16_t>(input2) + b * inp2_off,
        input2_dims + 1, op_params.input2_offset, op_params.input2_shift,
        op_params.input2_multiplier, op_params.left_shift);

    TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}
#endif
}  // namespace tflite
