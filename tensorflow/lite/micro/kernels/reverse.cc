/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>
#include "stdio.h"
#include <algorithm>
#include <array>
#include <cstring>

#include "tensorflow/lite/core/c/common.h"
//#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
//#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/reference/reverse.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

#if 0
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TF_LITE_ENSURE_EQ(context, NumDimensions(axis), 1);
  TF_LITE_ENSURE(context, NumDimensions(input) <= 8);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16 && input->type != kTfLiteInt64 &&
      input->type != kTfLiteBool) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (axis->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Axis Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  return context->ResizeTensor(context, output, output_shape);
}
#else

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
	

  //const TfLiteTensor* input;
  //TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

  TfLiteTensor* axis =
      micro_context->AllocateTempInputTensor(node, kAxisTensor);
  TF_LITE_ENSURE(context, axis != nullptr);

  //const TfLiteTensor* axis;
  //TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TF_LITE_ENSURE_EQ(context, NumDimensions(axis), 1);
  TF_LITE_ENSURE(context, NumDimensions(input) <= 8);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16 && input->type != kTfLiteInt64 &&
      input->type != kTfLiteBool) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (axis->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Axis Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  //TfLiteTensor* output;
  //TF_LITE_ENSURE_OK(context,
  //                  GetOutputSafe(context, node, kOutputTensor, &output));
  //TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);


  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
  //return (TfLiteStatus) 1 ;
}

#endif

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  //const TfLiteTensor* input;
  //TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  //const TfLiteTensor* axis_tensor;
  //TF_LITE_ENSURE_OK(context,
  //                  GetInputSafe(context, node, kAxisTensor, &axis_tensor));
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis_tensor =
      tflite::micro::GetEvalInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, axis_tensor->type, kTfLiteInt32);
  //const int num_axes = NumElements(axis_tensor);
	const int num_axes = static_cast<int>(ElementCount(*axis_tensor->dims));
  TF_LITE_ENSURE(context, num_axes <= 8);

  std::array<int32_t, 8> axes;
  memcpy(axes.data(), tflite::micro::GetTensorData<int32_t>(axis_tensor),
         num_axes * sizeof(int32_t));
  //const int rank = NumDimensions(input);
	const int rank = input->dims->size;
  for (int i = 0; i < num_axes; ++i) {
    if (axes[i] < 0) {
      axes[i] += rank;
    }
    TF_LITE_ENSURE(context, axes[i] >= 0 && axes[i] < rank);
  }

  std::sort(axes.begin(), axes.begin() + num_axes);

  bool is_contiguous = true;
  for (int i = 1; i < num_axes; ++i) {
    if (axes[i - 1] + 1 != axes[i]) {
      is_contiguous = false;
      break;
    }
  }
  if (!is_contiguous) {
    TF_LITE_KERNEL_LOG(context, "Non-contiguous `axes` not supported");
    return kTfLiteError;
  }

  //TfLiteTensor* output;
  //TF_LITE_ENSURE_OK(context,
  //                  GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  switch (output->type) {
    case kTfLiteFloat32: {
      reference_ops::Reverse<float>(axes, num_axes, tflite::micro::GetTensorShape(input),
                                    tflite::micro::GetTensorData<float>(input),
                                    tflite::micro::GetTensorData<float>(output));
      break;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      reference_ops::Reverse<uint8_t>(axes, num_axes, tflite::micro::GetTensorShape(input),
                                      tflite::micro::GetTensorData<uint8_t>(input),
                                      tflite::micro::GetTensorData<uint8_t>(output));
      break;
    }
    case kTfLiteInt16: {
      reference_ops::Reverse<int16_t>(axes, num_axes, tflite::micro::GetTensorShape(input),
                                      tflite::micro::GetTensorData<int16_t>(input),
                                      tflite::micro::GetTensorData<int16_t>(output));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by reverse.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_REVERSE_V2() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
