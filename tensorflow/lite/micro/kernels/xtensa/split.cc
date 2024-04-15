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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

namespace {

#if defined(HIFI5) || defined(HIFI4)
constexpr int kMaxOutputNum = 10;  // Maximum number of output tensors
// Equal to kMaxSmallSize from tensorflow/lite/kernels/internal/runtime_shape.h
constexpr int kMaxDims = 6;  // Maximum number of dimensions

// Gets Dims from a list of tensors, equivalent to Int8, for higher
// datatypes multiply last dimension by bytes_per_element
// Also gets all output tensor data pointers
inline void GetAllOutputTensorDimsDataInt8(const TfLiteContext* context,
                                           const TfLiteNode* node,
                                           int8_t* all_data[kMaxOutputNum],
                                           int32_t all_shapes[kMaxOutputNum][kMaxDims],
                                           int32_t* num_valid_outputs,
                                           int32_t axis,
                                           int32_t bytes_per_element) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  int32_t num_outputs = node->outputs->size;
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteEvalTensor* t = tflite::micro::GetEvalOutput(context, node, i);
    int num_dims = t->dims->size;
    for(int j = 0; j < num_dims - 1; j++) {
      all_shapes[i][j] = t->dims->data[j];
    }
    all_shapes[i][num_dims - 1] = t->dims->data[num_dims - 1] * bytes_per_element;
    all_data[i] = tflite::micro::GetTensorData<int8_t>(t);
    if(all_data[i] == nullptr)
      --num_outputs;
  }
  *num_valid_outputs = num_outputs;
}

inline void GetAllOutputDimsPointers(int32_t all_shapes[kMaxOutputNum][kMaxDims],
                                     size_t num,
                                     const WORD32* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &all_shapes[i][0];
  }
}

TfLiteStatus SplitImplHifi(TfLiteContext* context, TfLiteNode* node,
                           const TfLiteEvalTensor* input, int axis_value) {
  // Collect the shapes and data pointer of input tensors
  WORD32 outputs_shape[kMaxOutputNum][kMaxDims];
  WORD8* outputs_data[kMaxOutputNum];
  const WORD32* outputs_dims_ptr[kMaxOutputNum];
  WORD32 num_valid_outputs;
  WORD32 input_shape[kMaxDims];

  TfLiteEvalTensor* output0 =
      tflite::micro::GetEvalOutput(context, node, 0);

  int32_t bytes_per_element = 1;
  switch (input->type) {
    case kTfLiteInt8:
      bytes_per_element = 1;
      break;
    case kTfLiteInt16:
      bytes_per_element = 2;
      break;
    case kTfLiteFloat32:
    case kTfLiteInt32:
      bytes_per_element = 4;
      break;
    case kTfLiteInt64:
      bytes_per_element = 8;
      break;
    default:
      return kTfLiteError;
      break;
  }
  for(int i = 0; i < input->dims->size; i++)
    input_shape[i] = input->dims->data[i];

  input_shape[input->dims->size - 1] *= bytes_per_element;
  GetAllOutputTensorDimsDataInt8(context, node, outputs_data, outputs_shape,
                                 &num_valid_outputs, axis_value,
                                 bytes_per_element);
  GetAllOutputDimsPointers(outputs_shape, num_valid_outputs, outputs_dims_ptr);

  int32_t ret = 0;
  ret = xa_nn_split_v_8_8(outputs_data,
                          (const int32_t *const *)outputs_dims_ptr,
                          tflite::micro::GetTensorData<int8_t>(input),
                          input_shape,
                          num_valid_outputs,
                          output0->dims->size,
                          input->dims->size,
                          axis_value);
  TF_LITE_ENSURE_EQ(context, ret, 0);
  return kTfLiteOk;
}
#endif

template <typename T>
TfLiteStatus SplitImpl(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteEvalTensor* input, int axis_value) {
  const int output_count = NumOutputs(node);
  const TfLiteIntArray* input_dims = input->dims;
  const TfLiteEvalTensor* output0 =
      tflite::micro::GetEvalOutput(context, node, 0);
  const TfLiteIntArray* output_dims = output0->dims;

  const int split_dimensions = input_dims->size;
  int axis = axis_value < 0 ? axis_value + split_dimensions : axis_value;

  TFLITE_DCHECK_LT(axis, split_dimensions);
  TFLITE_DCHECK_EQ(output_dims->size, split_dimensions);

  int64_t split_size = output_dims->data[axis] * output_count;

  TFLITE_DCHECK_EQ(split_size, input_dims->data[axis]);
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_dims->data[i];
  }

  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_dims->data[i];
  }

  const T* input_ptr = tflite::micro::GetTensorData<T>(input);
  for (int k = 0; k < outer_size; ++k) {
    for (int i = 0; i < output_count; ++i) {
      TfLiteEvalTensor* t = tflite::micro::GetEvalOutput(context, node, i);
      T* output_data = tflite::micro::GetTensorData<T>(t);
      const int copy_size = output_dims->data[axis] * base_inner_size;
      T* output_ptr = output_data + k * copy_size;
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
      input_ptr += copy_size;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, axis != nullptr);

  // Dynamic output tensors are needed if axis tensor is not constant.
  // But Micro doesn't support dynamic memory allocation, so we only support
  // constant axis tensor for now.
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(axis),
                     "Non constant axis tensor not supported");

  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 1);

  int axis_value = tflite::micro::GetTensorData<int32_t>(axis)[0];
  if (axis_value < 0) {
    axis_value += input->dims->size;
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < input->dims->size);

#if defined(HIFI5) || defined(HIFI4)
  if(NumOutputs(node) <= 10) {
    switch (input->type) {
      case kTfLiteFloat32:
      case kTfLiteInt8:
      case kTfLiteInt16:
      case kTfLiteInt32: {
        return SplitImplHifi(context, node, input, axis_value);
      }
      default: {
        MicroPrintf("Type %s currently not supported.",
                    TfLiteTypeGetName(input->type));
        return kTfLiteError;
      }
    }
  }
  else {
#else
  {
#endif
    switch (input->type) {
      case kTfLiteFloat32: {
        return SplitImpl<float>(context, node, input, axis_value);
      }
      case kTfLiteInt8: {
        return SplitImpl<int8_t>(context, node, input, axis_value);
      }
      case kTfLiteInt16: {
        return SplitImpl<int16_t>(context, node, input, axis_value);
      }
      case kTfLiteInt32: {
        return SplitImpl<int32_t>(context, node, input, axis_value);
      }
      default:
        MicroPrintf("Type %s currently not supported.",
                    TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration_V1 Register_SPLIT() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
