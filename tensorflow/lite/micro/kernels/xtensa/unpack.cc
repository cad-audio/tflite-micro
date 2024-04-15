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

constexpr int kInputTensor = 0;

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
                                           int32_t axis,
                                           int32_t bytes_per_element) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteEvalTensor* t = tflite::micro::GetEvalOutput(context, node, i);
    int num_dims = t->dims->size;
    int32_t axis_dim_set = 0;
    // Input has one more dimension than output and that is axis, since we are
    // using split_v kernel from NNLib which needs inputs/output to have same
    // number of dimensions inserting axis dimension in output shapes
    for(int j = 0; j < num_dims + 1; j++) {
      if(j == axis) {
        all_shapes[i][j] = 1;
        axis_dim_set = 1;
      }
      else {
        all_shapes[i][j] = t->dims->data[j - axis_dim_set];
      }
    }
    all_shapes[i][num_dims] = all_shapes[i][num_dims] * bytes_per_element;
    all_data[i] = tflite::micro::GetTensorData<int8_t>(t);
  }
}

inline void GetAllOutputDimsPointers(int32_t all_shapes[kMaxOutputNum][kMaxDims],
                                     size_t num,
                                     const WORD32* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &all_shapes[i][0];
  }
}

TfLiteStatus UnpackImplHifi(TfLiteContext* context, TfLiteNode* node,
                            const TfLiteEvalTensor* input, int output_count,
                            int axis) {
  // Collect the shapes and data pointer of input tensors
  WORD32 outputs_shape[kMaxOutputNum][kMaxDims];
  WORD8* outputs_data[kMaxOutputNum];
  const WORD32* outputs_dims_ptr[kMaxOutputNum];
  WORD32 input_shape[kMaxDims];

  const int dimensions = input->dims->size;

  if (axis < 0) {
    axis += dimensions;
  }

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
                                 axis, bytes_per_element);
  GetAllOutputDimsPointers(outputs_shape, output_count, outputs_dims_ptr);

  int32_t ret = 0;
  ret = xa_nn_split_v_8_8(outputs_data,
                          (const int32_t *const *)outputs_dims_ptr,
                          tflite::micro::GetTensorData<int8_t>(input),
                          input_shape,
                          output_count,
                          dimensions,
                          dimensions,
                          axis);
  TF_LITE_ENSURE_EQ(context, ret, 0);
  return kTfLiteOk;
}
#endif

template <typename T>
TfLiteStatus UnpackImpl(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteEvalTensor* input, int output_count,
                        int axis) {
  const TfLiteEvalTensor* output0 =
      tflite::micro::GetEvalOutput(context, node, 0);
  const TfLiteIntArray* input_dims = input->dims;
  const TfLiteIntArray* output_dims = output0->dims;
  const int dimensions = input_dims->size;

  if (axis < 0) {
    axis += input->dims->size;
  }

  TFLITE_DCHECK_LT(axis, dimensions);

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_dims->data[i];
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= input_dims->data[i];
  }
  int output_size = 1;
  for (int i = 0; i < output_dims->size; ++i) {
    output_size *= output_dims->data[i];
  }
  TFLITE_DCHECK_EQ(output_size, copy_size * outer_size);

  const T* input_data = tflite::micro::GetTensorData<T>(input);

  for (int i = 0; i < output_count; ++i) {
    TfLiteEvalTensor* t = tflite::micro::GetEvalOutput(context, node, i);
    T* output_data = tflite::micro::GetTensorData<T>(t);
    for (int k = 0; k < outer_size; ++k) {
      T* output_ptr = output_data + copy_size * k;
      int loc = k * output_count * copy_size + i * copy_size;
      const T* input_ptr = input_data + loc;
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteUnpackParams* data =
      reinterpret_cast<TfLiteUnpackParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);

#if defined(HIFI5) || defined(HIFI4)
  if(NumOutputs(node) <= 10) {
    switch (input->type) {
      case kTfLiteFloat32:
      case kTfLiteInt32:
      case kTfLiteInt16:
      case kTfLiteInt8: {
        return UnpackImplHifi(context, node, input, data->num, data->axis);
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
        return UnpackImpl<float>(context, node, input, data->num, data->axis);
      }
      case kTfLiteInt32: {
        return UnpackImpl<int32_t>(context, node, input, data->num, data->axis);
      }
      case kTfLiteInt16: {
        return UnpackImpl<int16_t>(context, node, input, data->num, data->axis);
      }
      case kTfLiteInt8: {
        return UnpackImpl<int8_t>(context, node, input, data->num, data->axis);
      }
      default: {
        MicroPrintf("Type '%s' is not supported by unpack.",
                    TfLiteTypeGetName(input->type));
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration_V1 Register_UNPACK() {
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval);
}

}  // namespace tflite
