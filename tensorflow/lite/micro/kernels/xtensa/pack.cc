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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"


namespace tflite {

namespace {

constexpr int kOutputTensor = 0;

#if defined(HIFI5) || defined(HIFI4)
constexpr int kMaxInputNum = 10;  // Maximum number of input tensors
// Equal to kMaxSmallSize from tensorflow/lite/kernels/internal/runtime_shape.h
constexpr int kMaxDims = 6;  // Maximum number of dimensions

// Gets Dims from a list of tensors, equivalent to Int8, for higher
// datatypes multiply last dimension by bytes_per_element
// Also gets all input tensor data pointers
inline void GetAllInputTensorDimsDataInt8(const TfLiteContext* context,
                                          const TfLiteNode* node,
                                          const int8_t* all_data[kMaxInputNum],
                                          int32_t all_shapes[kMaxInputNum][kMaxDims],
                                          int32_t axis,
                                          int32_t bytes_per_element) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    int num_dims = t->dims->size;
    int32_t axis_dim_set = 0;
    // Output has one more dimension than input and that is axis, since we are
    // using concat kernel from NNLib which needs inputs/output to have same
    // number of dimensions inserting axis dimension in input shapes
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

inline void GetAllInputDimsPointers(int32_t all_shapes[kMaxInputNum][kMaxDims],
                                    size_t num,
                                    const WORD32* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &all_shapes[i][0];
  }
}

TfLiteStatus PackImplHifi(TfLiteContext* context, TfLiteNode* node,
                          TfLiteEvalTensor* output, int values_count, int axis) {
  // Collect the shapes and data pointer of input tensors
  WORD32 inputs_shape[kMaxInputNum][kMaxDims];
  const WORD8* inputs_data[kMaxInputNum];
  const WORD32* inputs_dims_ptr[kMaxInputNum];
  WORD32 output_shape[kMaxDims];

  const int dimensions = output->dims->size;

  if (axis < 0) {
    axis += dimensions;
  }

  int32_t bytes_per_element = 1;
  switch (output->type) {
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
  for(int i = 0; i < output->dims->size; i++)
    output_shape[i] = output->dims->data[i];

  output_shape[output->dims->size - 1] *= bytes_per_element;
  GetAllInputTensorDimsDataInt8(context, node, inputs_data, inputs_shape,
                                axis, bytes_per_element);
  GetAllInputDimsPointers(inputs_shape, node->inputs->size, inputs_dims_ptr);

  int32_t ret = 0;
  ret = xa_nn_concat_8_8(tflite::micro::GetTensorData<int8_t>(output),
                         output_shape,
                         inputs_data,
                         (const int32_t *const *)inputs_dims_ptr,
                         output->dims->size,
                         node->inputs->size,
                         output->dims->size,
                         axis);
  TF_LITE_ENSURE_EQ(context, ret, 0);
  return kTfLiteOk;
}
#endif

template <typename T>
TfLiteStatus PackImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteEvalTensor* output, int values_count, int axis) {
  const TfLiteEvalTensor* input0 =
      tflite::micro::GetEvalInput(context, node, 0);

  const int dimensions = output->dims->size;
  const TfLiteIntArray* input_dims = input0->dims;
  const TfLiteIntArray* output_dims = output->dims;

  if (axis < 0) {
    axis += dimensions;
  }

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_dims->data[i];
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= output_dims->data[i];
  }
  int input_size = 1;
  for (int i = 0; i < input_dims->size; ++i) {
    input_size *= input_dims->data[i];
  }
  TFLITE_DCHECK_EQ(input_size, copy_size * outer_size);

  T* output_data = tflite::micro::GetTensorData<T>(output);

  for (int i = 0; i < values_count; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    const T* input_data = tflite::micro::GetTensorData<T>(t);
    for (int k = 0; k < outer_size; ++k) {
      const T* input_ptr = input_data + copy_size * k;
      int loc = k * values_count * copy_size + i * copy_size;
      T* output_ptr = output_data + loc;
#if defined(HIFI5) || defined(HIFI4)
      memcpy(output_ptr, input_ptr, copy_size*sizeof(T));
#else
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
#endif // defined(HIFI5) || defined(HIFI4)
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

#if defined(HIFI5) || defined(HIFI4)
  if(data->values_count <= 10) {
    switch (output->type) {
      case kTfLiteFloat32:
      case kTfLiteInt8:
      case kTfLiteInt16:
      case kTfLiteInt32:
      case kTfLiteInt64: {
        return PackImplHifi(context, node, output, data->values_count,
                            data->axis);
      }
      default: {
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by pack.",
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
      }
    }
  }
  else {
#else
  {
#endif
    switch (output->type) {
      case kTfLiteFloat32: {
        return PackImpl<float>(context, node, output, data->values_count,
                               data->axis);
      }
      case kTfLiteInt8: {
        return PackImpl<int8_t>(context, node, output, data->values_count,
                                data->axis);
      }
      case kTfLiteInt16: {
        return PackImpl<int16_t>(context, node, output, data->values_count,
                                 data->axis);
      }
      case kTfLiteInt32: {
        return PackImpl<int32_t>(context, node, output, data->values_count,
                                 data->axis);
      }
      case kTfLiteInt64: {
        return PackImpl<int64_t>(context, node, output, data->values_count,
                                 data->axis);
      }
      default: {
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by pack.",
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration_V1 Register_PACK() {
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval);
}

}  // namespace tflite
