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

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace ops {
namespace micro {
namespace elementwise {
namespace hifi {

#if defined(HIFI5)
inline TfLiteStatus EvalLogicalNot(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    const size_t num_elements = ElementCount(*input->dims);

    int err;
    const int8_t *input_data_ptr;
    int8_t *output_data_ptr;

    input_data_ptr  = tflite::micro::GetTensorData<int8_t>(input);
    output_data_ptr  = tflite::micro::GetTensorData<int8_t>(output);

    err = xa_nn_elm_logicalnot_bool_bool(output_data_ptr,
        input_data_ptr,
        num_elements);
    TF_LITE_ENSURE(context, err==0);
    return kTfLiteOk;
  }
#endif // defined(HIFI5)

} //namespae hifi

namespace {

bool IsNumericSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
  return type == kTfLiteBool;
}

typedef bool (*IsSupportedType)(TfLiteType);
template <IsSupportedType>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!IsSupportedType(input->type)) {
    TF_LITE_KERNEL_LOG(context, "Input data type %s (%d) is not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             T func(T), TfLiteType expected_type) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const size_t num_elements = ElementCount(*input->dims);
  const T* in_data = tflite::micro::GetTensorData<T>(input);
  T* out_data = tflite::micro::GetTensorData<T>(output);
  for (size_t i = 0; i < num_elements; ++i) {
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

#if !defined(HIFI5)
inline TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}
#endif // !defined(HIFI5)

#if !defined(HIFI5)
inline TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}
#endif // !defined(HIFI5)

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_abs_f32_f32(out_data,
                              in_data,
                              num_elements
                              );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, std::abs);
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_sine_f32_f32(out_data,
                               in_data,
                               num_elements
                              );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, std::sin);
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_cosine_f32_f32(out_data,
                                 in_data,
                                 num_elements
                                );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, std::cos);
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_logn_f32_f32(out_data,
                               in_data,
                               num_elements
                              );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, std::log);
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_sqrt_f32_f32(out_data,
                               in_data,
                               num_elements
                              );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, std::sqrt);
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_rsqrt_f32_f32(out_data,
                                in_data,
                                num_elements
                               );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, [](float f) { return 1.f / std::sqrt(f); });
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
#if HIFI_VFPU && defined(HIFI5)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err;
  err = xa_nn_elm_square_f32_f32(out_data,
                                 in_data,
                                 num_elements
                                );
  TF_LITE_ENSURE(context, (err==0) );
  return kTfLiteOk;
#else
  return EvalNumeric(context, node, [](float f) { return f * f; });
#endif // HIFI_VFPU && defined(HIFI5)
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI5)
  return hifi::EvalLogicalNot(context, node);
#else
  return EvalLogical(context, node, [](bool v) { return !v; });
#endif // defined(HIFI5)
}

}  // namespace
}  // namespace elementwise

TfLiteRegistration Register_ABS() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::AbsEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SIN() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::SinEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_COS() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::CosEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOG() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::LogEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SQRT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::SqrtEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RSQRT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::RsqrtEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SQUARE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsNumericSupportedType>,
          /*invoke=*/elementwise::SquareEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOGICAL_NOT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          elementwise::GenericPrepare<elementwise::IsLogicalSupportedType>,
          /*invoke=*/elementwise::LogicalNotEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
