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

#include <type_traits>

#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteSelectParams;

template <typename T>
void TestReverse(int* input1_dims_data, const T* input1_data,
                int* axis_dims_data, const int* axis_data,
                int* output_dims_data, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(axis_data, axis_dims),
                                        CreateTensor(output_data, output_dims)};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteSelectParams builtin_data;
  const TFLMRegistration registration = tflite::Register_REVERSE_V2();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void ExpectEqual(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

template <typename T>
void ExpectNear(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], 1e-5f);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ReverseInt8) {
  int input_shape[] = {1, 4};
  int axis_shape[] = {1, 1};
  int output_shape[] = {1, 4};
  const int8_t input[] = {1, 2, 3, 4};
  const int axis[] = {0};
  int8_t expected_output[] = {4, 3, 2, 1};

  int8_t output_data[4];

  tflite::testing::TestReverse(input_shape, input, axis_shape, axis, output_shape, output_data);
  tflite::testing::ExpectEqual(input_shape, expected_output, output_data);
}

TF_LITE_MICRO_TESTS_END
