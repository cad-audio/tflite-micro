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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
TfLiteStatus ValidateGoldens(TfLiteTensor* tensors, const int tensors_size,
                             const T* expected_output_data, T* output_data,
                             int output_length,
                             TfLiteGeluParams* params,
                             float tolerance = 1e-5) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_GELU();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, params);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }

  return kTfLiteOk;
}

void TestGelu(int* input_dims_data, const float* input_data,
                        int* output_dims_data, const float* expected_output_data, 
                        float* output_data, TfLiteGeluParams* params,
                        float tolerance = 1e-5) {

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size]{
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateGoldens(tensors, tensors_size, expected_output_data, output_data,
                      output_dims_count, params, tolerance));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatExact) {

  int input_dims[] = {2, 2, 3}; // Changed to reflect 2 rows, 3 columns
  const float input_data[] = {
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  };

  const float expected_output_data[] = {
      0.0f, 0.841345f, 2.99595f,           // Row 1
      0.841345f, -0.158655f, -0.0455003f,  // Row 2
  };
  int output_dims[] = {2, 2, 3}; // Changed to reflect 2 rows, 3 columns
  constexpr int OutputCount = std::extent<decltype(expected_output_data)>::value;
  float output_data[OutputCount];

  TfLiteGeluParams params = {
      false, /*approximate*/
  };

  tflite::testing::TestGelu(input_dims, input_data,
                            output_dims, expected_output_data,
                            output_data, &params);
}

TF_LITE_MICRO_TEST(FloatApproximate) {
  int input_dims[] = {2, 2, 3}; // Changed to reflect 2 rows, 3 columns
  const float input_data[] = {
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  };

  const float expected_output_data[] = {
       0.0f, 0.841192f, 2.99636f,           // Row 1
       0.841192f, -0.158808f, -0.0454023f,  // Row 2
  };
  int output_dims[] = {2, 2, 3}; // Changed to reflect 2 rows, 3 columns
  constexpr int OutputCount = std::extent<decltype(expected_output_data)>::value;
  float output_data[OutputCount];

  TfLiteGeluParams params = {
      true, /*approximate*/
  };

  tflite::testing::TestGelu(input_dims, input_data,
                            output_dims, expected_output_data,
                            output_data, &params, /*tolerance=*/1e-3);
}


TF_LITE_MICRO_TESTS_END