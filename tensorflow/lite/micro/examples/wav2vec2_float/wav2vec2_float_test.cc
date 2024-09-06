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
#include "tensorflow/lite/micro/examples/wav2vec2_float/wav2vec2_dynamic.cc"

#include "tensorflow/lite/micro/examples/wav2vec2_float/model_settings.h"
#include "tensorflow/lite/micro/examples/wav2vec2_float/tensor_input.h"
// #include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern unsigned char wav2vec2_dynamic_tflite[];
extern unsigned int wav2vec2_dynamic_tflite_len;
// Create an area of memory to use for input, output, and intermediate arrays.
#if defined(XTENSA) && defined(VISION_P6)
constexpr int tensor_arena_size =4* 352000*1024;
#else
// constexpr int tensor_arena_size =4*35200*400;
// constexpr long long tensor_arena_size = 1195853630LL;
constexpr int tensor_arena_size = 150000000;

#endif  // defined(XTENSA) && defined(VISION_P6)
uint8_t tensor_arena[tensor_arena_size];

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(wav2vec2_dynamic_tflite); 
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
	tflite::MicroMutableOpResolver<19> micro_op_resolver;
	micro_op_resolver.AddAdd();
	micro_op_resolver.AddAveragePool2D();
	micro_op_resolver.AddBatchMatMul();
	micro_op_resolver.AddConv2D();
	micro_op_resolver.AddDepthwiseConv2D();
	micro_op_resolver.AddDequantize();
	micro_op_resolver.AddFullyConnected();
	micro_op_resolver.AddGelu();
	micro_op_resolver.AddMean();
	micro_op_resolver.AddMul();
	micro_op_resolver.AddPad();
	micro_op_resolver.AddQuantize();
	micro_op_resolver.AddReshape();
	micro_op_resolver.AddRsqrt();
	micro_op_resolver.AddSoftmax();
	micro_op_resolver.AddSlice();
	micro_op_resolver.AddSquaredDifference();
	micro_op_resolver.AddSub();
	micro_op_resolver.AddTranspose();

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(26239, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  // Copy an image with a person into the memory area used for the input.
  TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(26239*sizeof(float)));
  memcpy(input->data.f, tensor_input_bin, input->bytes);

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(3, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(81, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // print out the raw output, no processing for now
  float* output_data = output->data.f;
  int argmax[81]; 

  for (int i = 0; i < 81; ++i) {
    float max_value = output_data[i*32];  // Initialize max to first element of the row
    int max_index = 0;
    
    for (int j = 0; j < 32; ++j) {
        float current_value = output_data[i*32 + j];   
        if (current_value > max_value) {
            max_value = current_value;
            max_index = j;
        }
    }
    argmax[i] = max_index;
}

  MicroPrintf("Argmax values:\n");
  for (int i = 0; i < 81; ++i) {
    MicroPrintf("%d ", argmax[i]);
  }
  MicroPrintf("\n");
  MicroPrintf("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END