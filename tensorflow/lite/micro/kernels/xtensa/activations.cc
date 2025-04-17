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

#include "tensorflow/lite/micro/kernels/activations.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace {

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(ReluOpData));
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const ReluOpData& data = *(static_cast<const ReluOpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if HIFI_VFPU
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<float>(input);
      out_data_ptr = tflite::micro::GetTensorData<float>(output);

      err = xa_nn_vec_relu_std_f32_f32(out_data_ptr, inp_data_ptr, flat_size);
      TF_LITE_ENSURE(context, err == 0);
#else
      ReluFloat(tflite::micro::GetTensorShape(input),
                tflite::micro::GetTensorData<float>(input),
                tflite::micro::GetTensorShape(output),
                tflite::micro::GetTensorData<float>(output));
#endif // HIFI_VFPU
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI5) || defined(HIFI4)
      int err;
      const int8_t* inp_data_ptr;
      int8_t* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
      out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

      err = xa_nn_vec_relu_asym8s_asym8s(out_data_ptr,
                                         inp_data_ptr,
                                         data.params.input_offset,
                                         data.params.output_multiplier,
                                         data.params.output_shift,
                                         data.params.output_offset,
                                         data.params.quantized_activation_min,
                                         data.params.quantized_activation_max,
                                         flat_size);

      TF_LITE_ENSURE(context, err == 0);
#else
      tflite::ReluQuantized(data, tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(input),
                            tflite::micro::GetTensorData<int8_t>(output));
#endif // defined(HIFI5) || defined(HIFI4)
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      tflite::ReluQuantized<int16_t>(
          data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

void* Relu6Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(Relu6OpData));
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const Relu6OpData& data = *(static_cast<const Relu6OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if HIFI_VFPU
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<float>(input);
      out_data_ptr = tflite::micro::GetTensorData<float>(output);

      err = xa_nn_vec_relu6_f32_f32(out_data_ptr, inp_data_ptr, flat_size);
      TF_LITE_ENSURE(context, err == 0);
#else
      Relu6Float(tflite::micro::GetTensorShape(input),
                 tflite::micro::GetTensorData<float>(input),
                 tflite::micro::GetTensorShape(output),
                 tflite::micro::GetTensorData<float>(output));
#endif // HIFI_VFPU

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI5) || defined(HIFI4)
      int err;
      const int8_t* inp_data_ptr;
      int8_t* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
      out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

      err = xa_nn_vec_activation_min_max_8_8(out_data_ptr, inp_data_ptr,
                                                     data.zero, data.six, flat_size);
      TF_LITE_ENSURE(context, err == 0);
#else
      Relu6Quantized(data.zero, data.six,
                     tflite::micro::GetTensorShape(input),
                     tflite::micro::GetTensorData<int8_t>(input),
                     tflite::micro::GetTensorShape(output),
                     tflite::micro::GetTensorData<int8_t>(output));
#endif // defined(HIFI5) || defined(HIFI4)
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      Relu6Quantized<int16_t>(data.zero, data.six,
                              tflite::micro::GetTensorShape(input),
                              tflite::micro::GetTensorData<int16_t>(input),
                              tflite::micro::GetTensorShape(output),
                              tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_RELU() {
  return tflite::micro::RegisterOp(ReluInit, ReluPrepare, ReluEval);
}

TFLMRegistration Register_RELU6() {
  return tflite::micro::RegisterOp(Relu6Init, Relu6Prepare, Relu6Eval);
}

}  // namespace tflite
