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
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_sub.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

void* SubInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSub));
}

TfLiteStatus EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpDataSub* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
  int err;
  const RuntimeShape& input1_shape = tflite::micro::GetTensorShape(input1);
  const RuntimeShape& input2_shape = tflite::micro::GetTensorShape(input2);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);  
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, input1_shape);
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, input2_shape);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  const int* input1_dims = extended_input1_shape.DimsData();
  const int* input2_dims = extended_input2_shape.DimsData();
  const int* output_dims = extended_output_shape.DimsData();

  err = xa_nn_elm_sub_broadcast_4D_f32xf32_f32(
      tflite::micro::GetTensorData<float>(output), output_dims, 
      tflite::micro::GetTensorData<float>(input1), input1_dims,
      tflite::micro::GetTensorData<float>(input2), input2_dims);
  TF_LITE_ENSURE(context, err == 0);

  const int flat_size = output_shape.FlatSize();
  err = xa_nn_vec_activation_min_max_f32_f32(
      tflite::micro::GetTensorData<float>(output), tflite::micro::GetTensorData<float>(output),
      output_activation_min, output_activation_max, flat_size);  
#else
  EvalSubFloat32Reference(context, node);
#endif
  return kTfLiteOk;
}

TfLiteStatus EvalSubQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
  switch (output->type) {
    case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      EvalSubQuantizedInt8(context, node, params, data,  input1, input2, output);
#else // #if defined(HIFI4) || defined(HIFI5)
      EvalSubInt8Reference(context, node);
#endif // #if defined(HIFI4) || defined(HIFI5)
      break;
    }
    case kTfLiteInt16: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      EvalSubQuantizedInt16(context, node, params, data,  input1, input2, output);
#else // #if defined(HIFI4) || defined(HIFI5)
      EvalSubInt16Reference(context, node);
#endif // #if defined(HIFI4) || defined(HIFI5)
      break;
    }
    default:
      MicroPrintf("Quantized type %s not currently supported.",
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SubEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kSubInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kSubInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSubOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSub& data = *(static_cast<const OpDataSub*>(node->user_data));

  if (output->type == kTfLiteFloat32) {
    EvalSub(context, node, params, &data, input1, input2, output);
  } else if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(context, EvalSubQuantized(context, node, params, &data,
                                                input1, input2, output));
  } else {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(output->type),
                output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TFLMRegistration Register_SUB() {
  return tflite::micro::RegisterOp(SubInit, SubPrepare, SubEval);
}

}  // namespace tflite
