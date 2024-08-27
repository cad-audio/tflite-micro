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

#include "tensorflow/lite/kernels/internal/reference/gelu.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/gelu.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {


void* GeluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(GeluParams));
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  GeluParams op_data = *static_cast<GeluParams*>(node->user_data);


  switch (input->type) {
    case kTfLiteFloat32: {
	   tflite::reference_ops::Gelu(
		tflite::micro::GetTensorShape(input),
		tflite::micro::GetTensorData<float>(input),
		op_data.approximate,
		tflite::micro::GetTensorShape(output),
		tflite::micro::GetTensorData<float>(output));
    //   GeluParams op_params = {};
    //   const auto* params =
    //       static_cast<TfLiteGeluParams*>(node->builtin_data);

    //   op_params.alpha = params->alpha;
    //   reference_ops::Gelu(op_params, tflite::micro::GetTensorShape(input),
    //                            tflite::micro::GetTensorData<float>(input),
    //                            tflite::micro::GetTensorShape(output),
    //                            tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    // case kTfLiteInt8: {
    //   QuantizeGelu<int8_t>(data, input, output);
    //   return kTfLiteOk;
    // } break;
    // case kTfLiteInt16: {
    //   QuantizeGelu<int16_t>(data, input, output);
    //   return kTfLiteOk;
    // } break;
    default:
      MicroPrintf("Only float32 is supported by GELU, got %s.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteError;
}

TFLMRegistration Register_GELU() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare,
                                   GeluEval);
}

}  // namespace tflite