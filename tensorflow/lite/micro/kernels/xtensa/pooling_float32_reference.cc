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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

TfLiteStatus AverageReferenceEvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  const OpDataPooling* reference_op_data =
      static_cast<const OpDataPooling*>(node->user_data);
  AveragePoolingEvalFloat(context, node, params,
                          reference_op_data, input, output);
  return kTfLiteOk;
}

TfLiteStatus MaxReferenceEvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
  micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  const OpDataPooling* reference_op_data =
      static_cast<const OpDataPooling*>(node->user_data);
  MaxPoolingEvalFloat(context, node, params, reference_op_data,
                                  input, output);

  return kTfLiteOk;
}

TFLMRegistration Register_AVERAGE_POOL_2D_FLOAT32REF() {
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   AverageReferenceEvalFloat32);
}

TFLMRegistration Register_MAX_POOL_2D_FLOAT32REF() {
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   MaxReferenceEvalFloat32);
}

}  // namespace tflite
