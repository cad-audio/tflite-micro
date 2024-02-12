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
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include <string.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pad.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

TfLiteStatus PadReferenceEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataPad* data = static_cast<OpDataPad*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);
  const TfLiteEvalTensor* constant_values =
      NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, /*index=*/2)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  int16_t pad_value =
      constant_values == nullptr
          ? 0
          : *tflite::micro::GetTensorData<int16_t>(constant_values);

  reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                     tflite::micro::GetTensorData<int16_t>(input),
                     &pad_value, tflite::micro::GetTensorShape(output),
                     tflite::micro::GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

TFLMRegistration Register_PAD_INT16REF() {
  return tflite::micro::RegisterOp(XtensaPadInit, XtensaPadPrepare,
                                   PadReferenceEvalInt16);
}

}  // namespace tflite