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

namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
#if defined(VISION_P6)
  XtensaPadData* op_data_xtensa = static_cast<XtensaPadData*>(node->user_data);
  OpDataPad* data = &op_data_xtensa->reference_op_data;
#endif

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);

  switch (input->type) {
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
        PadEvalHifiFloat32(context, node);
#else      
        PadReferenceEvalInt32(context, node);
#endif           
      } break;
    case kTfLiteInt8: {
#if defined(VISION_P6)
      PadEvalVision(*op_data_xtensa, input, output);
#elif defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      PadEvalHifiInt8(context, node);
#else
      PadReferenceEvalInt8(context, node);
#endif // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    } break;
    case kTfLiteInt16: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
        PadEvalHifiInt16(context, node);
#else  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
        PadReferenceEvalInt16(context, node);
#endif
    } break;
    case kTfLiteInt32: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
        PadEvalHifiInt32(context, node);
#else
        PadReferenceEvalInt32(context, node);
#endif   
    } 
    break;
    default:

      MicroPrintf("Type %s not currently supported by Pad.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_PAD() {
  return tflite::micro::RegisterOp(XtensaPadInit, XtensaPadPrepare, Eval);
}

// Also register Pad as PadV2.
TFLMRegistration Register_PADV2() {
  return tflite::micro::RegisterOp(XtensaPadInit, XtensaPadPrepare, Eval);
}

}  // namespace tflite