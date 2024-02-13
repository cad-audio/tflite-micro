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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_MUL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_MUL_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/mul.h"
namespace tflite {


#if defined(XTENSA)
TfLiteStatus EvalMulFloatHiFi(TfLiteContext* context, TfLiteNode* node,
    TfLiteMulParams* params, const OpDataMul* data,
    const TfLiteEvalTensor* input1, const TfLiteEvalTensor* input2,
    TfLiteEvalTensor* output);
TfLiteStatus EvalMulQuantizedHiFiInt8(TfLiteContext* context,
                   TfLiteNode* node, const OpDataMul* data,
                   const TfLiteEvalTensor* input1,
                   const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);
TfLiteStatus EvalMulQuantizedHiFiInt16(TfLiteContext* context,
                   TfLiteNode* node, const OpDataMul* data,
                   const TfLiteEvalTensor* input1,
                   const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);                                                        
#endif

TfLiteStatus MulEvalQuantizedReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus MulEvalFloatReference(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_MUL_H_
