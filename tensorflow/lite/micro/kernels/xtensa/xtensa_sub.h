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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SUB_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SUB_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/sub.h"
namespace tflite {

void* SubInit(TfLiteContext* context, const char* buffer, size_t length);
TfLiteStatus EvalSubInt8Reference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus EvalSubInt16Reference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus EvalSubFloat32Reference(TfLiteContext* context, TfLiteNode* node);
#if defined(XTENSA)
TfLiteStatus EvalSubQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output);   

TfLiteStatus EvalSubQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output);                               
TfLiteStatus EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpDataSub* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);

TfLiteStatus EvalSubFloat32(TfLiteContext* context, TfLiteNode* node);
#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SUB_H_
