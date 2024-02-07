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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_STRIDED_SLICE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_STRIDED_SLICE_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
namespace tflite {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)


TfLiteStatus StridedSlice_int16_hifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus StridedSlice_int32_hifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus StridedSlice_int8_hifi(TfLiteContext* context, TfLiteNode* node);

#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

TfLiteStatus StridedSliceInt8Ref(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus StridedSliceInt16Ref(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus StridedSliceInt32Ref(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus StridedSliceFloat32Ref(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_STRIDED_SLICE_H_
