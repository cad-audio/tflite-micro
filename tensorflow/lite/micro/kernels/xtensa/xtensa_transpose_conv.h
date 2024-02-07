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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"

namespace tflite {
struct OpData {
  ConvParams params;

  // A scratch buffer is required for quantized implementations.
  int scratch_buffer_index;

  // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
  // 64-bit biases.
  int bias_converted_buffer_index;

  // Multiplier and shift arrays are required for the int8 implementation.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
};

constexpr int kFilterTensor = 1;
constexpr int kInputTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kOutputTensor = 0;
constexpr int kConvQuantizedDimension = 0;

#if defined(HIFI4) || defined(HIFI5)
TfLiteStatus TransposeConvPrepareHifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus TransposeConvEvalHifiInt8(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteTransposeConvParams& params,
                              const OpData& data,
                              const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* filter,
                              const TfLiteEvalTensor* bias,
                              TfLiteEvalTensor* output);

TfLiteStatus TransposeConvEvalHifiInt16(TfLiteContext* context, TfLiteNode* node,
                               const TfLiteTransposeConvParams& params,
                               const OpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output);

#if defined(INCLUDE_FLOAT_OPT) 
TfLiteStatus TransposeConvEvalHifiFloat32(TfLiteContext* context, TfLiteNode* node,
                               const TfLiteTransposeConvParams& params,
                               const OpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output);
#endif

#endif  // defined(HIFI4) || defined(HIFI5)

TfLiteStatus TransposeConvReferenceEvalInt8(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus TransposeConvReferenceEvalInt16(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus TransposeConvReferenceEvalFloat32(TfLiteContext* context, TfLiteNode* node);

void* TransposeConvInit(TfLiteContext* context, const char* buffer, size_t length);
TfLiteStatus TransposeConvPrepare(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus TransposeConvPrepareXtensa(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTransposeConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data);
PaddingType RuntimePaddingType(TfLitePadding padding);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_
