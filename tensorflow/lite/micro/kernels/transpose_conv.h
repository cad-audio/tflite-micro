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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

TFLMRegistration Register_TRANSPOSE_CONV();
TFLMRegistration Register_TRANSPOSE_CONV_INT8REF();
TFLMRegistration Register_TRANSPOSE_CONV_INT16REF();
TFLMRegistration Register_TRANSPOSE_CONV_FLOAT32REF();

#if defined(XTENSA)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_TRANSPOSE_CONV_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_TRANSPOSE_CONV_INT16();

// Returns a TFLMRegistration struct for kernel variant that only supports
// float32 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_TRANSPOSE_CONV_FLOAT32();

#else
inline TFLMRegistration Register_TRANSPOSE_CONV_INT8() { return Register_TRANSPOSE_CONV(); }

inline TFLMRegistration Register_TRANSPOSE_CONV_INT16() { return Register_TRANSPOSE_CONV(); }

inline TFLMRegistration Register_TRANSPOSE_CONV_FLOAT32() { return Register_TRANSPOSE_CONV(); }
#endif  // defined(XTENSA)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
