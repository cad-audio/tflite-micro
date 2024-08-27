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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_

#include <cmath>
#include <functional>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"


namespace tflite {
namespace reference_ops {

namespace gelu_internal {

constexpr float kSqrt2dPi = (1.12837916709551257390 /* 2/sqrt(pi) */) * (0.70710678118654752440 /* 1/sqrt(2) */);  // sqrt( 2 / pi )
}  // namespace gelu_internal

// Plain implementations for GELU. Used for populating lookup table.
inline float GeluTransform(float in) {
  // Note: 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) ) is commonly used, but cause
  // catastropic cancellation for large negative inputs. Rewriting the
  // expression via erfc avoids the numerical stability issues.
  return 0.5f * in * std::erfc(in * static_cast<float>(-(0.70710678118654752440 /* 1/sqrt(2) */)));
}

inline float GeluTransformApproximate(float in) {
  // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
  return 0.5f * in *
         (1.f + std::tanh(gelu_internal::kSqrt2dPi *
                          // Note: Avoid std::pow for integer exponents
                          // as it leads to much slower performance.
                          (in + 0.044715f * in * in * in)));
}

template <typename T>
inline void Gelu(const RuntimeShape& input_shape, const T* input_data,
                 bool approximate, const RuntimeShape& output_shape,
                 T* output_data) {

	if (approximate) {
		for (int i = 0; i < input_shape.FlatSize(); ++i) {
		float x = input_data[i];
		output_data[i] = 0.5f * x * (1.f + std::tanh(gelu_internal::kSqrt2dPi *
							(x + 0.044715f * x * x * x)));
		}
	} else {
		for (int i = 0; i < input_shape.FlatSize(); ++i) {
		float x = input_data[i];
		output_data[i] = 0.5f * x * std::erfc(x * static_cast<float>(-(0.70710678118654752440 /* 1/sqrt(2) */)));
		}
	}
}



}  // namespace reference_ops
}  // namespace tflite


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_