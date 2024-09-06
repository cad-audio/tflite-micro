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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
namespace tflite {
namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {

	    // setup to get filter scale
	    MicroContext* micro_context = GetMicroContext(context);
      TfLiteTensor* filter_quant =
        micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
      const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(filter_quant->quantization.params);
      const float* filter_scales = affine_quantization->scale->data;

      // quantize input to int 16
	    RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
      const int flat_size =input_shape.FlatSize();
      int16_t* quantized_input_data = new int16_t[flat_size];

      tflite::QuantizationParams op_params;
      op_params.zero_point = 0;
      const float* input_val = tflite::micro::GetTensorData<float>(input);
        float max_value = 0.0;
        for (int j = 0; j < flat_size; ++j) {
          if (input_val[j] > max_value) {
            max_value = input_val[j];
          }
        }
        // op_params.scale = static_cast<double>(256)/static_cast<double>(max_value);
	    const float input_scale = static_cast<double>(max_value*2)/static_cast<double>(65536);
	    op_params.scale = input_scale;
      tflite::reference_ops::AffineQuantize(
        op_params, input_shape, tflite::micro::GetTensorData<float>(input),
        input_shape, quantized_input_data
      );	

	  // set bias int 64
      RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);
      const int bias_flat_size =bias_shape.FlatSize();
      int64_t* new_bias = new int64_t[bias_flat_size];
      std::fill_n(new_bias, bias_flat_size, 0);

      // set output int 16
      RuntimeShape new_output_shape = tflite::micro::GetTensorShape(output);
      const int new_output_flat_size =new_output_shape.FlatSize();
      int16_t* new_output = new int16_t[new_output_flat_size];

      const int num_channels = filter->dims->data[kConvQuantizedDimension];

      // set and calculate scales and shift
      const float output_scale = (554.2135*2)/65536;

      int32_t* per_channel_output_multiplier = new int32_t[num_channels];
      std::fill_n(per_channel_output_multiplier, num_channels, 0);
	    int32_t* per_channel_output_shift = new int32_t[num_channels];
      std::fill_n(per_channel_output_shift, num_channels, 0);
	  
      for (int i = 0; i < num_channels; ++i) {
        const double effective_output_scale = static_cast<double>(input_scale) *
                                              static_cast<double>(filter_scales[i]) /
                                              static_cast<double>(output_scale);
        int32_t significand;
        int channel_shift;
        tflite::QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
        per_channel_output_multiplier[i] = significand;
        per_channel_output_shift[i] = channel_shift;
      }

      micro_context->DeallocateTempTfLiteTensor(filter_quant);
    
      reference_integer_ops::ConvPerChannel(
        ConvParamsQuantized(params, data),
        per_channel_output_multiplier, per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        quantized_input_data,
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        new_bias,
        tflite::micro::GetTensorShape(output),
        new_output);

      RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
      tflite::DequantizationParams dequantization_params;
      dequantization_params.scale = output_scale;
      dequantization_params.zero_point = 0;

      tflite::reference_ops::Dequantize(dequantization_params,
                                output_shape,
                                new_output,
                                output_shape,
                                tflite::micro::GetTensorData<float>(output));

    //   tflite::reference_ops::Conv(
    //       ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
    //       tflite::micro::GetTensorData<float>(input),
    //       tflite::micro::GetTensorShape(filter),
    //       tflite::micro::GetTensorData<float>(filter),
    //       tflite::micro::GetTensorShape(bias),
    //       tflite::micro::GetOptionalTensorData<float>(bias),
    //       tflite::micro::GetTensorShape(output),
    //       tflite::micro::GetTensorData<float>(output),
    //       tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else if (bias->type == kTfLiteInt64) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare, Eval);
}

}  // namespace tflite
