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

TfLiteStatus AverageEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteInt16: {
#if defined(HIFI5) || defined(HIFI4)
      auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
      AverageEvalQuantizedInt16Hifi(context, node, params, op_data, input, output);
#else
      const OpDataPooling* reference_op_data =
          static_cast<const OpDataPooling*>(node->user_data);
      AveragePoolingEvalQuantized<int16_t>(context, node, params,
                                          reference_op_data, input, output);
#endif
      break;
    }
    default: {
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteInt16: {
#if defined(HIFI5) || defined(HIFI4)
      auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
      MaxEvalQuantizedInt16Hifi(context, node, params, op_data, input, output);
#else
      const OpDataPooling* reference_op_data =
          static_cast<const OpDataPooling*>(node->user_data);
      MaxPoolingEvalQuantized<int16_t>(context, node, params, reference_op_data,
                                      input, output);
#endif
      break;
    }
    default: {
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus AverageEvalQuantizedInt16Hifi(TfLiteContext* context,
                                      const TfLiteNode* node,
                                      const TfLitePoolParams* params,
                                      const XtensaOpDataPooling* data,
                                      const TfLiteEvalTensor* input,
                                      TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt16);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  const int16_t* inp_data_ptr = tflite::micro::GetTensorData<int16_t>(input);
  int16_t* out_data_ptr = tflite::micro::GetTensorData<int16_t>(output);

  for (int batch = 0; batch < batches; ++batch) {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_avgpool_16(
            &out_data_ptr[output_height * output_width * depth * batch],
            const_cast<int16_t*>(
                &inp_data_ptr[output_height * output_width * depth * batch]),
            input_height, input_width, depth, params->filter_height,
            params->filter_width, params->stride_width, params->stride_height,
            data->reference_op_data.padding.width,
            data->reference_op_data.padding.height, output_height, output_width,
            0, 0, p_scratch),
        0);
  }

  const int out_length = batches * output_height * output_width * depth;
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_16_16(
          out_data_ptr, out_data_ptr, data->reference_op_data.activation_min,
          data->reference_op_data.activation_max, out_length),
      0);

  return kTfLiteOk;
}

TfLiteStatus MaxEvalQuantizedInt16Hifi(TfLiteContext* context,
                                      const TfLiteNode* node,
                                      const TfLitePoolParams* params,
                                      const XtensaOpDataPooling* data,
                                      const TfLiteEvalTensor* input,
                                      TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt16);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  const int16_t* inp_data_ptr = tflite::micro::GetTensorData<int16_t>(input);
  int16_t* out_data_ptr = tflite::micro::GetTensorData<int16_t>(output);

  for (int batch = 0; batch < batches; ++batch) {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_maxpool_16(
            &out_data_ptr[output_height * output_width * depth * batch],
            const_cast<int16_t*>(
                &inp_data_ptr[output_height * output_width * depth * batch]),
            input_height, input_width, depth, params->filter_height,
            params->filter_width, params->stride_width, params->stride_height,
            data->reference_op_data.padding.width,
            data->reference_op_data.padding.height, output_height, output_width,
            0, 0, p_scratch),
        0);
  }

  const int out_length = batches * output_height * output_width * depth;
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_16_16(
          out_data_ptr, out_data_ptr, data->reference_op_data.activation_min,
          data->reference_op_data.activation_max, out_length),
      0);

  return kTfLiteOk;
}

TFLMRegistration Register_AVERAGE_POOL_2D_INT16() {
#if defined(HIFI5) || defined(HIFI4)
  return tflite::micro::RegisterOp(XtensaPoolingInit, AveragePrepareHifi,
                                   AverageEvalInt16);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   AverageEvalInt16);
#endif
}

TFLMRegistration Register_MAX_POOL_2D_INT16() {
#if defined(HIFI5) || defined(HIFI4)
  return tflite::micro::RegisterOp(XtensaPoolingInit, MaxPrepareHifi,
                                   MaxEvalInt16);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   MaxEvalInt16);
#endif
}

}  // namespace tflite
