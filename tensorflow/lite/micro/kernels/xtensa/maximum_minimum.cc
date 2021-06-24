/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace ops {
namespace micro {
namespace maximum_minimum {
namespace {

// This file has the HiFi implementation of TFMaximum/TFMinimum.
enum KernelType {
  kHiFi,
  kReference,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input1 = tflite::micro::GetEvalInput(context, node, kInputTensor1);
    input2 = tflite::micro::GetEvalInput(context, node, kInputTensor2);
    output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  }
  const TfLiteEvalTensor* input1;
  const TfLiteEvalTensor* input2;
  TfLiteEvalTensor* output;
};

struct MaximumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 > el2 ? el1 : el2;
  }
};

struct MinimumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 < el2 ? el1 : el2;
  }
};

}  // namespace maximum_minimum

namespace hifi {

enum class basic_op {
  no_op,
  min,
  max
};

template <typename data_type>
TfLiteStatus TFLiteOperation( TfLiteContext* context, TfLiteNode* node,
                              const OpContext& op_context,
                              basic_op operation);

template <typename T, int N = 4>
int MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const T* input1_data,
                             const RuntimeShape& unextended_input2_shape, const T* input2_data,
                             const RuntimeShape& unextended_output_shape,       T* output_data,
                             hifi::basic_op op);

template <typename data_type>
int ExecElemKernel( hifi::basic_op op, data_type *out_data,
                    const data_type *in1_data, const data_type *in2_data, size_t num_Elements);

}  // namespace hifi

/* Execute an element-wise kernel depending on the type specified by elem_op.
 * For now, the only data_type supported is int8_t
 * hifi::basic_op::min -> xa_nn_elm_min_8x8_8
 * hifi::basic_op::max -> xa_nn_elm_max_8x8_8
 */
template <typename data_type>
int hifi::ExecElemKernel( hifi::basic_op elem_op, data_type *out_data,
                          const data_type *in1_data, const data_type *in2_data,
                          size_t num_Elements) {
  int err = 0;
  
  if(!std::is_same<data_type, int8_t>::value){
      err = -1;
  } else {
    switch(elem_op){
      case hifi::basic_op::min :
        err = xa_nn_elm_min_8x8_8(out_data, in1_data, in2_data, num_Elements);
        break;
      case hifi::basic_op::max :
        err = xa_nn_elm_max_8x8_8(out_data, in1_data, in2_data, num_Elements);
        break;
      default :
        err = -1;
    }
  }

  return err;
}

template <typename data_type, typename op_type>
void TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context) {
  reference_ops::MaximumMinimumBroadcastSlow(
      tflite::micro::GetTensorShape(op_context.input1),
      tflite::micro::GetTensorData<data_type>(op_context.input1),
      tflite::micro::GetTensorShape(op_context.input2),
      tflite::micro::GetTensorData<data_type>(op_context.input2),
      tflite::micro::GetTensorShape(op_context.output),
      tflite::micro::GetTensorData<data_type>(op_context.output),
      op_type::template op<data_type>);
}

/*
 * Invoke element-wise kernels if the shapes match, else,
 * call hifi::MaximumMinimumBroadcast(...)
 */
template <typename data_type>
TfLiteStatus hifi::TFLiteOperation(
    TfLiteContext* context, TfLiteNode* node,
    const OpContext& op_context, hifi::basic_op elem_op) {

  int err = 0;

  const RuntimeShape &unextended_input1_shape = tflite::micro::GetTensorShape(op_context.input1),
                     &unextended_input2_shape = tflite::micro::GetTensorShape(op_context.input2),
                     &unextended_output_shape = tflite::micro::GetTensorShape(op_context.output);

  if (unextended_input1_shape == unextended_input2_shape) {
    err = hifi::ExecElemKernel<data_type>( elem_op,
                    tflite::micro::GetTensorData<data_type>(op_context.output),
                    tflite::micro::GetTensorData<data_type>(op_context.input1),
                    tflite::micro::GetTensorData<data_type>(op_context.input2),
                    unextended_output_shape.FlatSize());
  } else {
    err = hifi::MaximumMinimumBroadcast(
            tflite::micro::GetTensorShape(op_context.input1), tflite::micro::GetTensorData<data_type>(op_context.input1),
            tflite::micro::GetTensorShape(op_context.input2), tflite::micro::GetTensorData<data_type>(op_context.input2),
            tflite::micro::GetTensorShape(op_context.output), tflite::micro::GetTensorData<data_type>(op_context.output),
            elem_op);
  }

  TF_LITE_ENSURE_EQ(context, err, 0);

  return kTfLiteOk;
}

template <typename data_type, int N>
int hifi::MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const data_type *input1_data,
                                   const RuntimeShape& unextended_input2_shape, const data_type *input2_data,
                                   const RuntimeShape& unextended_output_shape,       data_type *output_data,
                                   hifi::basic_op op) {

  int err = 0;
  const int numDims = (N<=4) ? 4 :
                          (N<=8) ? 8 : 0;

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

  const RuntimeShape extended_input1_shape = RuntimeShape::ExtendedShape(numDims, unextended_input1_shape);
  const RuntimeShape extended_input2_shape = RuntimeShape::ExtendedShape(numDims, unextended_input2_shape);
  const RuntimeShape extended_output_shape = RuntimeShape::ExtendedShape(numDims, unextended_output_shape);

  NdArrayDesc<numDims> input1_desc;
  NdArrayDesc<numDims> input2_desc;
  NdArrayDesc<numDims> output_desc;

  NdArrayDescsForElementwiseBroadcast( unextended_input1_shape, unextended_input2_shape,
                                                  &input1_desc,            &input2_desc );

  CopyDimsToDesc(extended_output_shape, &output_desc);

  if ( (extended_input1_shape != extended_output_shape) &&
       (extended_input2_shape == extended_output_shape)     ) {         // input 1 needs broadcast

    err = xa_nn_broadcast_8_8(output_data, output_desc.extents,                       
            input1_data, extended_input1_shape.DimsData(), N);          // broadcast input 1 into output_data buffer

    err |= hifi::ExecElemKernel(op, output_data,                        // exec element-wise op after bcast
                   output_data, input2_data,
                   extended_output_shape.FlatSize());

  } else if( (extended_input1_shape == extended_output_shape) &&
             (extended_input2_shape != extended_output_shape)     ) {   // input 2 needs broadcast

    err = xa_nn_broadcast_8_8(output_data, output_desc.extents,                        
            input2_data, extended_input2_shape.DimsData(), N);          // broadcast input 2 into output_data buffer

    err |= hifi::ExecElemKernel(op, output_data,                        // exec element-wise op after bcast
                   input1_data, output_data,
                   extended_output_shape.FlatSize());
  } else {

    /* Both inputs need broadcast.
     * Call any of the xa_nn_elm_[min,max]_[4D,8D]_Bcast_8x8_8(...) kernels.
     * All these kernels have same the same function signature.
     */

    decltype(&xa_nn_elm_min_4D_Bcast_8x8_8) kernel = NULL;

    if(numDims == 4){
      kernel = (op == hifi::basic_op::min) ? xa_nn_elm_min_4D_Bcast_8x8_8 :
                 (op == hifi::basic_op::max) ? xa_nn_elm_max_4D_Bcast_8x8_8 : NULL;
    }else if(numDims == 8){
      kernel = (op == hifi::basic_op::min) ? xa_nn_elm_min_8D_Bcast_8x8_8 :
                 (op == hifi::basic_op::max) ? xa_nn_elm_max_8D_Bcast_8x8_8 : NULL;
    }

    if(kernel!=NULL){
      err = kernel(output_data, output_desc.extents,
                   input1_data, input1_desc.strides,
                   input2_data, input2_desc.strides );
    }
  }

  return err;
}

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  if (kernel_type == kReference || kernel_type == kHiFi) {

    hifi::basic_op operation =
        std::is_same<OpType, MinimumOp>::value ? hifi::basic_op::min :
            std::is_same<OpType, MaximumOp>::value ? hifi::basic_op::max :
                hifi::basic_op::no_op;

    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteUInt8:
        TFLiteOperation<uint8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt8:
        if (kernel_type == kHiFi) {
          hifi::TFLiteOperation<int8_t>(context, node, op_context, operation);
        } else {
          TFLiteOperation<int8_t, OpType>(context, node, op_context);
        }
        break;
      case kTfLiteInt32:
        TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Type %s (%d) is not supported by Maximum/Minimum.",
                           TfLiteTypeGetName(op_context.output->type),
                           op_context.output->type);
        return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Kernel type not supported by Maximum/Minimum.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace maximum_minimum

TfLiteRegistration Register_MAXIMUM() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/
#if defined(HIFI5)
          maximum_minimum::Eval<maximum_minimum::kHiFi,
                                maximum_minimum::MaximumOp>,
#else
          maximum_minimum::Eval<maximum_minimum::kReference,
                                maximum_minimum::MaximumOp>,
#endif // defined(HIFI5)
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MINIMUM() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/
#if defined(HIFI5)
          maximum_minimum::Eval<maximum_minimum::kHiFi,
                                maximum_minimum::MinimumOp>,
#else
          maximum_minimum::Eval<maximum_minimum::kReference,
                                maximum_minimum::MinimumOp>,
#endif // defined(HIFI5)
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
