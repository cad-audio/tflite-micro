#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GELU_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GELU_H_


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"


namespace tflite {


extern const int kInputTensor;
extern const int kOutputTensor;


TfLiteStatus CalculateOpDataGelu(TfLiteContext* context, TfLiteNode* node);


TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node);


}  // namespace tflite


#endif  // TENSORFLOW_LITE_MICRO_KERNELS_GELU_H_