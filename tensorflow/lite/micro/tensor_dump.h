#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifndef TENSOR_DUMP_H
#define TENSOR_DUMP_H

#define FILE_DUMP(output, act_dump_count) \
{ \
    float *out_ptr = tflite::micro::GetTensorData<float>(output); \
    const int output_size = tflite::micro::GetTensorShape(output).FlatSize(); \
    size_t out_element_size = 0; \
    char fname[150];\
    TfLiteTypeSizeOf(output->type, &out_element_size); \
    sprintf(fname, "net_0_out_0_frame_0_act_dump_%zu.bin", act_dump_count); \
    fprintf(stdout, "Layer dump: %s\n", fname); \
    fflush(stdout); \
    FILE *fp = fopen(fname, "wb"); \
    fwrite(out_ptr, out_element_size , output_size, fp); \
    fclose(fp); \
}

#endif // TENSOR_DUMP_H