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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace py = pybind11;

const bool dump_buffers = true;

void dump_model_buffers(const char* tflite_file_name) {
  FILE *model_file = fopen(tflite_file_name, "rb"); 
  fseek(model_file, 0L, SEEK_END);
  unsigned int model_size = ftell(model_file);
  rewind(model_file);
 
  uint8_t *model_buf = (uint8_t *)malloc(model_size+63);
  int offset = ((uintptr_t)model_buf) & 63;
  uint8_t *aligned_model_buf = model_buf + offset;
  fread(aligned_model_buf, model_size, 1, model_file);
  
  const tflite::Model* model = tflite::GetModel(aligned_model_buf);
  
  fprintf(stderr, "Model Base Address(64 bytes aligned) %p\n", aligned_model_buf);
  fprintf(stderr, "Model Size %u bytes\n", model_size);

  // This is a pointer to a vector of offsets:
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      model->buffers();
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>&
      buffer_offsets = *buffers;
  int number_of_buffers = buffer_offsets.size();
  fprintf(stderr, "number of buffers: %d\n", buffer_offsets.size());
  for (int i = 0; i < number_of_buffers; ++i) {
    // C++ magic returns the actual buffer pointer here, rather than the
    // expected Offset that the Vector seems to hold:
    const tflite::Buffer* buffer = buffer_offsets[i];
    const flatbuffers::Vector<uint8_t>* data = buffer->data();
    // Only the weight buffers are allocated in the flatbuffer:
    if (data) {
      size_t buffer_size = data->size();
      const uint8_t* buffer_addr = data->Data();
      int buffer_offset = buffer_addr - reinterpret_cast<const uint8_t*>(aligned_model_buf);
      fprintf(stderr, "model buffer %3d size: %6zu, addr: %p, offset: 0x%x\n", i,
              buffer_size, buffer_addr, buffer_offset);
      //fprintf(stderr, "buffer contents: %x %x %x %x %x %x %x %x\n",
      //        buffer_addr[0], buffer_addr[1], buffer_addr[2], buffer_addr[3],
      //        buffer_addr[4], buffer_addr[5], buffer_addr[6], buffer_addr[7]);
    }
  }
  free(model_buf);
  fclose(model_file);
}

void align_tflite_model(const char* input_file_name,
                        const char* output_file_name) {
  std::string model_file;
  // Read the file into a string using the included util API call:
  flatbuffers::LoadFile(input_file_name, false, &model_file);
  // Parse the string into a C++ class.  Model is the root object of a tflite
  // flatbuffer file.
  const tflite::Model* model = tflite::GetModel(model_file.c_str());
  // A packed model is basically the file format mmaped into memory.
  // Unpacking it and then packing it with the C++ API should yield
  // a file with the force_align attributes respected.
  // ModelT is just the unpacked version of the model file.
  tflite::ModelT* unpacked_model = model->UnPack();
  flatbuffers::FlatBufferBuilder fbb;
  auto new_model = tflite::Model::Pack(fbb, unpacked_model);
  fbb.Finish(new_model, tflite::ModelIdentifier());
  flatbuffers::SaveFile(output_file_name,
                        reinterpret_cast<char*>(fbb.GetBufferPointer()),
                        fbb.GetSize(), /*binary*/ true);
  
  if(dump_buffers)
  {
    fprintf(stderr, "Input Model %s\n", input_file_name);
    dump_model_buffers(input_file_name);
    fprintf(stderr, "\nOutput Model %s\n", output_file_name);
    dump_model_buffers(output_file_name);
  }
}

PYBIND11_MODULE(tflite_flatbuffer_align_wrapper, m) {
  m.doc() = "tflite_flatbuffer_align_wrapper";
  m.def("align_tflite_model", &align_tflite_model,
        "Aligns the tflite flatbuffer to (16), by unpacking and repacking via "
        "the flatbuffer C++ API.",
        py::arg("input_file_name"), py::arg("output_file_name"));
}
