wav2vec2_float_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/model_settings.cc

wav2vec2_float_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/model_settings.h

wav2vec2_float_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/wav2vec2_float_test.cc \
$(wav2vec2_float_MODEL_SRCS)

wav2vec2_float_TEST_HDRS := \
$(wav2vec2_float_MODEL_HDRS)


wav2vec2_float_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/wav2vec2_dynamic.cc \
$(wav2vec2_float_MODEL_SRCS)

wav2vec2_float_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/wav2vec2_dynamic.h \
$(wav2vec2_float_MODEL_HDRS)



#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/wav2vec2_float/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,wav2vec2_float_test,\
  $(wav2vec2_float_TEST_SRCS),$(wav2vec2_float_TEST_HDRS),,))
endif


# Builds a standalone object recognition binary.
$(eval $(call microlite_test,wav2vec2_float,\
$(wav2vec2_float_SRCS),$(wav2vec2_float_HDRS)))

list_wav2vec2_float_example_sources:
	@echo $(wav2vec2_float_SRCS)

list_wav2vec2_float_example_headers:
	@echo $(wav2vec2_float_HDRS)