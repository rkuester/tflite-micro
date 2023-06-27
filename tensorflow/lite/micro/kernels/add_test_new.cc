#include "tensorflow/lite/micro/kernels/add.h"

#include <array>

#include "tensorflow/lite/micro/kernels/kernel_test.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

using tflite::TestTensor;
using tflite::ExpectGolden;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNoActivation) {
  TestTensor input1 {Shape{2, 2}, FlatData{11.1f, 11.1f, 11.1f, 11.1f}};
  TestTensor input2 {Shape{2, 2}, FlatData{22.2f, 22.2f, 22.2f, 22.2f}};
  TestTensor golden {Shape{2, 2}, FlatData{33.3f, 33.3f, 33.3f, 33.3f}};
  TestTensor output {SameCapacity{golden}};

  auto registration = tflite::Register_ADD();
  TfLiteAddParams params = {kTfLiteActNone};

  ExpectGolden(
      registration,
      params,
      {input1, input2},
      {output},
      {golden});
}

TF_LITE_MICRO_TESTS_END
