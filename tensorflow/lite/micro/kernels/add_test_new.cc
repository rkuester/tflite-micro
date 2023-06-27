#include "tensorflow/lite/micro/kernels/add.h"

#include <array>

#include "tensorflow/lite/micro/kernels/kernel_test.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

using tflite::Shape;
using tflite::TestTensor;
using tflite::ExpectGolden;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(NotQuantized) {
  const TestTensor input1 {Shape{2, 2}, std::array{11.1f, 11.1f, 11.1f, 11.1f}};
  const TestTensor input2 {Shape{2, 2}, std::array{22.2f, 22.2f, 22.2f, 22.2f}};
  const TestTensor golden {Shape{2, 2}, std::array{33.3f, 33.3f, 33.3f, 33.3f}};
  TestTensor<float, 4> output {Shape{2, 2}};

  std::array tensors_in{
      input1,
      input2,
      output,
  };
  auto input_indices = tflite::testing::MakeIntArray(0, 1);
  auto output_indices = tflite::testing::MakeIntArray(2);

  auto registration = tflite::Register_ADD();
  TfLiteAddParams params = {kTfLiteActNone};

  ExpectGolden(
      registration,
      params,
      tensors,
      input_indices,
      output_indices,
      golden);
}

TF_LITE_MICRO_TESTS_END
