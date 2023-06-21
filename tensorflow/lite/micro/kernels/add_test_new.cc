#include <array>

#include "absl/types/span.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_test.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Procedural) {
  auto dims = tflite::testing::MakeIntArray(2, 2);
  const std::array input1 = {11.1f, 11.1f, 11.1f, 11.1f};
  const std::array input2 = {22.2f, 22.2f, 22.2f, 22.2f};
  const std::array golden = {33.3f, 33.3f, 33.3f, 33.3f};
  const TfLiteTensor golden_tensor = tflite::testing::CreateTensor(golden.data(), dims);

  std::array<float, golden.size()> output;

  std::array tensors {
      tflite::testing::CreateTensor(input1.data(), dims),
      tflite::testing::CreateTensor(input2.data(), dims),
      tflite::testing::CreateTensor(output.data(), dims),
  };
  auto input_indices = tflite::testing::MakeIntArray(0, 1);
  auto output_indices = tflite::testing::MakeIntArray(2);

  auto registration = tflite::Register_ADD();
  TfLiteAddParams params = {kTfLiteActNone};

  tflite::ExpectGolden(
      registration,
      params,
      absl::Span<TfLiteTensor>{tensors},
      input_indices,
      output_indices,
      golden_tensor);
}

TF_LITE_MICRO_TESTS_END
