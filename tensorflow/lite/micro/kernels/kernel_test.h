
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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_TEST_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_TEST_H_

#include "absl/types/span.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {

template <typename Params>
void ExpectGolden(
    const TFLMRegistration& registration,
    Params& params,
    const absl::Span<TfLiteTensor> tensors,
    TfLiteIntArray* input_indices,
    TfLiteIntArray* output_indices,
    const TfLiteTensor& golden) {

  tflite::micro::KernelRunner runner{
      registration,
      tensors.data(),
      static_cast<int>(tensors.size()),
      input_indices,
      output_indices,
      &params
  };

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const TfLiteTensor output = tensors[2]; // TODO: do for all output_indices
  TF_LITE_MICRO_EXPECT(TfLiteIntArrayEqual(output.dims, golden.dims));
  for (int i = 0; i < ElementCount(*output.dims); ++i) {
    constexpr float tolerance = 1e-5; // TODO: make configurable
    TF_LITE_MICRO_EXPECT_NEAR(golden.data.f[i], output.data.f[i], tolerance); // TODO: dynamically find datatype
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_TEST_H_
