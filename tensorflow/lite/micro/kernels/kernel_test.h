
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

#include <cassert>
#include <initializer_list>

#include "absl/types/span.h"

#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {

class Shape {
 public:
  Shape(const std::initializer_list<int>& values) {
    assert(values.size() <= kMaxDims);
    std::copy(values.begin(), values.end(), dims_.begin() + 1);
    DimensionCount(values.size());
  }

  void DimensionCount(int set) { dims_[0] = set; }
  int DimensionCount() const { return dims_[0]; }

  operator TfLiteIntArray*() {
    return reinterpret_cast<TfLiteIntArray*>(dims_.data());
  }

 private:
  static constexpr int kMaxDims = 6;
  std::array<int, 1 + kMaxDims> dims_;
};

template <typename T, std::size_t FlatSize>
class TestTensor {
 public:
  TestTensor(const Shape& s, const std::array<T, FlatSize>& v)
      : shape_{s}, data_{v} {}

  TestTensor(const Shape& s)
      : shape_{s} {}

  Shape shape() const { return shape_; }

 private:
  Shape shape_;
  std::array<T, FlatSize> data_;
};

template <typename Params>
void ExpectGolden(
    const TFLMRegistration& registration,
    Params& params,
    const absl::Span<const TfLiteTensor> tensors,
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
