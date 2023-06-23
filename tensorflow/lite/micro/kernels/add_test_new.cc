#include <array>
#include <cassert>
#include <initializer_list>

#include "absl/types/span.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_test.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

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

// template <typename T, std::size_t N>
// TestTensor(const Shape&, const std::array<T, N>&) ->TestTensor<T, N>;

}

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(NotQuantized) {
  const TestTensor input1 {Shape{2, 2}, std::array{11.1f, 11.1f, 11.1f, 11.1f}};
  const TestTensor input2 {Shape{2, 2}, std::array{22.2f, 22.2f, 22.2f, 22.2f}};
  const TestTensor golden {Shape{2, 2}, std::array{33.3f, 33.3f, 33.3f, 33.3f}};
  TestTensor<float, 4> output {Shape{2, 2}};

  std::array tensors{
      input1,
      input2,
      output,
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
      golden);
}

TF_LITE_MICRO_TESTS_END
