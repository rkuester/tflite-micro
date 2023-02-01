# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test space_to_batch by comparing micro and lite outputs."""

from tensorflow import keras, nn, test
from tflite_micro.python.tflite_micro.test_utils import crosscheck


class TestPadding(test.TestCase):

  def _test(self, input_shape, block_shape, paddings):

    def _build_model():
      input = keras.Input(shape=input_shape, batch_size=1)
      output = nn.space_to_batch(input,
                                 block_shape=block_shape,
                                 paddings=paddings)
      model = keras.Model(input, output)
      model.compile()
      return model

    model = _build_model()
    is_match = crosscheck.versus_lite(keras_model=model)
    self.assertTrue(is_match, msg=is_match)

  def testWithoutPadding(self):
    self._test(input_shape=(4, 4, 1),
               block_shape=(2, 2),
               paddings=((0, 0), (0, 0)))

  def testWithPadding(self):
    self._test(input_shape=(4, 1, 3),
               block_shape=(2, 2),
               paddings=((0, 0), (0, 1)))


if __name__ == "__main__":
  test.main()
