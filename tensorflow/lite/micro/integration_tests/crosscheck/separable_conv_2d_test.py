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
"""Test SeparableConv2D by comparing micro and lite outputs."""

import numpy as np
from tensorflow import keras, test
from tflite_micro.python.tflite_micro.test_utils import crosscheck


class DilationTest(test.TestCase):
  """Test SeparableConv2D with various dilations"""

  def _test(self, dilation_rate, input_batch_size=1):
    input = keras.Input(shape=(16, 1, 3), batch_size=input_batch_size)
    rng = np.random.default_rng(seed=314)
    seed = rng.integers(np.iinfo(np.int32).max)
    output = keras.layers.SeparableConv2D(
        filters=12,
        kernel_size=(3, 1),
        strides=(2, 2),
        dilation_rate=dilation_rate,
        padding="valid",
        use_bias=True,
        depthwise_initializer=keras.initializers.RandomUniform(-1., 1., seed),
        pointwise_initializer=keras.initializers.RandomUniform(
            -1., 1., seed + 1),
    )(input)
    model = keras.Model(input, output)
    model.compile()
    is_match = crosscheck.versus_lite(keras_model=model, rng=rng)
    self.assertTrue(is_match, msg=is_match)

  def test_without_dilation(self):
    self._test(dilation_rate=1)

  def test_with_dilation(self):
    self._test(dilation_rate=2)

  def test_with_dilation_with_dynamic_batch(self):
    """Test with dilation but with a dynamic batch size in the network input."""
    self._test(dilation_rate=2, input_batch_size=None)


if __name__ == "__main__":
  test.main()
