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

from tensorflow import keras
from tensorflow import test
from tflite_micro.python.tflite_micro.test_utils import crosscheck
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime

example_model_path = "tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite"

class InputTest(test.TestCase):
  "Run with each supported type of model input."

  def test_tflite_path(self):
    is_match = crosscheck.versus_lite(tflite_path=example_model_path)
    self.assertTrue(is_match, msg=is_match)

  def test_tflite_model(self):
    with open(example_model_path, 'rb') as f:
      model = f.read()

    is_match = crosscheck.versus_lite(tflite_model=model)
    self.assertTrue(is_match, msg=is_match)

  def test_keras_model(self):
    first = keras.Input(shape=(16, 1, 3), batch_size=1)
    last = keras.layers.Dense(16, activation="relu")(first)
    model = keras.Model(first, last)
    model.compile()

    is_match = crosscheck.versus_lite(keras_model=model)
    self.assertTrue(is_match, msg=is_match)


class FailureTest(test.TestCase):

  class FakeInterpreter(tflm_runtime.Interpreter):
    "A fake which returns the wrong answers."

    def get_output(self, index):
      output = super().get_output(index)
      return output + 5

  def test_failure(self):
    is_match = crosscheck.versus_lite(tflite_path=example_model_path,
                                      tflm_interpreter=self.FakeInterpreter)
    self.assertFalse(is_match, msg=is_match)


if __name__ == "__main__":
  test.main()
