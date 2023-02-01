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
"""Test micro using lite as a baseline."""

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


class Result:

  def __init__(self, input, micro, lite):
    self.input = input
    self.micro = micro
    self.lite = lite
    self.diff = None
    self.matches = None

  def micro_output(self):
    return self.micro.get_output(0)

  def lite_output(self):
    index = self.lite.get_output_details()[0]["index"]
    return self.lite.get_tensor(index)

  def diff_output(self):
    if self.diff is None:
      self.diff = (tf.convert_to_tensor(self.micro_output(), dtype=tf.int16) -
                   tf.convert_to_tensor(self.lite_output(), dtype=tf.int16))
    return self.diff

  def output_matches(self):
    if self.matches is None:
      self.matches = np.allclose(self.micro_output(),
                                 self.lite_output(),
                                 atol=1)
    return self.matches

  def __bool__(self):
    return self.output_matches()

  def __str__(self):
    return f"input: shape={self.input.shape}\n" \
           f"{self.input}\n" \
           f"\n" \
           f"lite_output: shape={self.lite_output().shape}\n" \
           f"{tf.convert_to_tensor(self.lite_output())}\n" \
           f"\n" \
           f"micro_output: shape={self.micro_output().shape}\n" \
           f"{tf.convert_to_tensor(self.micro_output())}\n" \
           f"\n" \
           f"diff: shape={self.diff_output().shape}\n" \
           f"{self.diff_output()}"


def _make_int8_tflite_model(keras_model, rng):
  """Convert a keras model to an int8 tflite model.

    Convert the keras model to an int8 tflite model using random data for
    quantization.
    """

  def random_representative_dataset():
    shape = keras_model.layers[0].input_shape[0][1:]
    for _ in range(100):
      data = rng.random(size=(1, *shape), dtype=np.float32)
      yield [data]

  converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
  converter.representative_dataset = random_representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  return tflite_model


def versus_lite(keras_model=None,
                tflite_model=None,
                tflite_path=None,
                tflm_interpreter=tflm_runtime.Interpreter,
                rng=np.random.default_rng(seed=42)):
  """Test the micro interpreter against the lite interpreter.

    Automatically test the micro interpreter using the lite interpreter as a
    baseline. Feed the same random input to both and expect equivalent outputs.
    Raise an AssertionError if the outputs differ, else return None.

    Pass one of:
        keras_model  -- automatically converted and quantized
        tflite_model -- used as-is
        tflite_path  -- read from file

    rng -- optionally pass an np.random.Generator to control seeding
    """

  if keras_model:
    model = _make_int8_tflite_model(keras_model, rng)
  elif tflite_model:
    model = tflite_model
  elif tflite_path:
    with open(tflite_path, mode='rb') as f:
      model = f.read()

  lite = tf.lite.Interpreter(model_content=model)
  lite.allocate_tensors()
  input_shape = lite.get_input_details()[0]["shape"][1:]

  input = rng.integers(-128, 127, size=(1, *input_shape), dtype=np.int8)

  lite.set_tensor(lite.get_input_details()[0]["index"], input)
  lite.invoke()

  micro = tflm_interpreter.from_bytes(model)
  micro.set_input(input, 0)
  micro.invoke()

  return Result(input, micro, lite)
