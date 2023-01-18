"""Test space_to_batch by comparing micro and lite outputs."""


import numpy as np
from tensorflow import keras
import sys
import tensorflow as tf
from tflite_micro.python.tflite_micro import versus_lite


def _test(input_shape, block_shape, paddings):
    def _build_model():
        input = keras.Input(shape=input_shape, batch_size=1)
        output = tf.nn.space_to_batch(input, block_shape=block_shape, paddings=paddings)
        model = keras.Model(input, output)
        model.compile()
        return model

    model = _build_model()

    versus_lite.crosscheck(keras_model=model)


def test_no_padding():
    _test(input_shape=(4, 4, 1), block_shape=(2, 2), paddings=((0, 0), (0, 0)))


def test_padding():
    _test(input_shape=(4, 1, 3), block_shape=(2, 2), paddings=((0, 0), (0, 1)))


if __name__ == "__main__":
    import pytest
    code = pytest.main([
        "--import-mode=importlib",
        "-rA", # summarize all including passes
        __file__,
        ])
    sys.exit(code)
