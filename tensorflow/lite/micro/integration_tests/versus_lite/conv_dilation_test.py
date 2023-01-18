"""Test convolutions with dilation by comparing micro and lite outputs."""

import numpy as np
from tensorflow import keras
import sys
import tensorflow as tf
from tflite_micro.python.tflite_micro import versus_lite


def _test_separable_conv(dilation_rate):
    input = keras.Input(shape=(16, 1, 3), batch_size=1)
    rng = np.random.default_rng(seed=314)
    seed = rng.integers(np.iinfo(np.int32).max)
    output = keras.layers.SeparableConv2D(
        filters=12,
        kernel_size=(3, 1),
        strides=(2, 2),
        dilation_rate=dilation_rate,
        padding='valid',
        use_bias=True,
        depthwise_initializer=keras.initializers.RandomUniform(-1., 1., seed),
        pointwise_initializer=keras.initializers.RandomUniform(-1., 1., seed + 1),
        )(input)
    model = keras.Model(input, output)
    model.compile()
    versus_lite.crosscheck(keras_model=model, rng=rng)


def test_separable_conv():
    _test_separable_conv(dilation_rate=1)


def test_separable_conv_with_dilation():
    _test_separable_conv(dilation_rate=2)


if __name__ == "__main__":
    import pytest
    code = pytest.main([
        "--import-mode=importlib",
        "-rA", # summarize all including passes
        __file__,
        ])
    sys.exit(code)
