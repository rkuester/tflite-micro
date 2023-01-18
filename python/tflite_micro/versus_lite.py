"""Test micro using lite as a baseline.""" 

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


def _get_input_shape(keras_model):
    """Return the input shape of a keras model.

    Return the input shape of the given keras model, i.e., the shape of first
    layer's first tensor, ignoring the batch dimension.
    """

    return keras_model.layers[0].input_shape[0][1:]


def _make_int8_tflite_model(keras_model, rng):
    """Convert a keras model to an int8 tflite model.

    Convert the keras model to an int8 tflite model using random data for
    quantization.
    """

    def random_representative_dataset():
        shape = _get_input_shape(keras_model)
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


def crosscheck(keras_model=None, tflite_model=None, tflite_path=None,
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
    print(f"input: shape={input.shape}\n{input}\n")

    lite.set_tensor(lite.get_input_details()[0]["index"], input)
    lite.invoke()
    lite_output = lite.get_tensor(lite.get_output_details()[0]["index"])
    print(f"lite_output: shape={lite_output.shape}\n{tf.convert_to_tensor(lite_output)}\n")

    micro = tflm_runtime.Interpreter.from_bytes(model)
    micro.set_input(input, 0)
    micro.invoke()
    micro_output = micro.get_output(0)
    print(f"micro_output: shape={micro_output.shape}\n{np.array2string(micro_output)}\n")

    diff = (tf.convert_to_tensor(lite_output, dtype=tf.int16) - 
           tf.convert_to_tensor(micro_output, dtype=tf.int16))
    print(f"diff:\n{diff}")

    if not np.allclose(micro_output, lite_output, atol=1):
        raise AssertionError("tflite and micro interpreter outputs don't match\n")


def main(tflite_filename):
    crosscheck(tflite_path=tflite_filename)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1]))
