# TFLM Compression Tutorial

This directory contains a Jupyter notebook demonstrating TensorFlow Lite for
Microcontrollers compression features, including:

- Weight clustering for reducing unique weight values
- Post-training quantization to INT8
- TFLM's Look-Up Table (LUT) compression for clustered layers
- Combined compression pipeline achieving significant model size reduction

## Source Format

The notebook is maintained in Jupytext markdown format (`mnist_compression.md`) for
easier editing and reviewing in text editors and on GitHub. To generate the `.ipynb` 
file for Jupyter or Google Colab:

```bash
./convert.sh
```

## Running the Notebook

### Option 1: Run Locally with JupyterLab

```bash
# Install JupyterLab with Jupytext and optimization libraries
pip install jupyterlab jupytext tensorflow numpy matplotlib tf-keras tensorflow-model-optimization

# Build TFLM Python wheel (if not already built)
bazel build //python/tflite_micro:whl.dist
pip install bazel-bin/python/tflite_micro/whl.dist/tflite_micro-0.dev0-py3-none-any.whl
cd tensorflow/lite/micro/compression/tutorial

# Open the .md file directly in JupyterLab
jupyter lab mnist_compression.md
```

JupyterLab will automatically recognize the markdown format and display it as a notebook.

### Option 2: Open in Google Colab

Only the .md file is committed to GitHub (no .ipynb), but you can still use it in Colab:

1. Open a new Colab notebook
2. Run this setup cell:

```python
# Install required packages and clone the repo
!pip install jupytext tf-keras tensorflow-model-optimization
!git clone https://github.com/tensorflow/tflite-micro.git
%cd tflite-micro/tensorflow/lite/micro/compression/tutorial

# Convert .md to .ipynb
!jupytext --to ipynb mnist_compression.md

# Display link to open the converted notebook
from IPython.display import display, HTML
display(HTML('<a href="/notebooks/tflite-micro/tensorflow/lite/micro/compression/tutorial/mnist_compression.ipynb" target="_blank">Click here to open the converted notebook</a>'))
```

3. Click the link to open the converted notebook in a new tab
