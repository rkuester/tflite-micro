#!/bin/bash
# Convert Jupytext markdown format to .ipynb for Jupyter/Colab compatibility

# Check if jupytext is installed
if ! command -v jupytext &> /dev/null; then
    echo "jupytext not found. Installing..."
    pip install jupytext
fi

# Convert .md to .ipynb
echo "Converting mnist_compression.md to mnist_compression.ipynb..."
jupytext --to ipynb mnist_compression.md

echo "Conversion complete! The .ipynb file is ready for Jupyter or Colab."