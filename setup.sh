#!/bin/bash

# AskMyDocs Setup Script
echo "Setting up AskMyDocs..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n askmydocs python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate askmydocs

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the application:"
echo "  conda activate askmydocs"
echo "  streamlit run app.py"
echo ""
echo "Make sure Ollama is installed: https://ollama.ai/"