#!/usr/bin/env bash
# ==============================================================================
# setup_colab.sh - Fast Setup for Google Colab (T4 / L4 GPU)
# Preserves Colab's pre-installed PyTorch/CUDA and installs required packages
# ==============================================================================

set -e

echo "=========================================================="
echo " [1/3] Checking GPU & PyTorch Environment in Colab..."
echo "=========================================================="
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "=========================================================="
echo " [2/3] Installing Required Libraries (Lightweight & Fast)..."
echo "=========================================================="
pip install --quiet --upgrade pip
pip install --quiet \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "peft>=0.13.0" \
    "trl>=0.11.0" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils>=0.0.8" \
    "datasets>=2.18.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0" \
    "pillow>=9.5.0" \
    "pyarrow>=14.0.0" \
    "seaborn>=0.12.0" \
    "tqdm>=4.66.0"

echo "=========================================================="
echo " [3/3] Verifying Key Imports..."
echo "=========================================================="
python3 -c "import transformers, peft, bitsandbytes, qwen_vl_utils, sklearn; print('All key packages successfully verified!')"

echo "=========================================================="
echo " Setup Completed! You can now run:"
echo " !python -u run_experiment.py --quick_test"
echo "=========================================================="

