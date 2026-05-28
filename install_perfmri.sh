#!/usr/bin/env bash

set -e

echo "======================================"
echo " PerfMRI installation"
echo "======================================"

# --------------------------------------
# Detect OS
# --------------------------------------

OS="$(uname)"

echo ""
echo "Detected OS: $OS"

# --------------------------------------
# macOS dependencies
# --------------------------------------

if [[ "$OS" == "Darwin" ]]; then

    echo ""
    echo "Installing Python 3.10 and Tk (Homebrew)..."

    if ! command -v brew &> /dev/null; then
        echo "ERROR: Homebrew is not installed."
        echo "Install Homebrew first:"
        echo "https://brew.sh"
        exit 1
    fi

    brew update
    brew install python@3.10 python-tk@3.10

fi

# --------------------------------------
# Linux dependencies
# --------------------------------------

if [[ "$OS" == "Linux" ]]; then

    echo ""
    echo "Installing Python 3.10 and Tk (apt)..."

    sudo apt update
    sudo apt install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-tk \
        git

fi

# --------------------------------------
# Create virtual environment
# --------------------------------------

echo ""
echo "Creating virtual environment..."

python3.10 -m venv perfmri_env

# --------------------------------------
# Activate environment
# --------------------------------------

echo ""
echo "Activating environment..."

source perfmri_env/bin/activate

# --------------------------------------
# Upgrade pip
# --------------------------------------

echo ""
echo "Upgrading pip..."

python -m pip install --upgrade pip

# --------------------------------------
# Install requirements
# --------------------------------------

echo ""
echo "Installing PerfMRI dependencies..."

pip install -r requirements.txt

# --------------------------------------
# Install nipy from GitHub
# --------------------------------------

echo ""
echo "Installing nipy 0.6.1 from GitHub..."

pip install git+https://github.com/nipy/nipy.git@0.6.1

# --------------------------------------
# Optional packages
# --------------------------------------

echo ""
echo "Optional package:"
echo "  pip install antspyx"

# --------------------------------------
# Sanity check
# --------------------------------------

echo ""
echo "Running sanity check..."

python -c "import numpy, matplotlib, tkinter, nipy; print('PerfMRI environment OK')"

echo ""
echo "======================================"
echo " Installation completed successfully"
echo "======================================"
echo ""
echo "Run PerfMRI with:"
echo "./run_perfmri.sh"
echo ""
