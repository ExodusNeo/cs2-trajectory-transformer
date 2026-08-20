#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  CS2 Trajectory Transformer — Environment Setup (Linux/macOS)"
echo "=========================================================="

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[✓] Found $(python3 --version)"

if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment (venv)..."
    python3 -m venv venv
else
    echo "[✓] Existing virtual environment found."
fi

echo "[*] Activating virtual environment..."
source venv/bin/activate

echo "[*] Upgrading pip and installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install torch -r requirements.txt

echo "[*] Running verification test..."
python demo_sample.py

echo "=========================================================="
echo "  Setup Complete! To activate in the future:"
echo "    source venv/bin/activate"
echo "=========================================================="
