# Setup script for CS2 Trajectory Transformer (PowerShell / Windows)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  CS2 Trajectory Transformer — Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python is not found in PATH. Please install Python 3.10+ from python.org." -ForegroundColor Red
    Exit 1
}

$pyVersion = python --version
Write-Host "[✓] Found $pyVersion" -ForegroundColor Green

# 2. Create Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[✓] Existing virtual environment found." -ForegroundColor Green
}

# 3. Activate venv
Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# 4. Upgrade pip and install wheel
Write-Host "[*] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# 5. Install PyTorch with CUDA or CPU
Write-Host "[*] Installing PyTorch..." -ForegroundColor Yellow
# Try standard PyTorch install
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple

# 6. Install Project Dependencies
Write-Host "[*] Installing project requirements from requirements.txt..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 7. Verification Self-Test
Write-Host "[*] Running verification self-test..." -ForegroundColor Yellow
.\venv\Scripts\python.exe demo_sample.py

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Setup Complete! To activate your environment in the future:" -ForegroundColor Green
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
