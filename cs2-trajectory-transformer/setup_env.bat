@echo off
echo ==========================================================
echo   CS2 Trajectory Transformer -- Environment Setup (Windows CMD)
echo ==========================================================

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+.
    exit /b 1
)

if not exist "venv" (
    echo [*] Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo [✓] Existing virtual environment found.
)

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip and installing dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
python -m pip install -r requirements.txt

echo [*] Running verification test...
python demo_sample.py

echo ==========================================================
echo   Setup Complete! To activate in the future:
echo     venv\Scripts\activate.bat
echo ==========================================================
pause
