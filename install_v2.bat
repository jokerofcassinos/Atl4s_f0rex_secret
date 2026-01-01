@echo off
title Atl4s-Forex v2.0 - Installer
color 0A

echo ===================================================
echo      Atl4s-Forex v2.0 | AUTOMATED INSTALLER
echo ===================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org
    pause
    exit
)

echo [OK] Python found.
python --version
echo.

:: 2. Create Virtual Environment (Optional but recommended)
set /p use_venv="Do you want to create a new Virtual Environment (venv)? [Y/N] (Default: Y): "
if /i "%use_venv%" neq "N" (
    echo.
    echo [INFO] Creating venv...
    python -m venv venv
    echo [OK] venv created.
    echo [INFO] Activating venv...
    call venv\Scripts\activate.bat
)

:: 3. Install Dependencies
echo.
echo [INFO] Installing Dependencies from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Please check your internet connection or python version.
    pause
    exit
)

echo.
echo [SUCCESS] Installation Complete!
echo.
echo To run the bot:
echo 1. Ensure MT5 is open.
echo 2. Run: python main.py
echo.
pause
