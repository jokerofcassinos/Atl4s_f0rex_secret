@echo off
title Atl4s-Forex Auto-Installer
color 0a
echo ===================================================
echo   Atl4s-Forex: One-Click Environment Setup
echo   (Python 3.9 + Requirements)
echo ===================================================
echo.

:: 1. Check if Python 3.9 is installed
python --version 2>NUL
if errorlevel 1 goto install_python

:: Check specific version (simple check)
python --version | findstr "3.9" >NUL
if errorlevel 1 goto install_python

echo [OK] Python 3.9 detected.
goto setup_venv

:install_python
echo [!] Python 3.9 not found or incorrect version.
echo [!] Downloading Python 3.9.13 (64-bit)...
curl -o python_installer.exe https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe

echo [!] Installing Python 3.9... (This may ask for Admin privileges)
:: Silent install, add to PATH, install pip
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if errorlevel 1 (
    echo [ERROR] Python installation failed. Please install Python 3.9 manually.
    pause
    exit /b
)
echo [OK] Python 3.9 Installed.
del python_installer.exe

:setup_venv
echo.
echo [!] Setting up Virtual Environment...
if exist venv (
    echo [!] venv already exists. Skipping creation.
) else (
    python -m venv venv
    echo [OK] venv created.
)

echo.
echo [!] Activating venv and Installing Dependencies...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
    echo [OK] Dependencies installed.
) else (
    echo [WARNING] requirements.txt not found!
)

echo.
echo ===================================================
echo   SETUP COMPLETE
echo   To run the bot: use 'run_bot.bat'
echo ===================================================
pause
