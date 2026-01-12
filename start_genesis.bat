@echo off
echo.
echo ====================================================
echo   GENESIS TRADING SYSTEM - Quick Start
echo ====================================================
echo.
echo   Select Mode:
echo.
echo   1. Paper Trading (recommended first)
echo   2. Run Backtest Analysis
echo   3. Generate Dashboard
echo   4. Test Telegram Connection
echo   5. Full System Status
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo Starting Paper Trading...
    call venv\Scripts\activate
    python genesis_live.py paper
)

if "%choice%"=="2" (
    echo Running Backtest...
    call venv\Scripts\activate
    python genesis_live.py backtest
)

if "%choice%"=="3" (
    echo Generating Dashboard...
    call venv\Scripts\activate
    python genesis_live.py dashboard
)

if "%choice%"=="4" (
    echo Testing Telegram...
    call venv\Scripts\activate
    python test_telegram.py
)

if "%choice%"=="5" (
    echo Checking System Status...
    call venv\Scripts\activate
    python -c "from analytics import *; print('Analytics Suite:', __version__); print('Components:', len(__all__))"
)

echo.
pause
