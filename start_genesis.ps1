# Genesis Quick Start (PowerShell)

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   GENESIS TRADING SYSTEM - Quick Start" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

Write-Host "   Select Mode:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   1. Paper Trading (recommended first)"
Write-Host "   2. Run Backtest Analysis"
Write-Host "   3. Generate Dashboard"
Write-Host "   4. Test Telegram Connection"
Write-Host "   5. Show System Status"
Write-Host ""

$choice = Read-Host "Enter choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host "Starting Paper Trading..." -ForegroundColor Green
        python genesis_live.py paper
    }
    "2" {
        Write-Host "Running Backtest..." -ForegroundColor Green
        python genesis_live.py backtest
    }
    "3" {
        Write-Host "Generating Dashboard..." -ForegroundColor Green
        python genesis_live.py dashboard
    }
    "4" {
        Write-Host "Testing Telegram..." -ForegroundColor Green
        python test_telegram.py
    }
    "5" {
        Write-Host "System Status:" -ForegroundColor Green
        python -c "import sys; sys.path.insert(0, '.'); from analytics import *; print('Analytics Suite Version:', __version__); print('Total Components:', len(__all__)); print(); print('Components:'); [print(f'  - {x}') for x in __all__]"
    }
}

Write-Host ""
Read-Host "Press Enter to exit"
