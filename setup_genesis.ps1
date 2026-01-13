
# ══════════════════════════════════════════════════════════════════════════════
# Atl4s-Forex: Genesis Setup Protocol 🚀
# ══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

Write-Host "`n[INIT] Starting Atl4s-Forex Environment Setup..." -ForegroundColor Cyan

# 1. Verification of Python
Write-Host "[1/4] Verifying Python Environment..." -ForegroundColor Yellow
$pythonVersion = & python --version 2>&1
if ($pythonVersion -match "Python 3\.(9|10|11|12)") {
    Write-Host "✅ Python Found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python 3.9+ required. Found: $pythonVersion" -ForegroundColor Red
    Write-Host "Please install Python from python.org"
    exit
}

# 2. Virtual Environment Creation
Write-Host "[2/4] Initializing Virtual Environment (venv)..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ venv created." -ForegroundColor Green
} else {
    Write-Host "ℹ️ venv already exists, skipping creation." -ForegroundColor Gray
}

# 3. Dependency Installation
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\venv\Scripts\pip.exe install --upgrade pip
& .\venv\Scripts\pip.exe install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "❌ Error installing dependencies." -ForegroundColor Red
    exit
}

# 4. Final Directory Check
Write-Host "[4/4] Finalizing Directory Structure..." -ForegroundColor Yellow
$folders = @("data", "data/cache", "data/training", "logs", "reports")
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "✅ Created $folder" -ForegroundColor Gray
    }
}

Write-Host "`n[READY] Atl4s-Forex: Laplace Demon is ready to awaken!" -ForegroundColor Cyan
Write-Host "To start the system, run: .\venv\Scripts\python.exe main_laplace.py`n" -ForegroundColor Green
