# Run Bulk Training for Akashic Memory
# Automates training on all years available in historical_datas
$historical_dir = "D:\Atl4s-Forex\historical_datas"
$python_exe = ".\venv\Scripts\python.exe"
$script = "train_akashic_memory.py"

Write-Host "🚀 STARTING BULK TRAINING SEQUENCE (2000-2026)" -ForegroundColor Cyan
Write-Host "   Using Vectorized Trainer (Ultra Speed)" -ForegroundColor Gray

# Get all CSV files
$files = Get-ChildItem "$historical_dir\DAT_MT_GBPUSD_M1_*.csv" | Sort-Object Name

foreach ($file in $files) {
    Write-Host "`n------------------------------------------------------------------"
    Write-Host "🧠 Training on Year: $($file.Name)" -ForegroundColor Green
    Write-Host "------------------------------------------------------------------"
    
    # Run python script
    & $python_exe $script $file.FullName
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Training failed for $($file.Name)"
    } else {
        Write-Host "✅ Completed $($file.Name)" -ForegroundColor Cyan
    }
}

Write-Host "`n🎉 BULK TRAINING COMPLETE!" -ForegroundColor Magenta
