
# Script to Kill Zombie backtest_engine.py Python Processes
# Careful: This kills ALL python processes that have "backtest_engine.py" in the command line.

Write-Host "HUNTING ZOMBIES..." -ForegroundColor Red

# Find processes
$zombies = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*backtest_engine.py*" }

if ($zombies) {
    foreach ($zombie in $zombies) {
        Write-Host "Killing Process ID: $($zombie.ProcessId) - $($zombie.CommandLine)" -ForegroundColor Yellow
        Stop-Process -Id $zombie.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All zombies neutralized." -ForegroundColor Green
}
else {
    Write-Host "No zombies found." -ForegroundColor Gray
}
