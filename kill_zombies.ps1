
# Script to Kill Zombie main.py Python Processes
# Careful: This kills ALL python processes that have "main.py" in the command line.

Write-Host "HUNTING ZOMBIES..." -ForegroundColor Red

# Find processes
$zombies = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*main.py*" }

if ($zombies) {
    foreach ($zombie in $zombies) {
        Write-Host "Killing Process ID: $($zombie.ProcessId) - $($zombie.CommandLine)" -ForegroundColor Yellow
        Stop-Process -Id $zombie.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All zombies neutralized." -ForegroundColor Green
} else {
    Write-Host "No zombies found." -ForegroundColor Gray
}
