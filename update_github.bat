@echo off
echo ==========================================
echo    ATL4S FOREX - GITHUB UPDATE (v4.0)
echo ==========================================

cd /d d:\Atl4s-Forex

echo [1/5] Checking Git Status...
git status

echo [2/5] Switching to Branch v4.0...
git checkout -b v4.0 2>nul || git checkout v4.0

echo [3/5] Adding Files...
git add .

echo [4/5] Committing Changes...
set /p msg="Enter Commit Message (default: 'Auto-Update v4.0'): "
if "%msg%"=="" set msg="Auto-Update v4.0"
git commit -m "%msg%"

echo [5/5] Pushing to Remote...
git push origin v4.0

echo ==========================================
echo    UPDATE COMPLETE
echo ==========================================
pause
