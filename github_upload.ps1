
# GitHub Upload Script for Atl4s-Forex Singularity

Write-Host "INITIATING OMEGA UPLOAD SEQUENCE..." -ForegroundColor Cyan

# 1. Initialize/Check Git
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git Repository..."
    git init
}

# 2. Checkout Branch
$branch = "laplace-v0.1"
Write-Host "Checking out branch: $branch"
git checkout -b $branch 2>$null
if ($LASTEXITCODE -ne 0) {
    # Branch might exist, switch to it
    git checkout $branch
}

# 3. Add Files
Write-Host "Staging Files..."
git add .

# 4. Commit
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "Auto agi update - $timestamp"
Write-Host "Committing: $msg"
git commit -m "$msg"

# 5. Push
Write-Host "Pushing to Remote..."
git push -u origin $branch

Write-Host "UPLOAD COMPLETE." -ForegroundColor Green
Write-Host "The Singularity is now in the cloud."
