# Run week3 model with WRDS (2025 data) - uses WRDS_USERNAME and WRDS_PASSWORD from terminal
# No .pgpass; credentials from env vars only.
#
# Usage: In this same PowerShell window:
#   cd c:\Users\ocean\anaconda_projects\STAT-4830-OSO
#   $env:WRDS_USERNAME = "your_username"
#   $env:WRDS_PASSWORD = "your_password"
#   .\run_week3_wrds.ps1

if (-not $env:WRDS_USERNAME -or -not $env:WRDS_PASSWORD) {
    Write-Host "ERROR: Set WRDS_USERNAME and WRDS_PASSWORD first:" -ForegroundColor Red
    Write-Host '  $env:WRDS_USERNAME = "your_username"' -ForegroundColor Yellow
    Write-Host '  $env:WRDS_PASSWORD = "your_password"' -ForegroundColor Yellow
    exit 1
}

Write-Host "Running week3_implementation.ipynb with WRDS (2025 data)..." -ForegroundColor Cyan
& "C:\Users\ocean\anaconda3\python.exe" -m jupyter nbconvert --to notebook --execute --inplace notebooks/week3_implementation.ipynb --ExecutePreprocessor.timeout=1200
if ($LASTEXITCODE -eq 0) { Write-Host "Done." -ForegroundColor Green } else { Write-Host "Failed." -ForegroundColor Red }
