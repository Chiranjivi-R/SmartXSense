# SmartXSense Demo Startup Script for Windows PowerShell

$projectRoot = Resolve-Path "$PSScriptRoot\.."

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting SmartXSense Backend & Frontend Services" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Start Backend
Write-Host "Launching Flask Backend API on http://localhost:5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\backend'; python run.py"

# Start Frontend
Write-Host "Launching Vite Frontend App on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\frontend'; npm run dev"

Write-Host "`nBoth services launched in separate windows." -ForegroundColor White
Write-Host "Frontend URL: http://localhost:5173" -ForegroundColor Green
Write-Host "Backend URL:  http://localhost:5000/api/v1/health" -ForegroundColor Yellow
