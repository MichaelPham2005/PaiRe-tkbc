# Quick diagnostic test on current checkpoint
# Runs all comprehensive diagnostics to identify why model not learning

Write-Host "Running comprehensive diagnostics..." -ForegroundColor Green

# Find latest checkpoint
$checkpoint = "models\ICEWS14\checkpoint"
if (-not (Test-Path $checkpoint)) {
    Write-Host "No checkpoint found at $checkpoint" -ForegroundColor Red
    Write-Host "Train model first or specify correct checkpoint path" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nFound checkpoint: $checkpoint" -ForegroundColor Cyan

# Run diagnostics
python comprehensive_diagnostics.py `
    --dataset ICEWS14 `
    --checkpoint $checkpoint `
    --n_samples 500 `
    --device cuda

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Diagnostics complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nCheck the output above for:"
Write-Host "  1. Time discrimination accuracy (should be >55%)" -ForegroundColor Yellow
Write-Host "  2. Temporal sensitivity (score delta should be >0.01)" -ForegroundColor Yellow
Write-Host "  3. Ablation results (Full should beat Time-off)" -ForegroundColor Yellow
Write-Host "  4. Time embedding variation (variance should be >1e-4)" -ForegroundColor Yellow
Write-Host "`nIf all tests fail, model is NOT learning time!" -ForegroundColor Red
