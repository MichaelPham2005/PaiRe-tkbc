# Re-run preprocessing with corrected [-1, 1] normalization
# Run this before training with the updated model

Set-Location ..

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Re-preprocessing timestamps to [-1, 1] range" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

python preprocess_continuous_time.py

Write-Host ""
Write-Host "Preprocessing complete! Now you can train with:" -ForegroundColor Green
Write-Host "  .\scripts\train_continuous_pairre.ps1" -ForegroundColor Yellow
