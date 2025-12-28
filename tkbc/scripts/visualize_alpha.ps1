# PowerShell script to visualize alpha weights after training
# Sử dụng: .\visualize_alpha.ps1

$CHECKPOINT = "models/ICEWS14/ContinuousPairRE/best_valid.pt"
$OUTPUT_DIR = "visualizations/alpha"

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*69) -ForegroundColor Cyan
Write-Host "VISUALIZE ALPHA WEIGHTS FROM TRAINED MODEL" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*69) -ForegroundColor Cyan

# Check if checkpoint exists
if (-Not (Test-Path $CHECKPOINT)) {
    Write-Host "`n❌ Error: Checkpoint not found at $CHECKPOINT" -ForegroundColor Red
    Write-Host "Please train the model first using: .\train_continuous_pairre.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✓ Found checkpoint: $CHECKPOINT" -ForegroundColor Green
Write-Host "`n🎨 Generating visualizations..." -ForegroundColor Cyan

# Run visualization
Set-Location ..
python visualize_alpha.py --checkpoint $CHECKPOINT --output_dir $OUTPUT_DIR

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Visualization completed successfully!" -ForegroundColor Green
    Write-Host "`n📂 Output directory: $OUTPUT_DIR" -ForegroundColor Cyan
    Write-Host "`nGenerated plots:" -ForegroundColor Yellow
    Write-Host "  1. alpha_distribution.png  - Histogram of alpha values" -ForegroundColor White
    Write-Host "  2. alpha_sorted.png        - Sorted alpha values by relation" -ForegroundColor White
    Write-Host "  3. alpha_categories.png    - Pie chart (Static/Mixed/Dynamic)" -ForegroundColor White
    Write-Host "  4. alpha_top_bottom.png    - Top 20 static & dynamic relations" -ForegroundColor White
} else {
    Write-Host "`n❌ Visualization failed!" -ForegroundColor Red
    exit 1
}
