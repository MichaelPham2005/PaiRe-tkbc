# Quick test script to check time discrimination loss diagnostics
# Run a few training batches and inspect the detailed logs

python learner.py `
    --dataset ICEWS14 `
    --model ContinuousPairRE `
    --rank 64 `
    --K_frequencies 8 `
    --beta 0.5 `
    --time_scale 1.0 `
    --batch_size 500 `
    --learning_rate 0.001 `
    --max_epochs 1 `
    --valid_freq 1 `
    --emb_reg 0.001 `
    --time_reg 0.001 `
    --time_param_reg 0.0001 `
    --time_discrimination_weight 0.1 `
    --loss cross_entropy

Write-Host "`n================================"
Write-Host "DIAGNOSTIC SUMMARY"
Write-Host "================================"
Write-Host "Check the output above for:"
Write-Host "  [Batch 0] TIME DISCRIMINATION DIAGNOSTICS"
Write-Host "  [Batch 50] TIME DISCRIMINATION DIAGNOSTICS"
Write-Host ""
Write-Host "Expected behavior:"
Write-Host "  1. raw_t_disc should be > 0 (not constant)"
Write-Host "  2. weighted_t_disc = raw_t_disc × 0.1"
Write-Host "  3. requires_grad = True, has_grad_fn = True"
Write-Host "  4. ratio (t_disc/total) should be 0.01-0.1"
Write-Host "  5. mean(φ_pos - φ_neg) should be > 0 (positive scores higher)"
Write-Host "  6. std(φ_pos - φ_neg) should be > 0.001 (variation exists)"
Write-Host "  7. tau collision rate should be ~0 (< 0.05)"
Write-Host ""
Write-Host "If any of these fail, investigate further!"
