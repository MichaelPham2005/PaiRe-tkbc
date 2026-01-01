# Train ContinuousPairRE with Margin-Based Loss (Paper Version)
# This script uses the margin-based loss function from the paper:
# L = -log(σ(γ - f_r(h,t))) - Σ p(h',r,t') log(σ(f_r(h',t') - γ))

Write-Host "Training ContinuousPairRE with Margin-Based Loss" -ForegroundColor Green
Write-Host "Dataset: ICEWS14" -ForegroundColor Yellow
Write-Host ""

python learner.py `
    --dataset ICEWS14 `
    --model ContinuousPairRE `
    --rank 100 `
    --batch_size 1000 `
    --learning_rate 0.1 `
    --max_epochs 50 `
    --valid_freq 5 `
    --emb_reg 0.0 `
    --time_reg 0.01 `
    --smoothness_reg 0.001 `
    --use_margin_loss `
    --gamma 9.0 `
    --num_neg 64 `
    --adversarial_temp 1.0

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
