# Training script for ContinuousPairRE on ICEWS14 (PowerShell)
# Regularization helps prevent overfitting:
#   --emb_reg: N3 regularization for entity/relation embeddings (try 0.0, 0.001, 0.01)
#   --time_reg: ContinuousTimeLambda3 regularization for time embeddings (try 0.0, 0.001, 0.01)

# Move to parent directory where learner.py is located
Set-Location ..

# Disable debugger warnings
$env:PYDEVD_DISABLE_FILE_VALIDATION=1

python -Xfrozen_modules=off learner.py `
    --dataset ICEWS14 `
    --model ContinuousPairRE `
    --rank 156 `
    --batch_size 1000 `
    --learning_rate 0.1 `
    --max_epochs 50 `
    --valid_freq 5 `
    --emb_reg 0.01 `
    --time_reg 0.01 `
    --smoothness_reg 0.001 `
    --alpha_reg 0.01
