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
    --rank 128 `
    --batch_size 1000 `
    --learning_rate 0.01 `
    --max_epochs 250 `
    --valid_freq 25 `
    --emb_reg 0.001 `
    --time_reg 0.001 `
    --smoothness_reg 0.001
