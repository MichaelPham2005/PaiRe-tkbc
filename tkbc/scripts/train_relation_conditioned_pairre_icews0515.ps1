# Training script for Relation-Conditioned Continuous Time PairRE on ICEWS05-15
# 
# ICEWS05-15: Larger dataset (10 years of data)
# - More training data → can use higher rank
# - Longer time span → may need more frequencies
#
# Adjusted hyperparameters:
#   --rank: 256 (vs 128 for ICEWS14)
#   --K_frequencies: 32 (vs 16 for ICEWS14)
#   --max_epochs: 150 (vs 200 for ICEWS14, dataset is larger)
#   --batch_size: 1500 (if memory allows)

python learner.py `
    --dataset ICEWS05-15 `
    --model ContinuousPairRE `
    --rank 256 `
    --K_frequencies 32 `
    --beta 0.5 `
    --time_scale 1.0 `
    --batch_size 1500 `
    --learning_rate 0.001 `
    --max_epochs 150 `
    --valid_freq 10 `
    --emb_reg 0.001 `
    --time_reg 0.001 `
    --time_param_reg 0.0001 `
    --time_discrimination_weight 0.1 `
    --loss cross_entropy
