# Training script for Relation-Conditioned Continuous Time PairRE on ICEWS14
# 
# Model: ContinuousPairRE with RelationConditionedTimeEncoder + Residual Gate
# 
# Key hyperparameters:
#   --rank: Embedding dimension (default: 128)
#   --K_frequencies: Number of frequency components (default: 16)
#   --beta: Residual gate strength (default: 0.5, range [0.1, 1.0])
#   --time_scale: Time normalization scale (default: 1.0 for [0,1])
#
# Regularization:
#   --emb_reg: N3 regularization for entity/relation embeddings
#   --time_reg: Regularization for time gate deviation from identity
#   --time_param_reg: Light L2 for time parameters (a_r, A_r)
#   --time_discrimination_weight: Weight for time discrimination loss
#
# Loss function:
#   --loss: 'cross_entropy' (stable, fast) or 'self_adversarial' (PairRE-style)
#
# Note: Model uses init_size=0.1 by default for proper temporal sensitivity

python learner.py `
    --dataset ICEWS14 `
    --model ContinuousPairRE `
    --rank 128 `
    --K_frequencies 16 `
    --beta 0.5 `
    --time_scale 1.0 `
    --batch_size 1000 `
    --learning_rate 0.001 `
    --max_epochs 200 `
    --valid_freq 10 `
    --emb_reg 0.001 `
    --time_reg 0.001 `
    --time_param_reg 0.0001 `
    --time_discrimination_weight 0.1 `
    --loss cross_entropy
