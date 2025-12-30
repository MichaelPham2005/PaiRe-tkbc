# Check continuity of learned time embeddings
# Run this after training to verify m = cos(W·t + b) is smooth

$CHECKPOINT_PATH = "..\models\ICEWS14\ContinuousPairRE\best_valid.pt"

python check_continuity.py `
    --checkpoint $CHECKPOINT_PATH `
    --dataset ICEWS14 `
    --rank 156
