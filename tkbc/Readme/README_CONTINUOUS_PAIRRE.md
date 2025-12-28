# Continuous-Time PairRE Implementation

This implementation extends TKBC with a continuous-time PairRE model using relation-wise temporal gating, as specified in `TIME_MAPPING_INSTRUCTIONS.md`.

## Quick Start

### 1. Preprocess Timestamps

```bash
python preprocess_continuous_time.py
```

This normalizes timestamps to [0, 100] for all datasets (ICEWS14, ICEWS05-15, YAGO15k, wikidata).

### 2. Train Model

```bash
# Linux/Mac
./train_continuous_pairre.sh

# Windows
.\train_continuous_pairre.ps1

# Or manually
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100 --batch_size 1000 --learning_rate 0.1 --max_epochs 50 --valid_freq 5
```

## Model Architecture

**Scoring Function:**

```
score = -||(h ∘ r^H - t ∘ r^T) ∘ (α·cos(W·τ+b) + (1-α))||₁
```

Where:

- `h, t`: entity embeddings
- `r^H, r^T`: relation head/tail projections (PairRE)
- `τ`: normalized continuous time [0,100]
- `α`: relation-wise temporal gating [0,1]
- `W, b`: learnable time parameters

**Key Features:**

- Continuous time modeling (smooth interpolation/extrapolation)
- Relation-wise gating (learns static vs dynamic relations)
- Efficient parameterization (~45% fewer parameters than TComplEx)

## Files

**Core Implementation:**

- `models.py` - ContinuousTimeEmbedding & ContinuousPairRE classes
- `datasets.py` - Continuous time loading support
- `learner.py` - Training pipeline
- `optimizers.py` - ContinuousTimeOptimizer

**Scripts:**

- `preprocess_continuous_time.py` - Time normalization
- `train_continuous_pairre.sh/.ps1` - Training scripts
- `test_continuous_pairre.py` - Testing script

## Hyperparameters

```
--dataset: ICEWS14, ICEWS05-15, yago15k
--model: ContinuousPairRE
--rank: 100 (embedding dimension)
--batch_size: 1000
--learning_rate: 0.1
--max_epochs: 50
--valid_freq: 5
--emb_reg: 0.0
--time_reg: 0.0
```

## Evaluation Metrics

- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10

## Implementation Date

December 28, 2025
