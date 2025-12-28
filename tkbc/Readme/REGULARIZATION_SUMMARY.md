# Regularization Implementation for ContinuousPairRE

## Overview

Added comprehensive regularization support to ContinuousPairRE to prevent overfitting and improve model generalization.

## Changes Made

### 1. Fixed Regularization Factor Returns in ContinuousPairRE

**File**: `models.py`

**Previous**: Returned aggregated means (incorrect for N3 regularizer)

```python
factors = (
    torch.sqrt(h ** 2).mean(),  # Wrong - aggregated
    ...
)
```

**Updated**: Returns individual tensor factors (correct for N3 regularizer)

```python
factors = (
    torch.sqrt(h ** 2 + 1e-10),      # (batch, rank) - head entities
    torch.sqrt(r_h ** 2 + r_t ** 2 + 1e-10),  # (batch, rank) - relations
    torch.sqrt(t ** 2 + 1e-10)       # (batch, rank) - tail entities
)
```

### 2. Added ContinuousTimeLambda3 Regularizer

**File**: `regularizers.py`

Created a new regularizer specifically for continuous time embeddings:

```python
class ContinuousTimeLambda3(Regularizer):
    """
    Regularizer for continuous time embeddings.
    Applies N3 regularization to time embeddings to prevent overfitting.
    """
    def forward(self, factor):
        # Apply L3 norm (cubic norm) to time embeddings
        norm = self.weight * torch.sum(torch.abs(factor) ** 3)
        return norm / factor.shape[0]
```

**Why needed**: The original `Lambda3` regularizer was designed for discrete timestamp embeddings and computes differences between consecutive timestamps. For continuous time, we need direct regularization of the time embedding values.

### 3. Updated Training Pipeline

**File**: `learner.py`

Automatically selects appropriate temporal regularizer:

```python
time_reg = ContinuousTimeLambda3(args.time_reg) if args.model == 'ContinuousPairRE' else Lambda3(args.time_reg)
```

### 4. Updated Training Scripts

**Files**: `train_continuous_pairre.sh`, `train_continuous_pairre.ps1`

Added recommended regularization parameters:

- `--emb_reg 0.001`: Entity/relation embedding regularization (N3)
- `--time_reg 0.001`: Time embedding regularization (ContinuousTimeLambda3)

## Regularization Types

### N3 Regularization (Entity/Relation Embeddings)

- **Formula**: `∑|embedding|³`
- **Purpose**: Prevents entity and relation embeddings from growing too large
- **Applied to**: Head entities, tail entities, relation projections
- **Hyperparameter**: `--emb_reg` (try: 0.0, 0.001, 0.01)

### ContinuousTimeLambda3 (Time Embeddings)

- **Formula**: `∑|time_embedding|³`
- **Purpose**: Prevents time embeddings from overfitting to training timestamps
- **Applied to**: Continuous time embeddings `m = cos(W·τ + b)`
- **Hyperparameter**: `--time_reg` (try: 0.0, 0.001, 0.01)

## How Regularization Works in Training

```python
# Forward pass
predictions, factors, time = model.forward(batch)

# Loss components
l_fit = CrossEntropy(predictions, truth)      # Prediction loss
l_reg = N3(factors)                           # Embedding regularization
l_time = ContinuousTimeLambda3(time)          # Time regularization

# Total loss
l = l_fit + l_reg + l_time

# Backpropagation
l.backward()
```

## Benefits

1. **Prevents Overfitting**: Regularization penalizes large embedding values
2. **Better Generalization**: Model performs better on unseen data
3. **Temporal Smoothness**: Time regularization ensures smooth temporal representations
4. **Controlled Complexity**: Prevents model from memorizing training data

## Hyperparameter Tuning

### Starting Points

- **No regularization**: `--emb_reg 0.0 --time_reg 0.0`
- **Light regularization**: `--emb_reg 0.001 --time_reg 0.001` (recommended)
- **Strong regularization**: `--emb_reg 0.01 --time_reg 0.01`

### When to Increase Regularization

- Training MRR >> Validation MRR (overfitting)
- Model embeddings are very large
- Loss decreases but validation performance doesn't improve

### When to Decrease Regularization

- Model underfits (low training and validation MRR)
- Regularization term dominates the loss
- Model converges too slowly

## Comparison with Baselines

| Model                | Embedding Reg | Time Reg                                    |
| -------------------- | ------------- | ------------------------------------------- |
| TComplEx             | N3            | Lambda3 (discrete timestamps)               |
| TNTComplEx           | N3            | Lambda3 (discrete timestamps)               |
| **ContinuousPairRE** | **N3**        | **ContinuousTimeLambda3** (continuous time) |

## Example Training Commands

```bash
# Light regularization (recommended starting point)
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100 \
  --batch_size 1000 --learning_rate 0.1 --max_epochs 50 --valid_freq 5 \
  --emb_reg 0.001 --time_reg 0.001

# No regularization (baseline)
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100 \
  --batch_size 1000 --learning_rate 0.1 --max_epochs 50 --valid_freq 5 \
  --emb_reg 0.0 --time_reg 0.0

# Strong regularization (if overfitting)
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100 \
  --batch_size 1000 --learning_rate 0.1 --max_epochs 50 --valid_freq 5 \
  --emb_reg 0.01 --time_reg 0.01
```

## Verification

All files validated with no errors:

- ✅ `models.py` - Correct factor returns
- ✅ `regularizers.py` - ContinuousTimeLambda3 added
- ✅ `learner.py` - Automatic regularizer selection
- ✅ `optimizers.py` - Proper regularization application

## Date

December 28, 2025
