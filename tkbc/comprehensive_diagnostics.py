"""
Comprehensive diagnostics for Relation-Conditioned Continuous Time Model.
Tests if time is actually being learned and used by the model.

Usage:
    python comprehensive_diagnostics.py --dataset ICEWS14 --checkpoint models/ICEWS14/checkpoint --n_samples 500
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from datasets import TemporalDataset
from models import ContinuousPairRE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Load dataset and model
print(f"Loading dataset {args.dataset}...")
dataset = TemporalDataset(args.dataset, use_continuous_time=True)

print(f"Loading checkpoint from {args.checkpoint}...")
checkpoint = torch.load(args.checkpoint, map_location=args.device)
model_state = checkpoint if isinstance(checkpoint, dict) and 'model' not in checkpoint else checkpoint.get('model', checkpoint)

# Recreate model
sizes = dataset.get_shape()
model = ContinuousPairRE(sizes, rank=128, K=16, beta=0.5)  # Adjust params as needed
model.load_state_dict(model_state, strict=False)
model = model.to(args.device)
model.eval()

print("\n" + "="*80)
print("COMPREHENSIVE TIME LEARNING DIAGNOSTICS")
print("="*80)

# Get validation data
valid_data = dataset.get_examples('valid')
n_samples = min(args.n_samples, len(valid_data))
sample_indices = np.random.choice(len(valid_data), n_samples, replace=False)
samples = torch.from_numpy(valid_data[sample_indices].astype('int64')).to(args.device)

# Convert to continuous time
samples_continuous = samples.clone().float()
timestamp_ids = samples[:, 3].cpu().numpy()
continuous_times = torch.tensor(
    [dataset.ts_normalized[int(tid)] for tid in timestamp_ids],
    dtype=torch.float32
).to(args.device)
samples_continuous[:, 3] = continuous_times

print(f"\nUsing {n_samples} validation samples")
print(f"Device: {args.device}")

# ============================================================================
# TEST A1: Time Discrimination Accuracy
# ============================================================================
print("\n" + "="*80)
print("TEST A1: Time Discrimination Accuracy")
print("="*80)

with torch.no_grad():
    # Sample negative times
    n_times = len(dataset.ts_normalized)
    neg_time_ids = torch.randint(0, n_times, (n_samples,))
    neg_continuous_times = torch.tensor(
        [dataset.ts_normalized[int(tid)] for tid in neg_time_ids.cpu().numpy()],
        dtype=torch.float32
    ).to(args.device)
    
    samples_neg_time = samples_continuous.clone()
    samples_neg_time[:, 3] = neg_continuous_times
    
    # Compute scores
    scores_pos = model.score(samples_continuous)
    scores_neg = model.score(samples_neg_time)
    
    # Accuracy: fraction where positive scores higher
    correct = (scores_pos > scores_neg).float()
    time_acc = correct.mean().item()
    
    # Margin statistics
    margin = scores_pos - scores_neg
    mean_margin = margin.mean().item()
    std_margin = margin.std().item()
    
    print(f"\nTime Discrimination Accuracy: {time_acc*100:.2f}%")
    print(f"Mean Margin (φ_pos - φ_neg): {mean_margin:.6f}")
    print(f"Std Margin: {std_margin:.6f}")
    
    print("\nInterpretation:")
    if time_acc < 0.52:
        print("  ❌ FAIL: Time not learned (random ~50%)")
    elif time_acc < 0.60:
        print("  ⚠️  WEAK: Time starting to learn (55-60%)")
    elif time_acc < 0.65:
        print("  ✅ GOOD: Time learning clearly (60-65%)")
    else:
        print("  ✅ EXCELLENT: Strong time learning (>65%)")
    
    if abs(mean_margin) < 0.001:
        print("  ❌ FAIL: Margin near zero - scores don't vary with time")
    elif mean_margin < 0:
        print("  ❌ FAIL: Negative margin - model prefers WRONG times!")
    elif mean_margin < 0.01:
        print("  ⚠️  WEAK: Small margin - time signal weak")
    else:
        print("  ✅ GOOD: Positive margin - time signal present")

# ============================================================================
# TEST A2: Temporal Sensitivity
# ============================================================================
print("\n" + "="*80)
print("TEST A2: Temporal Sensitivity (Score Changes with Time)")
print("="*80)

with torch.no_grad():
    # For each sample, try 3 different times
    n_test = min(500, n_samples)
    test_samples = samples_continuous[:n_test]
    
    # Time 1: Original (true time)
    scores_t1 = model.score(test_samples)
    
    # Time 2: Random alternative time
    alt_time_ids = torch.randint(0, n_times, (n_test,))
    alt_continuous_times = torch.tensor(
        [dataset.ts_normalized[int(tid)] for tid in alt_time_ids.cpu().numpy()],
        dtype=torch.float32
    ).to(args.device)
    test_alt = test_samples.clone()
    test_alt[:, 3] = alt_continuous_times
    scores_t2 = model.score(test_alt)
    
    # Time 3: Adjacent time (tau ± 0.1)
    adj_times = torch.clamp(test_samples[:, 3] + 0.1, 0.0, 1.0)
    test_adj = test_samples.clone()
    test_adj[:, 3] = adj_times
    scores_t3 = model.score(test_adj)
    
    # Calculate deltas
    delta_random = torch.abs(scores_t1 - scores_t2)
    delta_adjacent = torch.abs(scores_t1 - scores_t3)
    
    mean_delta_random = delta_random.mean().item()
    std_delta_random = delta_random.std().item()
    mean_delta_adj = delta_adjacent.mean().item()
    
    # Insensitivity: fraction with delta < threshold
    eps = 0.01
    frac_insensitive_random = (delta_random < eps).float().mean().item()
    frac_insensitive_adj = (delta_adjacent < eps).float().mean().item()
    
    print(f"\nScore Delta (True vs Random Time):")
    print(f"  Mean: {mean_delta_random:.6f}")
    print(f"  Std:  {std_delta_random:.6f}")
    print(f"  Insensitive fraction (<{eps}): {frac_insensitive_random*100:.1f}%")
    
    print(f"\nScore Delta (True vs Adjacent Time):")
    print(f"  Mean: {mean_delta_adj:.6f}")
    print(f"  Insensitive fraction (<{eps}): {frac_insensitive_adj*100:.1f}%")
    
    print("\nInterpretation:")
    if mean_delta_random < 0.001:
        print("  ❌ FAIL: Scores barely change with time (delta < 0.001)")
    elif mean_delta_random < 0.01:
        print("  ⚠️  WEAK: Small temporal sensitivity (delta 0.001-0.01)")
    else:
        print("  ✅ GOOD: Significant temporal sensitivity (delta > 0.01)")
    
    if frac_insensitive_random > 0.8:
        print("  ❌ FAIL: 80%+ samples insensitive to time changes")
    elif frac_insensitive_random > 0.5:
        print("  ⚠️  WEAK: 50%+ samples insensitive to time")
    else:
        print("  ✅ GOOD: Most samples sensitive to time changes")

# ============================================================================
# TEST A3: Ablation - Full vs Time-Off
# ============================================================================
print("\n" + "="*80)
print("TEST A3: Ablation Study (Full vs Time-Off)")
print("="*80)

def evaluate_mrr(model, samples, batch_size=100, time_off=False):
    """Evaluate MRR with optional time-off mode."""
    ranks = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_size_actual = batch.shape[0]
        
        # Get predictions for all entities
        h_idx = batch[:, 0].long()
        r_idx = batch[:, 1].long()
        t_idx = batch[:, 2].long()
        
        # Time-off mode: set time embedding to zero
        if time_off:
            # Create a modified model that returns identity gate
            original_forward = model.time_encoder.forward
            model.time_encoder.forward = lambda rel_id, tau: torch.zeros(
                rel_id.shape[0], model.rank, device=rel_id.device
            )
        
        try:
            # Score all possible tails
            predictions = model.forward(batch)[0]  # (batch, n_entities)
            
            # Get rank of true tail
            for j in range(batch_size_actual):
                true_tail = t_idx[j].item()
                scores = predictions[j]
                # Rank: how many entities score higher than true tail
                rank = (scores > scores[true_tail]).sum().item() + 1
                ranks.append(1.0 / rank)
        
        finally:
            if time_off:
                model.time_encoder.forward = original_forward
    
    return np.mean(ranks)

with torch.no_grad():
    n_eval = min(1000, n_samples)  # Limit to avoid OOM
    eval_samples = samples_continuous[:n_eval]
    
    print(f"\nEvaluating on {n_eval} samples...")
    print("Computing MRR with full model (time enabled)...")
    mrr_full = evaluate_mrr(model, eval_samples, batch_size=50, time_off=False)
    
    print("Computing MRR with time-off (gate = 1)...")
    mrr_timeoff = evaluate_mrr(model, eval_samples, batch_size=50, time_off=True)
    
    delta_mrr = mrr_full - mrr_timeoff
    relative_change = (delta_mrr / mrr_timeoff * 100) if mrr_timeoff > 0 else 0
    
    print(f"\nResults:")
    print(f"  MRR (Full):     {mrr_full:.4f}")
    print(f"  MRR (Time-off): {mrr_timeoff:.4f}")
    print(f"  Δ MRR:          {delta_mrr:+.4f} ({relative_change:+.1f}%)")
    
    print("\nInterpretation:")
    if abs(delta_mrr) < 0.001:
        print("  ❌ FAIL: Full ≈ Time-off → Time not being used!")
    elif delta_mrr < -0.01:
        print("  ❌ FAIL: Time-off > Full → Time hurts performance!")
    elif delta_mrr < 0.01:
        print("  ⚠️  WEAK: Small improvement from time")
    else:
        print("  ✅ GOOD: Time clearly helps performance")

# ============================================================================
# TEST A4: Time Embedding Variation
# ============================================================================
print("\n" + "="*80)
print("TEST A4: Time Embedding Variation Over Time")
print("="*80)

with torch.no_grad():
    # Pick top 5 most frequent relations
    relation_counts = {}
    for quad in valid_data:
        r = int(quad[1])
        relation_counts[r] = relation_counts.get(r, 0) + 1
    top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nAnalyzing top {len(top_relations)} relations:")
    for r, count in top_relations:
        print(f"  Relation {r}: {count} occurrences")
    
    # For each relation, sample time embeddings across time range
    n_time_points = 20
    time_points = torch.linspace(0, 1, n_time_points).to(args.device)
    
    total_var = []
    total_cos_adj = []
    
    for r, _ in top_relations:
        rel_ids = torch.full((n_time_points,), r, dtype=torch.long).to(args.device)
        
        # Get time embeddings
        m_t = model.time_encoder(rel_ids, time_points)  # (n_time_points, rank)
        
        # Variance over time
        var_over_time = m_t.var(dim=0).mean().item()
        
        # Cosine similarity between adjacent time points
        cos_similarities = []
        for i in range(n_time_points - 1):
            cos_sim = torch.nn.functional.cosine_similarity(
                m_t[i:i+1], m_t[i+1:i+2], dim=1
            ).item()
            cos_similarities.append(cos_sim)
        avg_cos_adj = np.mean(cos_similarities)
        
        total_var.append(var_over_time)
        total_cos_adj.append(avg_cos_adj)
        
        print(f"\n  Relation {r}:")
        print(f"    Variance over time: {var_over_time:.6f}")
        print(f"    Avg cosine (adjacent): {avg_cos_adj:.4f}")
    
    mean_var = np.mean(total_var)
    mean_cos_adj = np.mean(total_cos_adj)
    
    print(f"\nAggregate Statistics:")
    print(f"  Mean variance over time: {mean_var:.6f}")
    print(f"  Mean cosine (adjacent): {mean_cos_adj:.4f}")
    
    print("\nInterpretation:")
    if mean_var < 1e-6:
        print("  ❌ FAIL: Variance ≈ 0 → Time embeddings collapsed!")
    elif mean_var < 1e-4:
        print("  ⚠️  WEAK: Very low variance → Minimal temporal variation")
    else:
        print("  ✅ GOOD: Significant variance → Time embeddings vary")
    
    if mean_cos_adj > 0.99:
        print("  ❌ FAIL: Cosine ≈ 1 → Adjacent times nearly identical!")
    elif mean_cos_adj > 0.95:
        print("  ⚠️  WEAK: High cosine → Slow temporal variation")
    else:
        print("  ✅ GOOD: Lower cosine → Clear temporal changes")

# ============================================================================
# SUMMARY AND DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("OVERALL DIAGNOSIS")
print("="*80)

# Collect all signals
signals = {
    'time_acc': time_acc,
    'mean_margin': mean_margin,
    'mean_delta': mean_delta_random,
    'frac_insensitive': frac_insensitive_random,
    'mrr_full': mrr_full,
    'mrr_timeoff': mrr_timeoff,
    'delta_mrr': delta_mrr,
    'mean_var': mean_var,
    'mean_cos_adj': mean_cos_adj
}

print("\nKey Metrics:")
for key, val in signals.items():
    print(f"  {key}: {val:.6f}")

# Diagnosis logic
print("\n" + "-"*80)
print("DIAGNOSIS:")
print("-"*80)

if time_acc < 0.52 and abs(mean_margin) < 0.001 and abs(delta_mrr) < 0.001:
    print("\n⚠️  CASE 1: Static Learning, Time Not Learning")
    print("  - Time discrimination accuracy ~50% (random)")
    print("  - Margin ~0 (scores don't vary with time)")
    print("  - Full ≈ Time-off (time not used)")
    print("\n  Possible causes:")
    print("  1. Time discrimination loss weight too low")
    print("  2. Negative time sampling issue")
    print("  3. Time path has vanishing gradients")
    print("  4. Regularization too strong on time parameters")

elif mean_var > 1e-4 and mean_delta_random < 0.01 and abs(delta_mrr) < 0.001:
    print("\n⚠️  CASE 2: Time Embeddings Vary But Scores Don't")
    print("  - Time embeddings have variation (var > 0)")
    print("  - But score sensitivity low (delta < 0.01)")
    print("  - Full ≈ Time-off (time not affecting predictions)")
    print("\n  Possible causes:")
    print("  1. Gate too weak (beta too small)")
    print("  2. L1 norm canceling time effect")
    print("  3. Time encoder output not connected to scoring")
    print("  4. Entity/relation embeddings dominating")

elif time_acc > 0.55 and mean_margin > 0.01 and delta_mrr > 0.01:
    print("\n✅ CASE 3: Time Learning Correctly!")
    print("  - Time discrimination accuracy >55%")
    print("  - Positive margin increasing")
    print("  - Temporal sensitivity present")
    print("  - Full > Time-off (time helps)")
    print("\n  Model is learning temporal patterns successfully!")

else:
    print("\n⚠️  MIXED SIGNALS: Partial Learning")
    print("  Some indicators positive, others negative.")
    print("  Check individual test results above for details.")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
