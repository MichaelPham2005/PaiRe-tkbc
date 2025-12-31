"""
Diagnostic script for temporal knowledge graph embedding models.
Tests 4 hypotheses about time embedding collapse and temporal learning.

Usage:
    python diagnose_time_module.py \
        --data_dir external/tkbc/data/ICEWS14 \
        --ckpt models/ICEWS14/ContinuousPairRE/best_valid.pt \
        --model_type fourier_linear \
        --device cuda \
        --out_dir diagnostics/run1
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle

# Add path for model imports
sys.path.append('external/tkbc')
from models import ContinuousPairRE, TComplEx, TNTComplEx


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose time module issues')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--ckpt', type=str, default='/kaggle/working/models/ICEWS14/ContinuousPairRE/best_valid.pt', 
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['tntcomplex', 'fourier_linear', 'time2vec'], 
                        default='fourier_linear', help='Type of model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_time_samples', type=int, default=512, help='Number of time samples for Test A')
    parser.add_argument('--num_triples', type=int, default=1000, help='Number of triples for Test B')
    parser.add_argument('--num_time_alts', type=int, default=10, help='Number of alternative times for Test B')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--do_grad_diagnostics', action='store_true', help='Run gradient diagnostics (Test D)')
    parser.add_argument('--out_dir', type=str, default='diagnostics', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir):
    """Load train/valid/test files"""
    print(f"Loading dataset from {data_dir}...")
    
    data = {}
    for split in ['train', 'valid', 'test']:
        # Try pickle first
        pickle_path = Path(data_dir) / f'{split}.pickle'
        if pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                triples = pickle.load(f)
            data[split] = np.array(triples)
            print(f"  {split}: {len(triples)} triples (from pickle)")
            continue
        
        # Try text files
        file_path = Path(data_dir) / split
        if not file_path.exists():
            file_path = Path(data_dir) / f'{split}.txt'
        
        if file_path.exists():
            triples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        # Parse as indices directly
                        h, r, t, time = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        triples.append([h, r, t, time])
            
            data[split] = np.array(triples)
            print(f"  {split}: {len(triples)} triples (from txt)")
        else:
            raise FileNotFoundError(f"Cannot find {split}.pickle or {split}.txt in {data_dir}")
    
    # Load time normalization if exists
    ts_norm_path = Path(data_dir) / 'ts_normalized.pickle'
    ts_normalized = None
    if ts_norm_path.exists():
        with open(ts_norm_path, 'rb') as f:
            ts_normalized = pickle.load(f)
        print(f"  Loaded {len(ts_normalized)} normalized timestamps")
    
    return data, ts_normalized


def load_model(ckpt_path, model_type, device):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt_path}...")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}
    
    # Infer model architecture from state dict
    if model_type == 'fourier_linear':
        # Assume ContinuousPairRE
        # Infer sizes from embeddings
        n_entities = state_dict['entity_embeddings.weight'].shape[0]
        n_relations = state_dict['relation_head.weight'].shape[0]
        rank = state_dict['entity_embeddings.weight'].shape[1]
        
        # Get n_timestamps from config or estimate
        n_timestamps = config.get('n_timestamps', 365)  # default for ICEWS14
        
        sizes = (n_entities, n_relations, n_entities, n_timestamps)
        model = ContinuousPairRE(sizes, rank)
    elif model_type == 'tntcomplex':
        n_entities = state_dict['embeddings.0.weight'].shape[0]
        n_relations = state_dict['embeddings.1.weight'].shape[0] 
        n_timestamps = state_dict['embeddings.2.weight'].shape[0]
        rank = state_dict['embeddings.0.weight'].shape[1]
        
        sizes = (n_entities, n_relations, n_entities, n_timestamps)
        model = TNTComplEx(sizes, rank)
    else:
        raise ValueError(f"Model type {model_type} not implemented yet")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded: {n_entities} entities, {n_relations} relations, {rank} rank")
    return model, config


def test_a_time_embedding_variation(model, ts_normalized, num_samples, device, out_dir):
    """
    Test A: Time embedding variation (H1 - collapse detection)
    """
    print("\n" + "="*70)
    print("TEST A: TIME EMBEDDING VARIATION (H1 - Collapse Detection)")
    print("="*70)
    
    if not hasattr(model, 'time_encoder'):
        print("Model has no time_encoder attribute. Skipping Test A.")
        return {}
    
    time_encoder = model.time_encoder
    
    # Get all time IDs
    if ts_normalized:
        time_ids = sorted(ts_normalized.keys())
        n_times = len(time_ids)
    else:
        n_times = 365  # default
        time_ids = list(range(n_times))
    
    # Sample if too many
    if n_times > num_samples:
        time_ids = np.random.choice(time_ids, num_samples, replace=False)
        time_ids = sorted(time_ids)
    
    print(f"Computing embeddings for {len(time_ids)} timestamps...")
    
    # Convert time IDs to normalized values
    if ts_normalized:
        normalized_times = torch.tensor([ts_normalized[tid] for tid in time_ids], 
                                       dtype=torch.float32, device=device)
    else:
        # Fallback: normalize to [0, 100]
        normalized_times = torch.tensor(time_ids, dtype=torch.float32, device=device)
        normalized_times = 100.0 * normalized_times / (n_times - 1)
    
    # Compute time embeddings
    with torch.no_grad():
        time_embeds = time_encoder(normalized_times)  # (n_times, d)
    
    time_embeds_np = time_embeds.cpu().numpy()
    
    # Compute statistics
    norms = np.linalg.norm(time_embeds_np, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Pairwise cosine similarity (adjacent)
    cosines = []
    l2_dists = []
    for i in range(len(time_embeds_np) - 1):
        v1 = time_embeds_np[i]
        v2 = time_embeds_np[i + 1]
        
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cosines.append(cos)
        
        l2_dist = np.linalg.norm(v2 - v1)
        l2_dists.append(l2_dist)
    
    avg_cosine_adjacent = np.mean(cosines)
    avg_l2_adjacent = np.mean(l2_dists)
    
    # Global variance per dimension
    global_variance = np.var(time_embeds_np, axis=0).mean()
    
    # Effective rank via SVD
    U, S, Vt = np.linalg.svd(time_embeds_np, full_matrices=False)
    S_norm = S / S.sum()
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-9))
    effective_rank = np.exp(entropy)
    
    # Top 5 singular values
    top5_singular_values = S[:5].tolist()
    
    stats = {
        'mean_norm': float(mean_norm),
        'std_norm': float(std_norm),
        'avg_cosine_adjacent': float(avg_cosine_adjacent),
        'avg_l2_adjacent': float(avg_l2_adjacent),
        'global_variance': float(global_variance),
        'effective_rank': float(effective_rank),
        'top5_singular_values': top5_singular_values,
        'diagnosis': ''
    }
    
    # Diagnosis
    if global_variance < 0.01:
        stats['diagnosis'] = 'COLLAPSE: Very low variance across time'
    elif avg_cosine_adjacent > 0.99:
        stats['diagnosis'] = 'COLLAPSE: Adjacent embeddings too similar'
    elif effective_rank < 5:
        stats['diagnosis'] = 'COLLAPSE: Very low effective rank'
    else:
        stats['diagnosis'] = 'OK: Time embeddings show variation'
    
    print(f"\nResults:")
    print(f"  Mean norm: {mean_norm:.4f}")
    print(f"  Std norm: {std_norm:.4f}")
    print(f"  Avg cosine (adjacent): {avg_cosine_adjacent:.4f}")
    print(f"  Avg L2 dist (adjacent): {avg_l2_adjacent:.4f}")
    print(f"  Global variance: {global_variance:.6f}")
    print(f"  Effective rank: {effective_rank:.2f}")
    print(f"  Top 5 singular values: {top5_singular_values}")
    print(f"  Diagnosis: {stats['diagnosis']}")
    
    # Save detailed CSV
    df = pd.DataFrame({
        'time_id': time_ids,
        'norm': norms,
        'first_dim': time_embeds_np[:, 0],
        'second_dim': time_embeds_np[:, 1],
        'third_dim': time_embeds_np[:, 2],
    })
    csv_path = Path(out_dir) / 'time_embedding_stats.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed stats to {csv_path}")
    
    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Norms over time
        axes[0, 0].plot(time_ids, norms)
        axes[0, 0].set_title('Time Embedding Norms')
        axes[0, 0].set_xlabel('Time ID')
        axes[0, 0].set_ylabel('L2 Norm')
        
        # Plot 2: First 3 dimensions
        axes[0, 1].plot(time_ids, time_embeds_np[:, 0], label='Dim 0')
        axes[0, 1].plot(time_ids, time_embeds_np[:, 1], label='Dim 1')
        axes[0, 1].plot(time_ids, time_embeds_np[:, 2], label='Dim 2')
        axes[0, 1].set_title('First 3 Dimensions')
        axes[0, 1].set_xlabel('Time ID')
        axes[0, 1].legend()
        
        # Plot 3: Singular values
        axes[1, 0].bar(range(len(S[:20])), S[:20])
        axes[1, 0].set_title('Top 20 Singular Values')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Value')
        
        # Plot 4: Adjacent cosine similarity
        axes[1, 1].plot(range(len(cosines)), cosines)
        axes[1, 1].axhline(y=0.99, color='r', linestyle='--', label='Collapse threshold')
        axes[1, 1].set_title('Adjacent Cosine Similarity')
        axes[1, 1].set_xlabel('Transition')
        axes[1, 1].set_ylabel('Cosine')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = Path(out_dir) / 'plots' / 'test_a_time_variation.png'
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return stats


def test_b_temporal_sensitivity(model, data, ts_normalized, num_triples, num_time_alts, device, out_dir):
    """
    Test B: Score sensitivity to time (H2 - temporal signal)
    """
    print("\n" + "="*70)
    print("TEST B: TEMPORAL SENSITIVITY (H2 - Time Signal Influence)")
    print("="*70)
    
    test_data = data['test']
    
    # Sample triples
    if len(test_data) > num_triples:
        indices = np.random.choice(len(test_data), num_triples, replace=False)
        sampled_triples = test_data[indices]
    else:
        sampled_triples = test_data
    
    print(f"Testing {len(sampled_triples)} triples...")
    
    results = []
    
    for h, r, t, time_true in sampled_triples:
        # Generate alternative times
        all_times = list(range(365))  # Assume 365 timestamps
        time_true_int = int(time_true)
        
        # Include: true, adjacent, random
        time_alts = [time_true_int]
        if time_true_int > 0:
            time_alts.append(time_true_int - 1)
        if time_true_int < 364:
            time_alts.append(time_true_int + 1)
        
        # Add random times
        while len(time_alts) < num_time_alts:
            t_rand = np.random.choice(all_times)
            if t_rand not in time_alts:
                time_alts.append(t_rand)
        
        # Convert to normalized continuous time
        if ts_normalized:
            time_norm_alts = [ts_normalized[t] for t in time_alts]
        else:
            time_norm_alts = [100.0 * t / 364 for t in time_alts]
        
        # Compute scores
        scores = []
        with torch.no_grad():
            for t_norm in time_norm_alts:
                # Create input: [h, r, t, time]
                x = torch.tensor([[h, r, t, t_norm]], dtype=torch.float32, device=device)
                score = model.score(x).item()
                scores.append(score)
        
        score_true = scores[0]
        score_alts = scores[1:]
        
        # Compute delta
        abs_deltas = [abs(score_true - s) for s in score_alts]
        mean_abs_delta = np.mean(abs_deltas)
        
        results.append({
            'h': h,
            'r': r,
            't': t,
            'time_true': time_true_int,
            'score_true': score_true,
            'mean_abs_delta': mean_abs_delta,
            'max_abs_delta': max(abs_deltas),
            'min_score_alt': min(score_alts),
            'max_score_alt': max(score_alts)
        })
    
    # Aggregate statistics
    df = pd.DataFrame(results)
    
    mean_abs_delta = df['mean_abs_delta'].mean()
    std_abs_delta = df['mean_abs_delta'].std()
    
    # What fraction has delta < 0.01 (very small)?
    frac_insensitive = (df['mean_abs_delta'] < 0.01).mean()
    
    stats = {
        'num_triples_tested': len(results),
        'mean_abs_score_delta': float(mean_abs_delta),
        'std_abs_score_delta': float(std_abs_delta),
        'fraction_insensitive': float(frac_insensitive),
        'diagnosis': ''
    }
    
    if mean_abs_delta < 0.1:
        stats['diagnosis'] = 'LOW SENSITIVITY: Scores barely change with time'
    elif frac_insensitive > 0.5:
        stats['diagnosis'] = 'MODERATE ISSUE: Many triples insensitive to time'
    else:
        stats['diagnosis'] = 'OK: Scores show temporal sensitivity'
    
    print(f"\nResults:")
    print(f"  Mean abs score delta: {mean_abs_delta:.6f}")
    print(f"  Std abs score delta: {std_abs_delta:.6f}")
    print(f"  Fraction insensitive (<0.01): {frac_insensitive:.2%}")
    print(f"  Diagnosis: {stats['diagnosis']}")
    
    # Save CSV
    csv_path = Path(out_dir) / 'temporal_sensitivity.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to {csv_path}")
    
    # Plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of mean_abs_delta
        axes[0].hist(df['mean_abs_delta'], bins=50, edgecolor='black')
        axes[0].axvline(0.01, color='r', linestyle='--', label='Insensitive threshold')
        axes[0].set_title('Distribution of Score Deltas')
        axes[0].set_xlabel('Mean Abs Score Delta')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        
        # Scatter: score_true vs mean_abs_delta
        axes[1].scatter(df['score_true'], df['mean_abs_delta'], alpha=0.5)
        axes[1].set_title('Score Delta vs True Score')
        axes[1].set_xlabel('Score (true time)')
        axes[1].set_ylabel('Mean Abs Delta')
        
        plt.tight_layout()
        plot_path = Path(out_dir) / 'plots' / 'test_b_temporal_sensitivity.png'
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return stats


def test_c_static_shortcut_ablation(model, data, ts_normalized, device, out_dir):
    """
    Test C: Static shortcut via ablation (H3)
    """
    print("\n" + "="*70)
    print("TEST C: STATIC SHORTCUT ABLATION (H3)")
    print("="*70)
    
    test_data = data['test']
    
    # Sample for faster evaluation (or use full test)
    if len(test_data) > 2000:
        indices = np.random.choice(len(test_data), 2000, replace=False)
        test_sample = test_data[indices]
    else:
        test_sample = test_data
    
    print(f"Evaluating on {len(test_sample)} test triples...")
    
    def evaluate_mrr(model, triples, time_mode='full'):
        """
        Compute MRR on triples with different time modes.
        time_mode: 'full', 'time_off', 'static_only'
        """
        ranks = []
        
        for h, r, t_true, time_id in triples:
            if ts_normalized:
                time_norm = ts_normalized[int(time_id)]
            else:
                time_norm = 100.0 * time_id / 364
            
            with torch.no_grad():
                if time_mode == 'time_off':
                    # Zero out time: use t=0 or mean time
                    time_norm = 0.0
                
                # Get all entity scores for (h, r, ?, time)
                x = torch.tensor([[h, r, 0, time_norm]], dtype=torch.float32, device=device)
                x = x.repeat(model.sizes[0], 1)  # (n_entities, 4)
                x[:, 2] = torch.arange(model.sizes[0], device=device)  # all tails
                
                scores, _, _ = model.forward(x)  # (n_entities,)
                scores = scores.cpu().numpy()
            
            # Rank of true tail
            rank = 1 + np.sum(scores > scores[t_true])
            ranks.append(rank)
        
        mrr = np.mean(1.0 / np.array(ranks))
        hits_at_1 = np.mean(np.array(ranks) <= 1)
        hits_at_3 = np.mean(np.array(ranks) <= 3)
        hits_at_10 = np.mean(np.array(ranks) <= 10)
        
        return {
            'mrr': float(mrr),
            'hits@1': float(hits_at_1),
            'hits@3': float(hits_at_3),
            'hits@10': float(hits_at_10)
        }
    
    print("\n  Mode: Full (with time)")
    metrics_full = evaluate_mrr(model, test_sample, time_mode='full')
    
    print("  Mode: Time-off (time=0)")
    metrics_time_off = evaluate_mrr(model, test_sample, time_mode='time_off')
    
    stats = {
        'full': metrics_full,
        'time_off': metrics_time_off,
        'mrr_drop': float(metrics_full['mrr'] - metrics_time_off['mrr']),
        'diagnosis': ''
    }
    
    # Diagnosis
    mrr_drop = stats['mrr_drop']
    if abs(mrr_drop) < 0.01:
        stats['diagnosis'] = 'STATIC SHORTCUT: Time has no effect on performance'
    elif mrr_drop < 0:
        stats['diagnosis'] = 'ANOMALY: Performance better without time (overfitting?)'
    else:
        stats['diagnosis'] = f'OK: Time improves MRR by {mrr_drop:.4f}'
    
    print(f"\nResults:")
    print(f"  Full MRR: {metrics_full['mrr']:.4f}")
    print(f"  Time-off MRR: {metrics_time_off['mrr']:.4f}")
    print(f"  MRR drop: {mrr_drop:.4f}")
    print(f"  Diagnosis: {stats['diagnosis']}")
    
    return stats


def test_d_gradient_diagnostics(model, data, ts_normalized, batch_size, device, out_dir):
    """
    Test D: Gradient flow diagnostics (H4)
    """
    print("\n" + "="*70)
    print("TEST D: GRADIENT FLOW DIAGNOSTICS (H4)")
    print("="*70)
    
    train_data = data['train']
    
    # Set model to train mode
    model.train()
    
    # Optimizer (dummy, we won't step)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    
    # Track gradient norms
    grad_logs = []
    
    num_batches = min(200, len(train_data) // batch_size)
    print(f"Logging gradients for {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        # Sample batch
        indices = np.random.choice(len(train_data), batch_size, replace=False)
        batch = train_data[indices]
        
        # Convert to continuous time
        batch_tensor = torch.from_numpy(batch).float()
        
        if ts_normalized:
            time_norm = torch.tensor([ts_normalized[int(tid)] for tid in batch[:, 3]], 
                                     dtype=torch.float32)
            batch_tensor[:, 3] = time_norm
        else:
            batch_tensor[:, 3] = 100.0 * batch_tensor[:, 3] / 364
        
        batch_tensor = batch_tensor.to(device)
        
        # Forward
        optimizer.zero_grad()
        scores, factors, time_emb = model.forward(batch_tensor)
        
        # Dummy loss (CrossEntropy)
        truth = batch_tensor[:, 2].long()
        loss = nn.CrossEntropyLoss()(scores, truth)
        
        # Backward
        loss.backward()
        
        # Log gradient norms
        grad_dict = {}
        
        # Entity embeddings
        if hasattr(model, 'entity_embeddings'):
            grad_dict['grad_entities'] = model.entity_embeddings.weight.grad.norm().item()
        
        # Relation embeddings
        if hasattr(model, 'relation_head'):
            grad_dict['grad_relations_head'] = model.relation_head.weight.grad.norm().item()
            grad_dict['grad_relations_tail'] = model.relation_tail.weight.grad.norm().item()
        
        # Time encoder
        if hasattr(model, 'time_encoder'):
            te = model.time_encoder
            if hasattr(te, 'W') and te.W.grad is not None:
                grad_dict['grad_time_W'] = te.W.grad.norm().item()
            if hasattr(te, 'b') and te.b.grad is not None:
                grad_dict['grad_time_b'] = te.b.grad.norm().item()
            if hasattr(te, 'linear'):
                if te.linear.weight.grad is not None:
                    grad_dict['grad_time_linear_weight'] = te.linear.weight.grad.norm().item()
                if te.linear.bias is not None and te.linear.bias.grad is not None:
                    grad_dict['grad_time_linear_bias'] = te.linear.bias.grad.norm().item()
        
        # Alpha (gating)
        if hasattr(model, 'alpha') and model.alpha.weight.grad is not None:
            grad_dict['grad_alpha'] = model.alpha.weight.grad.norm().item()
        
        grad_logs.append(grad_dict)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{num_batches} batches")
    
    # Convert to DataFrame
    df = pd.DataFrame(grad_logs)
    
    # Compute statistics
    stats = {}
    for col in df.columns:
        stats[f'{col}_mean'] = float(df[col].mean())
        stats[f'{col}_median'] = float(df[col].median())
    
    # Compute ratios
    time_grads = []
    all_grads = []
    
    for col in df.columns:
        if 'time' in col:
            time_grads.extend(df[col].values)
        all_grads.extend(df[col].values)
    
    if time_grads and all_grads:
        time_grad_ratio = np.sum(time_grads) / np.sum(all_grads)
        stats['time_grad_ratio'] = float(time_grad_ratio)
    else:
        stats['time_grad_ratio'] = 0.0
    
    # Diagnosis
    if stats['time_grad_ratio'] < 0.01:
        stats['diagnosis'] = 'GRADIENT VANISHING: Time module receives <1% of gradients'
    elif stats['time_grad_ratio'] < 0.05:
        stats['diagnosis'] = 'WEAK GRADIENTS: Time module receives weak gradients'
    else:
        stats['diagnosis'] = f'OK: Time module receives {stats["time_grad_ratio"]:.2%} of gradients'
    
    print(f"\nResults:")
    print(f"  Time grad ratio: {stats['time_grad_ratio']:.4f}")
    for col in df.columns:
        if col in ['grad_time_W', 'grad_time_b', 'grad_time_linear_weight']:
            print(f"  {col} mean: {stats[f'{col}_mean']:.6f}")
    print(f"  Diagnosis: {stats['diagnosis']}")
    
    # Save CSV
    csv_path = Path(out_dir) / 'grad_norms.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved gradient logs to {csv_path}")
    
    # Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of gradient norms
        df_plot = df[[c for c in df.columns if 'time' in c or 'entities' in c or 'relations' in c]]
        df_plot.boxplot(ax=ax, rot=45)
        ax.set_title('Gradient Norms by Module')
        ax.set_ylabel('Gradient Norm')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plot_path = Path(out_dir) / 'plots' / 'test_d_gradient_norms.png'
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Restore eval mode
    model.eval()
    
    return stats


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)
    
    print("="*70)
    print("TEMPORAL KNOWLEDGE GRAPH EMBEDDING DIAGNOSTICS")
    print("="*70)
    print(f"Dataset: {args.data_dir}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print(f"Output: {args.out_dir}")
    print("="*70)
    
    # Load data
    data, ts_normalized = load_dataset(args.data_dir)
    
    # Load model
    model, config = load_model(args.ckpt, args.model_type, device)
    
    # Initialize report
    report = {
        'config': {
            'data_dir': args.data_dir,
            'ckpt': args.ckpt,
            'model_type': args.model_type,
            'seed': args.seed
        },
        'tests': {}
    }
    
    # Run tests
    try:
        # Test A: Time embedding variation
        stats_a = test_a_time_embedding_variation(
            model, ts_normalized, args.num_time_samples, device, out_dir
        )
        report['tests']['test_a_time_embedding_variation'] = stats_a
    except Exception as e:
        print(f"Error in Test A: {e}")
        import traceback
        traceback.print_exc()
        report['tests']['test_a_time_embedding_variation'] = {'error': str(e)}
    
    try:
        # Test B: Temporal sensitivity
        stats_b = test_b_temporal_sensitivity(
            model, data, ts_normalized, args.num_triples, args.num_time_alts, device, out_dir
        )
        report['tests']['test_b_temporal_sensitivity'] = stats_b
    except Exception as e:
        print(f"Error in Test B: {e}")
        import traceback
        traceback.print_exc()
        report['tests']['test_b_temporal_sensitivity'] = {'error': str(e)}
    
    try:
        # Test C: Static shortcut ablation
        stats_c = test_c_static_shortcut_ablation(
            model, data, ts_normalized, device, out_dir
        )
        report['tests']['test_c_static_shortcut'] = stats_c
    except Exception as e:
        print(f"Error in Test C: {e}")
        import traceback
        traceback.print_exc()
        report['tests']['test_c_static_shortcut'] = {'error': str(e)}
    
    if args.do_grad_diagnostics:
        try:
            # Test D: Gradient diagnostics
            stats_d = test_d_gradient_diagnostics(
                model, data, ts_normalized, args.batch_size, device, out_dir
            )
            report['tests']['test_d_gradient_flow'] = stats_d
        except Exception as e:
            print(f"Error in Test D: {e}")
            import traceback
            traceback.print_exc()
            report['tests']['test_d_gradient_flow'] = {'error': str(e)}
    
    # Save report
    report_path = out_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"Report saved to: {report_path}")
    print("\nSummary of findings:")
    for test_name, test_stats in report['tests'].items():
        if 'diagnosis' in test_stats:
            print(f"  {test_name}: {test_stats['diagnosis']}")
        elif 'error' in test_stats:
            print(f"  {test_name}: ERROR - {test_stats['error']}")
    print("="*70)


if __name__ == '__main__':
    main()
