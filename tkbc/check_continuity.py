#!/usr/bin/env python3
"""
Check continuity/smoothness of learned time embeddings m = cos(W·t + b)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from datasets import TemporalDataset
from models import ContinuousPairRE

def check_continuity(checkpoint_path, dataset_name='ICEWS14', rank=156):
    """
    Load trained model and visualize continuity of time embeddings.
    """
    print("="*70)
    print("CONTINUITY CHECK - Time Embeddings m = cos(W·t + b)")
    print("="*70)
    
    # Load dataset
    dataset = TemporalDataset(dataset_name, use_continuous_time=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sizes = dataset.get_shape()
    model = ContinuousPairRE(sizes, rank)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract W and b
    W = model.time_encoder.W.detach().cpu().numpy()
    b = model.time_encoder.b.detach().cpu().numpy()
    
    print(f"\n1. W and b Statistics:")
    print(f"   W shape: {W.shape}")
    print(f"   W mean: {W.mean():.4f}, std: {W.std():.4f}")
    print(f"   W min: {W.min():.4f}, max: {W.max():.4f}")
    print(f"   b mean: {b.mean():.4f}, std: {b.std():.4f}")
    print(f"   b min: {b.min():.4f}, max: {b.max():.4f}")
    
    # Get all normalized timestamps
    timestamps = sorted(dataset.ts_normalized.values())
    t_array = np.array(timestamps)
    
    print(f"\n2. Time Range:")
    print(f"   Normalized time range: [{t_array.min():.2f}, {t_array.max():.2f}]")
    print(f"   Number of timestamps: {len(t_array)}")
    
    # Compute m for all timestamps
    # m = cos(W·t + b) for each dimension
    print(f"\n3. Computing time embeddings m...")
    
    # Create tensor for all times
    t_tensor = torch.tensor(t_array, dtype=torch.float32).unsqueeze(-1)  # (n_times, 1)
    W_tensor = torch.tensor(W, dtype=torch.float32).unsqueeze(0)  # (1, rank)
    b_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)  # (1, rank)
    
    # m = cos(W·t + b)
    m_all = torch.cos(t_tensor * W_tensor + b_tensor)  # (n_times, rank)
    m_np = m_all.numpy()
    
    # Compute differences between consecutive timestamps
    m_diff = np.diff(m_all.numpy(), axis=0)  # (n_times-1, rank)
    
    # Statistics on differences
    diff_norms = np.linalg.norm(m_diff, axis=1)  # L2 norm per timestep
    
    print(f"\n4. Continuity Analysis:")
    print(f"   Mean ||m(t+1) - m(t)||_2: {diff_norms.mean():.6f}")
    print(f"   Std  ||m(t+1) - m(t)||_2: {diff_norms.std():.6f}")
    print(f"   Max  ||m(t+1) - m(t)||_2: {diff_norms.max():.6f}")
    print(f"   Min  ||m(t+1) - m(t)||_2: {diff_norms.min():.6f}")
    
    # Check for discontinuities (large jumps)
    threshold = diff_norms.mean() + 3 * diff_norms.std()
    discontinuities = np.where(diff_norms > threshold)[0]
    
    if len(discontinuities) > 0:
        print(f"\n   ⚠ Warning: {len(discontinuities)} potential discontinuities detected")
        print(f"   (where ||m(t+1) - m(t)||_2 > {threshold:.6f})")
        print(f"   Timestamps with large jumps: {discontinuities[:10].tolist()}")
    else:
        print(f"\n   ✓ No significant discontinuities detected")
        print(f"   All differences are within 3σ of mean")
    
    # Visualize some dimensions
    print(f"\n5. Visualizing time embeddings...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Time Embeddings m = cos(W·t + b) - {checkpoint_path}', fontsize=14)
    
    # Plot 1: First 3 dimensions
    ax = axes[0, 0]
    for dim in range(min(3, rank)):
        ax.plot(t_array, m_np[:, dim], label=f'dim {dim}', alpha=0.7)
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('m value')
    ax.set_title('First 3 dimensions of m(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Difference magnitudes
    ax = axes[0, 1]
    ax.plot(t_array[:-1], diff_norms, color='red', alpha=0.7)
    ax.axhline(y=diff_norms.mean(), color='blue', linestyle='--', label='Mean')
    ax.axhline(y=threshold, color='orange', linestyle='--', label='3σ threshold')
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('||m(t+1) - m(t)||_2')
    ax.set_title('Continuity: Difference Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: W distribution (frequencies)
    ax = axes[1, 0]
    ax.hist(W, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('W values (frequency)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of W (mean={W.mean():.3f}, std={W.std():.3f})')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: b distribution (phases)
    ax = axes[1, 1]
    ax.hist(b, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('b values (phase)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of b (mean={b.mean():.3f}, std={b.std():.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(checkpoint_path).parent
    output_path = output_dir / 'continuity_check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    
    # Check if W values are reasonable
    if np.abs(W).max() > 10:
        print("⚠ WARNING: Some W values are very large (>10)")
        print("  → High frequencies may cause rapid oscillations")
        print("  → Consider increasing --smoothness_reg")
    else:
        print("✓ W values are in reasonable range")
    
    # Check continuity
    if len(discontinuities) > len(t_array) * 0.1:
        print("⚠ WARNING: Many discontinuities detected (>10% of timestamps)")
        print("  → Time embeddings may not be smooth")
        print("  → Consider increasing --smoothness_reg")
    else:
        print("✓ Time embeddings appear to be smooth and continuous")
    
    print(f"\nContinuity score: {1 / (1 + diff_norms.mean()):.4f}")
    print("  (Higher is better, closer to 1 means more continuous)")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check continuity of time embeddings")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., models/ICEWS14/ContinuousPairRE/best_valid.pt)')
    parser.add_argument('--dataset', type=str, default='ICEWS14',
                        help='Dataset name')
    parser.add_argument('--rank', type=int, default=156,
                        help='Embedding rank')
    
    args = parser.parse_args()
    
    check_continuity(args.checkpoint, args.dataset, args.rank)
