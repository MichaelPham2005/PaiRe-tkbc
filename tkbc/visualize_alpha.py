#!/usr/bin/env python3
# Visualize Alpha Weights from trained ContinuousPairRE model

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_model_checkpoint(checkpoint_path):
    """Load trained model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def extract_alpha_weights(checkpoint):
    """Extract alpha weights from checkpoint"""
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Find alpha weights
    alpha_key = None
    for key in state_dict.keys():
        if 'alpha' in key.lower():
            alpha_key = key
            break
    
    if alpha_key is None:
        raise KeyError("Cannot find alpha weights in checkpoint")
    
    alpha_raw = state_dict[alpha_key].cpu()
    # Apply sigmoid to get final alpha values
    alpha_values = torch.sigmoid(alpha_raw).numpy().flatten()
    
    return alpha_values

def plot_alpha_distribution(alpha_values, save_path=None):
    """Plot histogram of alpha values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    counts, bins, patches = ax.hist(alpha_values, bins=50, edgecolor='black', alpha=0.7)
    
    # Color bars based on value (blue for static, red for dynamic)
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.3:
            patch.set_facecolor('blue')
        elif bin_center > 0.7:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')
    
    # Add vertical lines for thresholds
    ax.axvline(0.3, color='blue', linestyle='--', linewidth=2, label='Static threshold (0.3)')
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Dynamic threshold (0.7)')
    
    ax.set_xlabel('Alpha Value', fontsize=12)
    ax.set_ylabel('Number of Relations', fontsize=12)
    ax.set_title('Distribution of Alpha Weights\n(Static vs Dynamic Relations)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {alpha_values.mean():.4f}\n'
    stats_text += f'Std: {alpha_values.std():.4f}\n'
    stats_text += f'Min: {alpha_values.min():.4f}\n'
    stats_text += f'Max: {alpha_values.max():.4f}\n'
    stats_text += f'\nStatic (α<0.3): {(alpha_values < 0.3).sum()}\n'
    stats_text += f'Mixed (0.3≤α≤0.7): {((alpha_values >= 0.3) & (alpha_values <= 0.7)).sum()}\n'
    stats_text += f'Dynamic (α>0.7): {(alpha_values > 0.7).sum()}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved histogram to: {save_path}")
    
    return fig

def plot_alpha_sorted(alpha_values, save_path=None):
    """Plot sorted alpha values"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort alpha values
    sorted_alpha = np.sort(alpha_values)
    relation_ids = np.arange(len(sorted_alpha))
    
    # Create color map
    colors = []
    for val in sorted_alpha:
        if val < 0.3:
            colors.append('blue')
        elif val > 0.7:
            colors.append('red')
        else:
            colors.append('green')
    
    # Plot
    ax.scatter(relation_ids, sorted_alpha, c=colors, alpha=0.6, s=20)
    
    # Add horizontal lines
    ax.axhline(0.3, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Static threshold')
    ax.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Dynamic threshold')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Neutral (0.5)')
    
    ax.set_xlabel('Relation Index (sorted)', fontsize=12)
    ax.set_ylabel('Alpha Value', fontsize=12)
    ax.set_title('Sorted Alpha Values for All Relations', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sorted plot to: {save_path}")
    
    return fig

def plot_alpha_categories(alpha_values, save_path=None):
    """Plot pie chart of alpha categories"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Categorize
    static_count = (alpha_values < 0.3).sum()
    mixed_count = ((alpha_values >= 0.3) & (alpha_values <= 0.7)).sum()
    dynamic_count = (alpha_values > 0.7).sum()
    
    sizes = [static_count, mixed_count, dynamic_count]
    labels = [f'Static\n(α < 0.3)\n{static_count} relations',
              f'Mixed\n(0.3 ≤ α ≤ 0.7)\n{mixed_count} relations',
              f'Dynamic\n(α > 0.7)\n{dynamic_count} relations']
    colors = ['blue', 'green', 'red']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    
    ax.set_title('Categorization of Relations by Alpha Values', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved pie chart to: {save_path}")
    
    return fig

def plot_top_bottom_relations(alpha_values, n=20, save_path=None):
    """Plot top and bottom relations by alpha"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get indices
    sorted_indices = np.argsort(alpha_values)
    bottom_indices = sorted_indices[:n]
    top_indices = sorted_indices[-n:][::-1]
    
    # Bottom (most static)
    y_pos = np.arange(n)
    ax1.barh(y_pos, alpha_values[bottom_indices], color='blue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'Rel {i}' for i in bottom_indices])
    ax1.set_xlabel('Alpha Value', fontsize=11)
    ax1.set_title(f'Top {n} Most Static Relations\n(Lowest Alpha)', fontsize=12, fontweight='bold')
    ax1.axvline(0.3, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top (most dynamic)
    ax2.barh(y_pos, alpha_values[top_indices], color='red', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Rel {i}' for i in top_indices])
    ax2.set_xlabel('Alpha Value', fontsize=11)
    ax2.set_title(f'Top {n} Most Dynamic Relations\n(Highest Alpha)', fontsize=12, fontweight='bold')
    ax2.axvline(0.7, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved top/bottom plot to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Alpha weights from trained ContinuousPairRE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., models/ICEWS14/ContinuousPairRE/best_valid.pt)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualization plots')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (default: False)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ALPHA WEIGHTS VISUALIZATION")
    print("="*70)
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = load_model_checkpoint(args.checkpoint)
    
    # Extract alpha values
    print("📊 Extracting alpha weights...")
    alpha_values = extract_alpha_weights(checkpoint)
    
    print(f"\n✓ Loaded {len(alpha_values)} alpha values")
    print(f"\nStatistics:")
    print(f"  Mean:  {alpha_values.mean():.4f}")
    print(f"  Std:   {alpha_values.std():.4f}")
    print(f"  Min:   {alpha_values.min():.4f}")
    print(f"  Max:   {alpha_values.max():.4f}")
    print(f"  Median: {np.median(alpha_values):.4f}")
    
    print(f"\nCategories:")
    static_count = (alpha_values < 0.3).sum()
    mixed_count = ((alpha_values >= 0.3) & (alpha_values <= 0.7)).sum()
    dynamic_count = (alpha_values > 0.7).sum()
    print(f"  Static (α < 0.3):     {static_count:4d} ({100*static_count/len(alpha_values):.1f}%)")
    print(f"  Mixed (0.3 ≤ α ≤ 0.7): {mixed_count:4d} ({100*mixed_count/len(alpha_values):.1f}%)")
    print(f"  Dynamic (α > 0.7):    {dynamic_count:4d} ({100*dynamic_count/len(alpha_values):.1f}%)")
    
    # Generate plots
    print(f"\n🎨 Generating visualizations...")
    
    plot_alpha_distribution(alpha_values, output_dir / 'alpha_distribution.png')
    plot_alpha_sorted(alpha_values, output_dir / 'alpha_sorted.png')
    plot_alpha_categories(alpha_values, output_dir / 'alpha_categories.png')
    plot_top_bottom_relations(alpha_values, n=20, save_path=output_dir / 'alpha_top_bottom.png')
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("="*70)
    
    if args.show:
        print("\n📊 Displaying plots...")
        plt.show()
    else:
        print("\n💡 Tip: Use --show flag to display plots interactively")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
