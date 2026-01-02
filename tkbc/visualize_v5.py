"""
Visualize the difference between Dual-Component (v4) and Global Basis (v5).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('d:/LAB_NLP/PaiRe_Based_NLP_LAB/external/tkbc')

from models import GlobalTemporalBasisEncoder

def visualize_global_basis():
    """Visualize global Gaussian basis functions."""
    
    # Create encoder
    encoder = GlobalTemporalBasisEncoder(n_relations=5, dim=32, K=8, sigma_init=0.1, mu_fixed=True)
    
    # Create time grid
    tau = torch.linspace(0, 1, 200)
    
    # Compute basis
    B = encoder.compute_basis(tau)  # (200, 8)
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Individual basis functions
    plt.subplot(1, 2, 1)
    for k in range(8):
        plt.plot(tau.numpy(), B[:, k].detach().numpy(), 
                label=f'B_{k}(τ), μ={encoder.mu[k]:.2f}', linewidth=2)
    plt.xlabel('Normalized Time τ', fontsize=12)
    plt.ylabel('Basis Activation', fontsize=12)
    plt.title('Global Gaussian Basis Functions (SHARED)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Sum of basis (coverage)
    plt.subplot(1, 2, 2)
    total_coverage = B.sum(dim=1).detach().numpy()
    plt.plot(tau.numpy(), total_coverage, linewidth=3, color='red')
    plt.xlabel('Normalized Time τ', fontsize=12)
    plt.ylabel('Total Coverage Σ B_k(τ)', fontsize=12)
    plt.title('Temporal Coverage (Overlap)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target=1.0')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('global_basis_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: global_basis_visualization.png")
    plt.close()


def visualize_relation_evolution():
    """Visualize how different relations use the same global basis."""
    
    # Create encoder
    n_relations = 5
    encoder = GlobalTemporalBasisEncoder(n_relations=n_relations, dim=32, K=8, sigma_init=0.1, mu_fixed=True)
    
    # Manually set different W_r for visualization
    encoder.W_r.data = torch.randn(n_relations, 8) * 0.5
    
    # Create time grid
    tau = torch.linspace(0, 1, 200)
    
    # Compute evolution for each relation
    plt.figure(figsize=(14, 10))
    
    # Top: Global basis
    plt.subplot(3, 1, 1)
    B = encoder.compute_basis(tau)
    for k in range(8):
        plt.plot(tau.numpy(), B[:, k].detach().numpy(), alpha=0.3, linewidth=1)
    plt.ylabel('Global Basis B(τ)', fontsize=11)
    plt.title('Global Gaussian Basis (SHARED by all relations)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Middle: W_r weights
    plt.subplot(3, 1, 2)
    W_r_data = encoder.W_r.detach().cpu().numpy()
    im = plt.imshow(W_r_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, label='Weight')
    plt.xlabel('Global Basis Index k', fontsize=11)
    plt.ylabel('Relation r', fontsize=11)
    plt.title('W_r: Which Global Patterns Each Relation Uses (LEARNABLE)', fontsize=13, fontweight='bold')
    plt.xticks(range(8))
    plt.yticks(range(n_relations))
    
    # Bottom: Resulting temporal weights
    plt.subplot(3, 1, 3)
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for r in range(n_relations):
        rel_id = torch.full((len(tau),), r, dtype=torch.long)
        delta_e = encoder(rel_id, tau)  # (200, 32)
        temporal_weight = torch.norm(delta_e, dim=1).detach().numpy()
        plt.plot(tau.numpy(), temporal_weight, label=f'Relation {r}', 
                linewidth=2, color=colors[r])
    
    plt.xlabel('Normalized Time τ', fontsize=11)
    plt.ylabel('||Δe_r(τ)||', fontsize=11)
    plt.title('Resulting Evolution Magnitude (Different per Relation)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('relation_evolution_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: relation_evolution_comparison.png")
    plt.close()


def visualize_parameter_comparison():
    """Compare parameter counts between v4 and v5."""
    
    n_relations = 230  # ICEWS14
    dim = 128
    K = 8
    
    # Dual-Component v4
    n_W_trend_v4 = n_relations * dim
    n_A_v4 = n_relations * K * dim
    n_mu_v4 = n_relations * K
    n_s_v4 = n_relations * K
    n_total_v4 = n_W_trend_v4 + n_A_v4 + n_mu_v4 + n_s_v4
    
    # Global Basis v5
    n_W_r_v5 = n_relations * K
    n_V_base_v5 = n_relations * dim
    n_total_v5 = n_W_r_v5 + n_V_base_v5
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    models = ['Dual-Component\n(v4)', 'Global Basis\n(v5)']
    params = [n_total_v4, n_total_v5]
    colors_bar = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(models, params, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax1.set_title('Learnable Parameters Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart for v4 breakdown
    labels_v4 = ['W_trend\n(29,440)', f'A\n({n_A_v4:,})', 'μ\n(1,840)', 's\n(1,840)']
    sizes_v4 = [n_W_trend_v4, n_A_v4, n_mu_v4, n_s_v4]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    ax2.pie(sizes_v4, labels=labels_v4, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Dual-Component v4\nParameter Breakdown', fontsize=13, fontweight='bold')
    
    # Add reduction text
    reduction = (1 - n_total_v5/n_total_v4) * 100
    fig.text(0.5, 0.02, f'Parameter Reduction: {reduction:.1f}% ({n_total_v4:,} → {n_total_v5:,})',
             ha='center', fontsize=13, fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('parameter_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: parameter_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VISUALIZING GlobalTemporalBasisEncoder (v5)")
    print("="*80)
    
    print("\nGenerating visualizations...")
    visualize_global_basis()
    visualize_relation_evolution()
    visualize_parameter_comparison()
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print("\nFiles created:")
    print("  1. global_basis_visualization.png")
    print("  2. relation_evolution_comparison.png")
    print("  3. parameter_comparison.png")
    print("="*80)
