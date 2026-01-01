"""
Debug actual sigma values during training to see why width regularization = 0.
"""

import torch
import sys

def check_checkpoint_sigma(checkpoint_path):
    """Load checkpoint and check sigma values."""
    print("=" * 80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract sigma parameters
    if 'time_encoder.s' in state_dict:
        s = state_dict['time_encoder.s']  # Log-scale
        sigma = torch.exp(s)  # Actual width
        
        print(f"\nLog-scale (s) statistics:")
        print(f"  Shape: {s.shape}")
        print(f"  Mean: {s.mean().item():.4f}")
        print(f"  Std: {s.std().item():.4f}")
        print(f"  Min: {s.min().item():.4f}")
        print(f"  Max: {s.max().item():.4f}")
        
        print(f"\nSigma (σ = exp(s)) statistics:")
        print(f"  Mean: {sigma.mean().item():.4f}")
        print(f"  Std: {sigma.std().item():.4f}")
        print(f"  Min: {sigma.min().item():.4f}")
        print(f"  Max: {sigma.max().item():.4f}")
        
        print(f"\nSigma distribution:")
        print(f"  σ < 0.1: {(sigma < 0.1).sum().item()} / {sigma.numel()}")
        print(f"  0.1 ≤ σ < 0.2: {((sigma >= 0.1) & (sigma < 0.2)).sum().item()} / {sigma.numel()}")
        print(f"  0.2 ≤ σ < 0.3: {((sigma >= 0.2) & (sigma < 0.3)).sum().item()} / {sigma.numel()}")
        print(f"  σ ≥ 0.3: {(sigma >= 0.3).sum().item()} / {sigma.numel()}")
        
        # Check width penalty
        sigma_max = 0.2
        excess = torch.relu(sigma - sigma_max)
        penalty = excess.sum()
        
        print(f"\nWidth penalty (with σ_max={sigma_max}):")
        print(f"  Excess widths: {(excess > 0).sum().item()} / {sigma.numel()}")
        print(f"  Total penalty (unweighted): {penalty.item():.6f}")
        print(f"  With weight=0.01: {(0.01 * penalty).item():.6f}")
        
        # Sample some relations
        print(f"\nSample relations (first 3):")
        for r in range(min(3, sigma.shape[0])):
            print(f"  Relation {r}: σ = {sigma[r].numpy()}")
    else:
        print("ERROR: No 'time_encoder.s' found in checkpoint!")
        print(f"Available keys: {list(state_dict.keys())[:10]}")
    
    # Check amplitude
    if 'time_encoder.A' in state_dict:
        A = state_dict['time_encoder.A']
        A_norm = torch.norm(A, dim=2)  # (n_rel, K)
        
        print(f"\nAmplitude (A) statistics:")
        print(f"  Shape: {A.shape}")
        print(f"  ||A|| mean: {A_norm.mean().item():.6f}")
        print(f"  ||A|| max: {A_norm.max().item():.6f}")

if __name__ == "__main__":
    # Check latest checkpoint
    checkpoint_path = "models/ICEWS14/GEPairRE/latest.pt"
    
    try:
        check_checkpoint_sigma(checkpoint_path)
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or provide correct path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
