"""
Debug Gaussian temporal encoder to check if it produces non-zero outputs.
"""

import torch
from models import GaussianTemporalEncoder

def test_gaussian_encoder():
    print("=" * 80)
    print("TEST: Gaussian Temporal Encoder Output")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K_pulses = 4
    
    encoder = GaussianTemporalEncoder(n_relations=n_relations, dim=dim, K=K_pulses)
    encoder.eval()
    
    # Test with multiple times
    rel_id = torch.tensor([0, 1, 2])  # 3 different relations
    tau = torch.tensor([0.2, 0.5, 0.8])  # 3 different times
    
    print(f"\nInput:")
    print(f"  rel_id: {rel_id}")
    print(f"  tau: {tau}")
    
    # Get encoder parameters
    print(f"\nEncoder parameters:")
    print(f"  A (amplitude) shape: {encoder.A.shape}")
    print(f"  A values (rel 0, pulse 0): {encoder.A[0, 0, :5].detach().numpy()}")
    print(f"  mu (mean) shape: {encoder.mu.shape}")
    print(f"  mu values (rel 0): {encoder.mu[0, :].detach().numpy()}")
    print(f"  s (log-scale) shape: {encoder.s.shape}")
    print(f"  s values (rel 0): {encoder.s[0, :].detach().numpy()}")
    
    # Forward pass
    with torch.no_grad():
        delta_e = encoder(rel_id, tau)
    
    print(f"\nOutput:")
    print(f"  delta_e shape: {delta_e.shape}")
    print(f"  delta_e[0] (first 5 dims): {delta_e[0, :5].numpy()}")
    print(f"  delta_e[1] (first 5 dims): {delta_e[1, :5].numpy()}")
    print(f"  delta_e[2] (first 5 dims): {delta_e[2, :5].numpy()}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {delta_e.mean().item():.6f}")
    print(f"  Std: {delta_e.std().item():.6f}")
    print(f"  Min: {delta_e.min().item():.6f}")
    print(f"  Max: {delta_e.max().item():.6f}")
    
    # Check activation for each pulse
    print(f"\nPulse activations for rel_id=0:")
    with torch.no_grad():
        rel_expanded = rel_id[0:1].unsqueeze(1).expand(-1, K_pulses)
        tau_expanded = tau[0:1].unsqueeze(1).expand(-1, K_pulses)
        
        mu_rel = encoder.mu[rel_expanded, torch.arange(K_pulses)]
        s_rel = encoder.s[rel_expanded, torch.arange(K_pulses)]
        
        sigma = torch.exp(s_rel)
        variance = sigma ** 2 + 1e-8
        
        # Gaussian activation
        diff = tau_expanded - mu_rel
        G = torch.exp(-0.5 * (diff ** 2) / variance)
        
        print(f"  mu: {mu_rel.squeeze().numpy()}")
        print(f"  sigma: {sigma.squeeze().numpy()}")
        print(f"  G(tau={tau[0]:.2f}): {G.squeeze().numpy()}")
        print(f"  Max activation: {G.max().item():.6f}")


if __name__ == "__main__":
    test_gaussian_encoder()
