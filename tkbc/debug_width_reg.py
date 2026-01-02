"""
Debug width regularizer and sigma values.
"""
import torch
import sys
from models import GEPairRE

def check_width_regularizer():
    print("=" * 80)
    print("DEBUG: Width Regularizer")
    print("=" * 80)
    
    # Create small model
    n_entities = 100
    n_relations = 10
    n_timestamps = 50
    rank = 32
    K_pulses = 8
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes=sizes, rank=rank, K=K_pulses, sigma_max=0.2)
    
    # Check sigma
    sigma = model.time_encoder.get_sigma()
    print(f"\nSigma shape: {sigma.shape}")
    print(f"Sigma values (rel 0): {sigma[0].detach().numpy()}")
    print(f"Sigma min: {sigma.min().item():.6f}")
    print(f"Sigma max: {sigma.max().item():.6f}")
    print(f"Sigma mean: {sigma.mean().item():.6f}")
    
    # Check width regularization
    from regularizers import WidthPenaltyRegularizer
    
    # Test with different weights
    for weight in [0.0, 0.001, 0.01, 0.1]:
        reg = WidthPenaltyRegularizer(weight=weight, sigma_max=0.2)
        loss = reg.forward(sigma)
        print(f"\nWidth reg (weight={weight}):")
        print(f"  Loss: {loss.item():.6f}")
        
        # Check which sigmas violate
        excess = torch.relu(sigma - 0.2)
        n_violations = (excess > 0).sum().item()
        print(f"  Violations: {n_violations}/{sigma.numel()}")
        if n_violations > 0:
            print(f"  Max excess: {excess.max().item():.6f}")


if __name__ == "__main__":
    check_width_regularizer()
