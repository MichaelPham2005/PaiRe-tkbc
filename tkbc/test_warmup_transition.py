"""
Quick test to verify warmup transition logic works correctly.
"""

import torch
import sys
sys.path.append('.')

from models import GEPairRE
from optimizers import GEPairREOptimizer
from datasets import TemporalDataset

def test_warmup_transition():
    print("=" * 80)
    print("TEST: Warmup Transition Logic")
    print("=" * 80)
    
    # Create small test dataset
    dataset = TemporalDataset("ICEWS14")
    
    # Create model
    sizes = dataset.get_shape()
    model = GEPairRE(sizes=sizes, rank=32, K=4)
    
    # Create optimizer
    opt = torch.optim.Adagrad(model.parameters(), lr=0.1)
    
    emb_reg = 0.001
    time_reg = 0.0
    amplitude_reg = 0.001
    width_reg = 0.01
    
    optimizer = GEPairREOptimizer(
        model, emb_reg, time_reg, amplitude_reg, width_reg, opt, dataset,
        batch_size=100,
        margin=0.3,
        warmup_epochs=3,  # Warmup for 3 epochs
        max_lambda_time=1.0,
        verbose=False
    )
    
    print(f"\nInitial state:")
    print(f"  current_epoch: {optimizer.current_epoch}")
    print(f"  warmup_epochs: {optimizer.warmup_epochs}")
    print(f"  λ_time: {optimizer.get_lambda_time():.4f}")
    
    # Check Gaussian parameters are frozen
    print(f"\nGaussian parameters frozen?")
    print(f"  A.requires_grad: {model.time_encoder.A.requires_grad}")
    print(f"  mu.requires_grad: {model.time_encoder.mu.requires_grad}")
    print(f"  s.requires_grad: {model.time_encoder.s.requires_grad}")
    
    # Simulate epochs
    examples = torch.from_numpy(dataset.get_train().astype('int64'))
    small_batch = examples[:100]  # Just 100 examples for speed
    
    for epoch in range(5):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}")
        print(f"{'='*80}")
        
        # CRITICAL: Pass epoch number
        optimizer.epoch(small_batch, epoch_num=epoch)
        
        print(f"After epoch {epoch}:")
        print(f"  current_epoch: {optimizer.current_epoch}")
        print(f"  λ_time: {optimizer.get_lambda_time():.4f}")
        print(f"  A.requires_grad: {model.time_encoder.A.requires_grad}")
        print(f"  mu.requires_grad: {model.time_encoder.mu.requires_grad}")
        print(f"  s.requires_grad: {model.time_encoder.s.requires_grad}")
        
        if epoch < 3:
            expected_stage = "WARMUP"
            expected_lambda = 0.0
            expected_frozen = False  # Should be frozen (no grad)
        else:
            expected_stage = "DYNAMIC"
            expected_lambda = 0.1 + min((epoch - 3) / 10.0, 1.0) * (1.0 - 0.1)
            expected_frozen = True  # Should be unfrozen (has grad)
        
        actual_lambda = optimizer.get_lambda_time()
        actual_frozen = model.time_encoder.A.requires_grad
        
        print(f"\n  Expected: {expected_stage}, λ={expected_lambda:.4f}, requires_grad={expected_frozen}")
        print(f"  Actual:   λ={actual_lambda:.4f}, requires_grad={actual_frozen}")
        
        if abs(actual_lambda - expected_lambda) < 1e-6 and actual_frozen == expected_frozen:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
            return False
    
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED - Warmup transition works correctly!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_warmup_transition()
    sys.exit(0 if success else 1)
