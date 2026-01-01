"""
Quick verification script for GE-PairRE implementation.
Checks that the model can be instantiated and run a forward pass.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import GEPairRE
from optimizers import GEPairREOptimizer
from regularizers import N3, Lambda3, AmplitudeDecayRegularizer, WidthPenaltyRegularizer
from datasets import TemporalDataset

def verify_model_instantiation():
    """Verify model can be created and moved to GPU."""
    print("="*70)
    print("VERIFICATION 1: Model Instantiation")
    print("="*70)
    
    sizes = (100, 20, 100, 365)  # Small test sizes
    rank = 64
    K = 8
    
    try:
        model = GEPairRE(sizes, rank, K=K, sigma_max=0.2)
        print(f"  ✅ Model created successfully")
        print(f"     - Entities: {sizes[0]}")
        print(f"     - Relations: {sizes[1]}")
        print(f"     - Rank: {rank}")
        print(f"     - K Gaussians: {K}")
        
        # Check if CUDA available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"  ✅ Model moved to GPU")
        else:
            print(f"  ⚠️  CUDA not available, using CPU")
        
        # Check parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Total parameters: {n_params:,}")
        
        return True, model
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False, None


def verify_forward_pass(model):
    """Verify forward pass works."""
    print("\n" + "="*70)
    print("VERIFICATION 2: Forward Pass")
    print("="*70)
    
    try:
        # Create sample batch
        batch_size = 32
        device = next(model.parameters()).device
        
        x = torch.zeros((batch_size, 4), device=device)
        x[:, 0] = torch.randint(0, model.sizes[0], (batch_size,))  # heads
        x[:, 1] = torch.randint(0, model.sizes[1], (batch_size,))  # relations
        x[:, 2] = torch.randint(0, model.sizes[2], (batch_size,))  # tails
        x[:, 3] = torch.rand(batch_size)  # times [0,1]
        
        print(f"  Testing with batch_size={batch_size}")
        
        # Test score function
        scores = model.score(x)
        print(f"  ✅ score() works: output shape {scores.shape}")
        print(f"     - Mean score: {scores.mean().item():.6f}")
        print(f"     - Std score: {scores.std().item():.6f}")
        
        # Test forward (1-vs-All)
        predictions, factors, delta_e = model.forward(x)
        print(f"  ✅ forward() works: output shape {predictions.shape}")
        print(f"     - Predictions: {predictions.shape}")
        print(f"     - Evolution vectors: {delta_e.shape}")
        
        # Test forward_over_time
        x_no_time = x[:, :3]
        time_scores = model.forward_over_time(x_no_time)
        print(f"  ✅ forward_over_time() works: output shape {time_scores.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_optimizer_creation():
    """Verify optimizer can be created."""
    print("\n" + "="*70)
    print("VERIFICATION 3: Optimizer Creation")
    print("="*70)
    
    try:
        # Load a small dataset for testing
        print("  Loading ICEWS14 dataset...")
        dataset = TemporalDataset('ICEWS14', use_continuous_time=True)
        print(f"  ✅ Dataset loaded")
        print(f"     - Train examples: {dataset.get_train().shape[0]:,}")
        
        sizes = dataset.get_shape()
        rank = 64
        K = 4  # Small for quick test
        
        model = GEPairRE(sizes, rank, K=K, sigma_max=0.2)
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create optimizer
        opt = torch.optim.Adagrad(model.parameters(), lr=0.001)
        emb_reg = N3(0.001)
        time_reg = Lambda3(0.0)  # Not used for GE-PairRE
        amp_reg = AmplitudeDecayRegularizer(0.001)
        width_reg = WidthPenaltyRegularizer(0.01, sigma_max=0.2)
        
        optimizer = GEPairREOptimizer(
            model, emb_reg, time_reg, amp_reg, width_reg, opt, dataset,
            batch_size=128,
            margin=0.3,
            warmup_epochs=2,
            max_lambda_time=0.5,
            verbose=False
        )
        
        print(f"  ✅ GEPairREOptimizer created")
        print(f"     - Batch size: 128")
        print(f"     - Warm-up epochs: 2")
        print(f"     - Current epoch: {optimizer.current_epoch}")
        print(f"     - λ_time: {optimizer.get_lambda_time():.2f}")
        
        # Check Gaussian params are frozen
        is_frozen = not model.time_encoder.A.requires_grad
        print(f"  ✅ Gaussian params frozen: {is_frozen}")
        
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_backward_pass(model):
    """Verify gradients can be computed."""
    print("\n" + "="*70)
    print("VERIFICATION 4: Backward Pass")
    print("="*70)
    
    try:
        model.train()
        device = next(model.parameters()).device
        
        # Create sample batch
        x = torch.zeros((16, 4), device=device)
        x[:, 0] = torch.randint(0, model.sizes[0], (16,))
        x[:, 1] = torch.randint(0, model.sizes[1], (16,))
        x[:, 2] = torch.randint(0, model.sizes[2], (16,))
        x[:, 3] = torch.rand(16)
        
        # Forward pass
        predictions, factors, delta_e = model.forward(x)
        targets = x[:, 2].long()
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(predictions, targets)
        print(f"  ✅ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad[name] = param.grad.norm().item()
        
        print(f"  ✅ Gradients computed for {len(has_grad)} parameters")
        
        # Check key gradients
        key_params = ['entity_embeddings.weight', 'time_encoder.A', 'time_encoder.mu']
        for param_name in key_params:
            if param_name in has_grad:
                print(f"     - {param_name}: ||grad|| = {has_grad[param_name]:.2e}")
            else:
                print(f"     - {param_name}: no gradient (possibly frozen)")
        
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("GE-PAIRRE QUICK VERIFICATION")
    print("="*70)
    print()
    
    results = []
    
    # Test 1: Model instantiation
    success, model = verify_model_instantiation()
    results.append(("Model Instantiation", success))
    
    if not success:
        print("\n❌ Model creation failed. Cannot proceed with other tests.")
        return False
    
    # Test 2: Forward pass
    success = verify_forward_pass(model)
    results.append(("Forward Pass", success))
    
    # Test 3: Optimizer creation
    success = verify_optimizer_creation()
    results.append(("Optimizer Creation", success))
    
    # Test 4: Backward pass
    success = verify_backward_pass(model)
    results.append(("Backward Pass", success))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    print(f"\n  Total: {n_passed}/{n_total} verifications passed")
    print("="*70)
    
    if n_passed == n_total:
        print("\n✅ ALL VERIFICATIONS PASSED - Model ready for training!")
        return True
    else:
        print("\n❌ SOME VERIFICATIONS FAILED - Please check errors above")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
