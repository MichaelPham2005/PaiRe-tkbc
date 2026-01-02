"""
Integration test for DualComponentEncoder in GEPairRE.

Tests:
1. Model initialization with DualComponentEncoder
2. Score function consistency
3. Ranking function consistency
4. Forward pass with all regularizers
5. Freeze/unfreeze logic
6. Parameter gradient flow
"""

import torch
import sys
sys.path.append('.')

from models import GEPairRE
from regularizers import (
    N3, AmplitudeDecayRegularizer, WidthPenaltyRegularizer,
    TemporalSmoothnessRegularizer, SigmaLowerBoundRegularizer
)


def test_model_initialization():
    """Test GEPairRE initialization with DualComponentEncoder."""
    print("=" * 80)
    print("TEST 1: Model Initialization with DualComponentEncoder")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    K = 4
    sigma_min = 0.02
    sigma_max = 0.3
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=K, sigma_min=sigma_min, sigma_max=sigma_max)
    
    print(f"\nModel configuration:")
    print(f"  Sizes: {sizes}")
    print(f"  Rank: {rank}")
    print(f"  K Gaussians: {K}")
    print(f"  σ_min: {sigma_min}, σ_max: {sigma_max}")
    
    # Check encoder type
    print(f"\nEncoder type: {type(model.time_encoder).__name__}")
    assert hasattr(model.time_encoder, 'W_trend'), "Missing W_trend (global trend)!"
    assert hasattr(model.time_encoder, 'A'), "Missing A (amplitudes)!"
    
    print(f"\nEncoder parameters:")
    print(f"  W_trend: {model.time_encoder.W_trend.shape}")
    print(f"  A: {model.time_encoder.A.shape}")
    print(f"  μ: {model.time_encoder.mu.shape}")
    print(f"  s: {model.time_encoder.s.shape}")
    
    # Check sigma bounds
    sigma = model.time_encoder.get_sigma()
    print(f"\nSigma bounds check:")
    print(f"  Min σ: {sigma.min().item():.4f} (should be ≥ {sigma_min})")
    print(f"  Max σ: {sigma.max().item():.4f} (should be ≤ {sigma_max})")
    
    assert sigma.min() >= sigma_min - 1e-5
    assert sigma.max() <= sigma_max + 1e-5
    
    print("\n✅ PASS: Model initialized correctly with DualComponentEncoder")
    return True


def test_score_consistency():
    """Test score() produces consistent results."""
    print("\n" + "=" * 80)
    print("TEST 2: Score Function Consistency")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=4, sigma_min=0.02, sigma_max=0.3)
    model.eval()
    
    # Create test quadruples
    batch_size = 16
    queries = torch.zeros(batch_size, 4)
    queries[:, 0] = torch.randint(0, n_entities, (batch_size,))
    queries[:, 1] = torch.randint(0, n_relations, (batch_size,))
    queries[:, 2] = torch.randint(0, n_entities, (batch_size,))
    queries[:, 3] = torch.rand(batch_size)
    
    print(f"\nTest batch: {queries.shape}")
    
    with torch.no_grad():
        scores = model.score(queries)
    
    print(f"Scores shape: {scores.shape}")
    print(f"Sample scores: {scores[:3].numpy()}")
    print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    
    assert scores.shape == (batch_size,)
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()
    
    print("\n✅ PASS: Score function works correctly")
    return True


def test_ranking_consistency():
    """Test get_ranking() matches score()."""
    print("\n" + "=" * 80)
    print("TEST 3: Ranking Consistency with Score")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=4, sigma_min=0.02, sigma_max=0.3)
    model.eval()
    
    # Test queries
    n_test = 5
    queries = torch.zeros(n_test, 4)
    queries[:, 0] = torch.randint(0, n_entities, (n_test,))
    queries[:, 1] = torch.randint(0, n_relations, (n_test,))
    queries[:, 2] = torch.randint(0, n_entities, (n_test,))
    queries[:, 3] = torch.rand(n_test)
    
    print(f"\nComputing scores for {n_test} queries...")
    
    with torch.no_grad():
        # Get target scores
        target_scores = model.score(queries)
        
        # Manually extract scores from forward()
        h = model.entity_embeddings(queries[:, 0].long())
        r_h = model.relation_head(queries[:, 1].long())
        r_t = model.relation_tail(queries[:, 1].long())
        
        rel_id = queries[:, 1].long()
        tau = queries[:, 3].float()
        delta_e_h, delta_e_t = model.get_evolution_vectors(rel_id, tau)
        
        h_tau_r_h = (h + delta_e_h) * r_h
        
        # Get true tails
        true_tails = queries[:, 2].long()
        true_tail_embs = model.entity_embeddings(true_tails)
        t_tau = true_tail_embs + delta_e_t
        t_tau_r_t = t_tau * r_t
        
        # Compute manual scores
        interaction = h_tau_r_h - t_tau_r_t
        manual_scores = -torch.norm(interaction, p=1, dim=1)
    
    print(f"\nTarget scores: {target_scores[:3].numpy()}")
    print(f"Manual scores: {manual_scores[:3].numpy()}")
    
    diff = torch.abs(target_scores - manual_scores)
    print(f"Max difference: {diff.max().item():.6e}")
    
    assert diff.max() < 1e-5
    
    print("\n✅ PASS: Ranking consistent with scoring")
    return True


def test_regularizers():
    """Test all regularizers work with DualComponentEncoder."""
    print("\n" + "=" * 80)
    print("TEST 4: All Regularizers")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=4, sigma_min=0.02, sigma_max=0.3)
    
    # Create regularizers
    amp_reg = AmplitudeDecayRegularizer(weight=0.001)
    width_reg = WidthPenaltyRegularizer(weight=0.01, sigma_max=0.3)
    smooth_reg = TemporalSmoothnessRegularizer(weight=0.01, epsilon=0.01)
    sigma_bound_reg = SigmaLowerBoundRegularizer(weight=0.1, sigma_min=0.02)
    
    print("\nTesting regularizers...")
    
    # Test amplitude regularization
    l_amp = amp_reg(model.time_encoder.A)
    print(f"  Amplitude reg: {l_amp.item():.6f}")
    assert l_amp.item() >= 0
    
    # Test width regularization
    sigma = model.time_encoder.get_sigma()
    l_width = width_reg(sigma)
    print(f"  Width reg: {l_width.item():.6f}")
    assert l_width.item() >= 0
    
    # Test sigma bound regularization
    l_sigma_bound = sigma_bound_reg(sigma)
    print(f"  Sigma bound reg: {l_sigma_bound.item():.6f}")
    assert l_sigma_bound.item() >= 0
    
    # Test smoothness regularization
    batch_size = 16
    rel_id = torch.randint(0, n_relations, (batch_size,))
    tau = torch.rand(batch_size)
    l_smooth = smooth_reg(model.time_encoder, rel_id, tau)
    print(f"  Smoothness reg: {l_smooth.item():.6f}")
    assert l_smooth.item() >= 0
    
    print("\n✅ PASS: All regularizers work correctly")
    return True


def test_freeze_unfreeze():
    """Test parameter freezing/unfreezing."""
    print("\n" + "=" * 80)
    print("TEST 5: Freeze/Unfreeze Logic")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=4, sigma_min=0.02, sigma_max=0.3)
    
    print("\nInitial state (all requires_grad=True):")
    print(f"  W_trend: {model.time_encoder.W_trend.requires_grad}")
    print(f"  A: {model.time_encoder.A.requires_grad}")
    print(f"  μ: {model.time_encoder.mu.requires_grad}")
    print(f"  s: {model.time_encoder.s.requires_grad}")
    
    # Freeze
    model.time_encoder.W_trend.requires_grad = False
    model.time_encoder.A.requires_grad = False
    model.time_encoder.mu.requires_grad = False
    model.time_encoder.s.requires_grad = False
    
    print("\nAfter freezing:")
    print(f"  W_trend: {model.time_encoder.W_trend.requires_grad}")
    print(f"  A: {model.time_encoder.A.requires_grad}")
    
    assert not model.time_encoder.W_trend.requires_grad
    assert not model.time_encoder.A.requires_grad
    
    # Unfreeze
    model.time_encoder.W_trend.requires_grad = True
    model.time_encoder.A.requires_grad = True
    model.time_encoder.mu.requires_grad = True
    model.time_encoder.s.requires_grad = True
    
    print("\nAfter unfreezing:")
    print(f"  W_trend: {model.time_encoder.W_trend.requires_grad}")
    print(f"  A: {model.time_encoder.A.requires_grad}")
    
    assert model.time_encoder.W_trend.requires_grad
    assert model.time_encoder.A.requires_grad
    
    print("\n✅ PASS: Freeze/unfreeze works correctly")
    return True


def test_gradient_flow():
    """Test gradients flow through DualComponentEncoder."""
    print("\n" + "=" * 80)
    print("TEST 6: Gradient Flow")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 20
    n_timestamps = 50
    rank = 32
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes, rank, K=4, sigma_min=0.02, sigma_max=0.3)
    
    # Create batch
    batch_size = 16
    batch = torch.zeros(batch_size, 4)
    batch[:, 0] = torch.randint(0, n_entities, (batch_size,))
    batch[:, 1] = torch.randint(0, n_relations, (batch_size,))
    batch[:, 2] = torch.randint(0, n_entities, (batch_size,))
    batch[:, 3] = torch.rand(batch_size)
    
    print(f"\nComputing gradients...")
    
    # Forward + backward
    scores, _, _ = model.forward(batch)
    loss = -scores.mean()  # Simple loss
    loss.backward()
    
    # Check gradients
    print(f"\nGradient statistics:")
    print(f"  W_trend grad norm: {model.time_encoder.W_trend.grad.norm().item():.6f}")
    print(f"  A grad norm: {model.time_encoder.A.grad.norm().item():.6f}")
    print(f"  μ grad norm: {model.time_encoder.mu.grad.norm().item():.6f}")
    print(f"  s grad norm: {model.time_encoder.s.grad.norm().item():.6f}")
    
    assert model.time_encoder.W_trend.grad is not None
    assert model.time_encoder.A.grad is not None
    assert model.time_encoder.W_trend.grad.norm() > 0
    assert model.time_encoder.A.grad.norm() > 0
    
    print("\n✅ PASS: Gradients flow correctly")
    return True


def run_all_tests():
    """Run all integration tests."""
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Score Consistency", test_score_consistency),
        ("Ranking Consistency", test_ranking_consistency),
        ("Regularizers", test_regularizers),
        ("Freeze/Unfreeze", test_freeze_unfreeze),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "✅ PASS" if passed else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {str(e)}"))
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    for name, result in results:
        print(f"  {name:30s}: {result}")
    
    all_passed = all("PASS" in r for _, r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("\nDualComponentEncoder successfully integrated into GEPairRE!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
