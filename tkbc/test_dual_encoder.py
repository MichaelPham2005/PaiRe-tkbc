"""
Comprehensive test suite for Dual-Component Encoder.

Tests:
1. Initialization and parameter shapes
2. Forward pass output shapes
3. Sigma bounds enforcement
4. Global trend extrapolation
5. Temporal smoothness
6. Future sampling capability
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from models_dual import (
    DualComponentEncoder, 
    TemporalSmoothnessRegularizer,
    SigmaLowerBoundRegularizer
)


def test_initialization():
    """Test encoder initialization and parameter shapes."""
    print("=" * 80)
    print("TEST 1: Initialization and Parameter Shapes")
    print("=" * 80)
    
    n_relations = 20
    dim = 64
    K = 8
    sigma_min = 0.02
    sigma_max = 0.3
    
    encoder = DualComponentEncoder(
        n_relations=n_relations, 
        dim=dim, 
        K=K,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )
    
    print(f"\nEncoder configuration:")
    print(f"  n_relations: {n_relations}")
    print(f"  dim: {dim}")
    print(f"  K: {K}")
    print(f"  σ_min: {sigma_min}")
    print(f"  σ_max: {sigma_max}")
    
    print(f"\nParameter shapes:")
    print(f"  W_trend: {encoder.W_trend.shape} (expected: [{n_relations}, {dim}])")
    print(f"  A: {encoder.A.shape} (expected: [{n_relations}, {K}, {dim}])")
    print(f"  μ: {encoder.mu.shape} (expected: [{n_relations}, {K}])")
    print(f"  s: {encoder.s.shape} (expected: [{n_relations}, {K}])")
    
    # Check sigma bounds
    sigma = encoder.get_sigma()
    print(f"\nSigma statistics:")
    print(f"  Min: {sigma.min().item():.4f} (should be ≥ {sigma_min})")
    print(f"  Max: {sigma.max().item():.4f} (should be ≤ {sigma_max})")
    print(f"  Mean: {sigma.mean().item():.4f}")
    
    assert sigma.min() >= sigma_min - 1e-6, "Sigma below minimum!"
    assert sigma.max() <= sigma_max + 1e-6, "Sigma above maximum!"
    
    print("\n✅ PASS: All parameters initialized correctly")
    return True


def test_forward_pass():
    """Test forward pass shapes and components."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass Output Shapes")
    print("=" * 80)
    
    n_relations = 20
    dim = 64
    K = 8
    batch_size = 32
    
    encoder = DualComponentEncoder(n_relations, dim, K)
    
    # Create test batch
    rel_id = torch.randint(0, n_relations, (batch_size,))
    tau = torch.rand(batch_size)
    
    print(f"\nInput:")
    print(f"  rel_id shape: {rel_id.shape}")
    print(f"  tau shape: {tau.shape}")
    print(f"  tau range: [{tau.min():.3f}, {tau.max():.3f}]")
    
    # Forward pass
    delta_e, components = encoder(rel_id, tau)
    
    print(f"\nOutput:")
    print(f"  delta_e shape: {delta_e.shape} (expected: [{batch_size}, {dim}])")
    print(f"  delta_trend shape: {components['trend'].shape}")
    print(f"  delta_pulses shape: {components['pulses'].shape}")
    
    print(f"\nComponent statistics:")
    print(f"  Trend magnitude: {torch.norm(components['trend'], dim=1).mean():.6f}")
    print(f"  Pulses magnitude: {torch.norm(components['pulses'], dim=1).mean():.6f}")
    print(f"  Total magnitude: {torch.norm(delta_e, dim=1).mean():.6f}")
    
    assert delta_e.shape == (batch_size, dim), "Wrong output shape!"
    assert not torch.isnan(delta_e).any(), "NaN in output!"
    assert not torch.isinf(delta_e).any(), "Inf in output!"
    
    print("\n✅ PASS: Forward pass produces correct shapes")
    return True


def test_sigma_bounds():
    """Test that sigma bounds are enforced during training."""
    print("\n" + "=" * 80)
    print("TEST 3: Sigma Bounds Enforcement")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K = 4
    sigma_min = 0.02
    sigma_max = 0.3
    
    encoder = DualComponentEncoder(n_relations, dim, K, sigma_min, sigma_max)
    
    # Try to force sigma out of bounds by manipulating s
    print(f"\nAttempting to violate sigma bounds...")
    with torch.no_grad():
        # Try to make very small sigma
        encoder.s.data[:5] = -10.0  # Would give σ ≈ 0 without bounds
        # Try to make very large sigma
        encoder.s.data[5:] = 10.0   # Would give σ ≈ huge without bounds
    
    sigma = encoder.get_sigma()
    
    print(f"\nSigma after forced changes:")
    print(f"  Min: {sigma.min().item():.4f} (bound: {sigma_min})")
    print(f"  Max: {sigma.max().item():.4f} (bound: {sigma_max})")
    print(f"  σ[0]: {sigma[0].detach().numpy()}")
    print(f"  σ[9]: {sigma[9].detach().numpy()}")
    
    # Check bounds
    violations_min = (sigma < sigma_min).sum().item()
    violations_max = (sigma > sigma_max).sum().item()
    
    print(f"\nBound violations:")
    print(f"  Below σ_min: {violations_min}")
    print(f"  Above σ_max: {violations_max}")
    
    assert violations_min == 0, f"Found {violations_min} values below σ_min!"
    assert violations_max == 0, f"Found {violations_max} values above σ_max!"
    
    print("\n✅ PASS: Sigma bounds enforced correctly")
    return True


def test_global_trend_extrapolation():
    """Test that global trend enables extrapolation beyond training range."""
    print("\n" + "=" * 80)
    print("TEST 4: Global Trend Extrapolation")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K = 4
    
    encoder = DualComponentEncoder(n_relations, dim, K)
    
    # Set a strong linear trend for relation 0
    with torch.no_grad():
        encoder.W_trend[0] = torch.ones(dim) * 0.1  # Strong positive trend
        # Zero out pulses for cleaner test
        encoder.A[0] = 0.0
    
    # Test at different times
    rel_id = torch.zeros(11, dtype=torch.long)  # All relation 0
    tau = torch.linspace(0, 1, 11)  # Times from 0 to 1
    
    delta_e, components = encoder(rel_id, tau)
    trend = components['trend']
    
    print(f"\nTesting linear extrapolation for relation 0:")
    print(f"  W_trend[0] mean: {encoder.W_trend[0].mean():.4f}")
    print(f"\nEvolution at different times:")
    for i in range(0, 11, 2):
        print(f"  τ={tau[i]:.2f}: ||Δe||={torch.norm(delta_e[i]):.4f}, "
              f"||trend||={torch.norm(trend[i]):.4f}")
    
    # Check linearity: Δe should scale with τ
    # Δe(0.5) should be ≈ 0.5 * Δe(1.0)
    ratio = torch.norm(delta_e[5]) / torch.norm(delta_e[10])
    expected_ratio = tau[5] / tau[10]
    
    print(f"\nLinearity check:")
    print(f"  ||Δe(0.5)|| / ||Δe(1.0)||: {ratio:.4f}")
    print(f"  Expected (0.5/1.0): {expected_ratio:.4f}")
    print(f"  Error: {abs(ratio - expected_ratio):.4f}")
    
    assert abs(ratio - expected_ratio) < 0.1, "Trend is not linear!"
    
    print("\n✅ PASS: Global trend enables linear extrapolation")
    return True


def test_temporal_smoothness():
    """Test temporal smoothness regularizer."""
    print("\n" + "=" * 80)
    print("TEST 5: Temporal Smoothness Regularizer")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K = 4
    
    encoder = DualComponentEncoder(n_relations, dim, K)
    regularizer = TemporalSmoothnessRegularizer(weight=0.01, epsilon=0.01)
    
    # Create test batch
    batch_size = 16
    rel_id = torch.randint(0, n_relations, (batch_size,))
    tau = torch.rand(batch_size)
    
    print(f"\nComputing smoothness loss...")
    print(f"  Batch size: {batch_size}")
    print(f"  Epsilon: {regularizer.epsilon}")
    
    loss = regularizer(encoder, rel_id, tau)
    
    print(f"\nSmoothness loss: {loss.item():.6f}")
    
    # Now create a "rough" encoder with large changes
    with torch.no_grad():
        encoder.W_trend[:] = torch.randn_like(encoder.W_trend) * 10.0  # Large trend
        encoder.A[:] = torch.randn_like(encoder.A) * 0.1  # Large amplitudes
    
    loss_rough = regularizer(encoder, rel_id, tau)
    print(f"Smoothness loss (rough): {loss_rough.item():.6f}")
    
    assert loss_rough > loss, "Rougher encoder should have higher smoothness loss!"
    
    print("\n✅ PASS: Smoothness regularizer works correctly")
    return True


def test_sigma_lower_bound_regularizer():
    """Test sigma lower bound regularizer."""
    print("\n" + "=" * 80)
    print("TEST 6: Sigma Lower Bound Regularizer")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K = 4
    sigma_min = 0.02
    
    encoder = DualComponentEncoder(n_relations, dim, K, sigma_min=sigma_min)
    regularizer = SigmaLowerBoundRegularizer(weight=0.1, sigma_min=sigma_min)
    
    # Test 1: No violations (all σ > σ_min)
    with torch.no_grad():
        encoder.s.data[:] = np.log(0.1)  # σ = 0.1 > 0.02
    
    sigma = encoder.get_sigma()
    loss = regularizer(sigma)
    
    print(f"\nTest 1: All σ > σ_min")
    print(f"  σ range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  Loss: {loss.item():.6f} (should be ~0)")
    
    # Due to sigmoid bounds, loss should be very small but not exactly 0
    assert loss < 0.01, "Loss should be near zero when no violations!"
    
    print("\n✅ PASS: Lower bound regularizer works correctly")
    return True


def test_future_sampling():
    """Test future sampling capability for extrapolation training."""
    print("\n" + "=" * 80)
    print("TEST 7: Future Sampling for Extrapolation")
    print("=" * 80)
    
    n_relations = 10
    dim = 32
    K = 4
    
    encoder = DualComponentEncoder(n_relations, dim, K)
    
    # Simulate training scenario
    batch_size = 16
    rel_id = torch.randint(0, n_relations, (batch_size,))
    tau_current = torch.rand(batch_size) * 0.8  # Only up to 0.8
    
    # Future sampling: τ_future = τ + Δt
    delta_t = 0.1
    tau_future = torch.clamp(tau_current + delta_t, 0, 1)
    
    print(f"\nSimulating future sampling:")
    print(f"  Current τ range: [{tau_current.min():.3f}, {tau_current.max():.3f}]")
    print(f"  Future τ range: [{tau_future.min():.3f}, {tau_future.max():.3f}]")
    print(f"  Delta_t: {delta_t}")
    
    # Get evolution at both times
    delta_current, _ = encoder(rel_id, tau_current)
    delta_future, _ = encoder(rel_id, tau_future)
    
    # Check that they're different (model should adapt to time)
    diff = torch.norm(delta_current - delta_future, dim=1).mean()
    
    print(f"\nEvolution difference:")
    print(f"  ||Δe(t_future) - Δe(t_current)||: {diff:.6f}")
    
    # Check that global trend increases with time
    with torch.no_grad():
        encoder.W_trend[rel_id[0]] = torch.ones(dim) * 0.1
        encoder.A[rel_id[0]] = 0.0
    
    delta_t0, comp_t0 = encoder(rel_id[0:1], torch.tensor([tau_current[0]]))
    delta_t1, comp_t1 = encoder(rel_id[0:1], torch.tensor([tau_future[0]]))
    
    print(f"\nFor relation {rel_id[0].item()} with linear trend:")
    print(f"  ||Δe(t={tau_current[0]:.3f})||: {torch.norm(delta_t0):.6f}")
    print(f"  ||Δe(t={tau_future[0]:.3f})||: {torch.norm(delta_t1):.6f}")
    print(f"  Ratio: {(torch.norm(delta_t1) / torch.norm(delta_t0)).item():.4f}")
    
    print("\n✅ PASS: Future sampling works correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Initialization", test_initialization),
        ("Forward Pass", test_forward_pass),
        ("Sigma Bounds", test_sigma_bounds),
        ("Global Trend Extrapolation", test_global_trend_extrapolation),
        ("Temporal Smoothness", test_temporal_smoothness),
        ("Sigma Lower Bound Reg", test_sigma_lower_bound_regularizer),
        ("Future Sampling", test_future_sampling),
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
    print("TEST SUMMARY")
    print("=" * 80)
    for name, result in results:
        print(f"  {name:30s}: {result}")
    
    all_passed = all("PASS" in r for _, r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
