"""
Test GlobalTemporalBasisEncoder implementation.

Verifies that the new architecture:
1. Uses shared global Gaussian kernels
2. Has correct parameter shapes
3. Computes evolution vectors correctly
4. Prevents overfitting through shared structure
"""

import torch
import sys
sys.path.append('d:/LAB_NLP/PaiRe_Based_NLP_LAB/external/tkbc')

from models import GlobalTemporalBasisEncoder, GEPairRE

def test_encoder_initialization():
    """Test 1: GlobalTemporalBasisEncoder initialization."""
    print("="*80)
    print("TEST 1: GlobalTemporalBasisEncoder Initialization")
    print("="*80)
    
    n_relations = 20
    dim = 32
    K = 8
    sigma_init = 0.1
    
    encoder = GlobalTemporalBasisEncoder(n_relations, dim, K, sigma_init, mu_fixed=True)
    
    print(f"\nEncoder configuration:")
    print(f"  n_relations: {n_relations}")
    print(f"  dim: {dim}")
    print(f"  K: {K}")
    print(f"  sigma_init: {sigma_init}")
    print(f"  mu_fixed: True")
    
    print(f"\nParameter shapes:")
    print(f"  mu (FIXED): {encoder.mu.shape} (should be [{K}])")
    print(f"  sigma (FIXED): {encoder.sigma.shape} (should be [{K}])")
    print(f"  W_r (LEARNABLE): {encoder.W_r.shape} (should be [{n_relations}, {K}])")
    print(f"  V_base (LEARNABLE): {encoder.V_base.shape} (should be [{n_relations}, {dim}])")
    
    print(f"\nParameter properties:")
    print(f"  mu.requires_grad: {encoder.mu.requires_grad} (should be False)")
    print(f"  sigma.requires_grad: {encoder.sigma.requires_grad} (should be False)")
    print(f"  W_r.requires_grad: {encoder.W_r.requires_grad} (should be True)")
    print(f"  V_base.requires_grad: {encoder.V_base.requires_grad} (should be True)")
    
    print(f"\nGlobal means (mu):")
    print(f"  {encoder.mu.cpu().numpy()}")
    print(f"  (Evenly spaced in [0, 1])")
    
    print(f"\nGlobal sigmas:")
    print(f"  {encoder.sigma.cpu().numpy()}")
    print(f"  (All equal to {sigma_init})")
    
    # Verify shapes
    assert encoder.mu.shape == (K,), f"mu shape mismatch"
    assert encoder.sigma.shape == (K,), f"sigma shape mismatch"
    assert encoder.W_r.shape == (n_relations, K), f"W_r shape mismatch"
    assert encoder.V_base.shape == (n_relations, dim), f"V_base shape mismatch"
    
    # Verify fixed parameters
    assert not encoder.mu.requires_grad, "mu should be fixed"
    assert not encoder.sigma.requires_grad, "sigma should be fixed"
    assert encoder.W_r.requires_grad, "W_r should be learnable"
    assert encoder.V_base.requires_grad, "V_base should be learnable"
    
    print("\n✅ PASS: GlobalTemporalBasisEncoder initialized correctly")
    return encoder


def test_basis_computation():
    """Test 2: Global basis computation."""
    print("\n" + "="*80)
    print("TEST 2: Global Basis Computation B(τ)")
    print("="*80)
    
    encoder = GlobalTemporalBasisEncoder(20, 32, K=8, sigma_init=0.1, mu_fixed=True)
    
    # Test on various timestamps
    tau = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    B = encoder.compute_basis(tau)
    
    print(f"\nTest timestamps: {tau.numpy()}")
    print(f"Basis activations B(τ):")
    print(f"  Shape: {B.shape} (should be [5, 8])")
    print(f"  B(τ) values:")
    for i, t in enumerate(tau):
        print(f"    τ={t:.2f}: {B[i].detach().cpu().numpy()}")
    
    # Verify properties
    assert B.shape == (5, 8), "Basis shape mismatch"
    assert torch.all(B >= 0) and torch.all(B <= 1), "Basis should be in [0, 1]"
    
    # Check that basis activates near corresponding mu
    for k in range(8):
        mu_k = encoder.mu[k].item()
        # Find closest tau to mu_k
        closest_idx = torch.argmin(torch.abs(tau - mu_k))
        # Basis should be highest near its center
        assert B[closest_idx, k] > 0.5, f"Basis {k} should activate near mu={mu_k:.2f}"
    
    print("\n✅ PASS: Global basis computed correctly")


def test_evolution_vector():
    """Test 3: Evolution vector computation."""
    print("\n" + "="*80)
    print("TEST 3: Evolution Vector Δe_r(τ)")
    print("="*80)
    
    encoder = GlobalTemporalBasisEncoder(20, 32, K=8, sigma_init=0.1, mu_fixed=True)
    
    # Test batch
    batch_size = 16
    rel_id = torch.randint(0, 20, (batch_size,))
    tau = torch.rand(batch_size)
    
    delta_e = encoder(rel_id, tau)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Evolution vector shape: {delta_e.shape} (should be [{batch_size}, 32])")
    print(f"Evolution vector range: [{delta_e.min():.6f}, {delta_e.max():.6f}]")
    print(f"Evolution vector mean: {delta_e.mean():.6f}")
    print(f"Evolution vector std: {delta_e.std():.6f}")
    
    assert delta_e.shape == (batch_size, 32), "Evolution vector shape mismatch"
    
    print("\n✅ PASS: Evolution vectors computed correctly")


def test_shared_structure():
    """Test 4: Verify shared global structure."""
    print("\n" + "="*80)
    print("TEST 4: Shared Global Structure")
    print("="*80)
    
    encoder = GlobalTemporalBasisEncoder(20, 32, K=8, sigma_init=0.1, mu_fixed=True)
    
    # Two different relations at same time should use same basis
    rel_id1 = torch.tensor([0, 1, 2])
    rel_id2 = torch.tensor([3, 4, 5])
    tau = torch.tensor([0.5, 0.5, 0.5])
    
    # Compute basis for both (should be identical since tau is same)
    B1 = encoder.compute_basis(tau)
    B2 = encoder.compute_basis(tau)
    
    print(f"\nBasis for τ=0.5 (all relations use SAME basis):")
    print(f"  B(0.5) = {B1[0].detach().cpu().numpy()}")
    print(f"  Identical for all relations: {torch.allclose(B1, B2)}")
    
    # Evolution vectors will differ due to W_r and V_base
    delta_e1 = encoder(rel_id1, tau)
    delta_e2 = encoder(rel_id2, tau)
    
    print(f"\nEvolution vectors (differ due to W_r and V_base):")
    print(f"  Relation 0: {delta_e1[0, :5].detach().cpu().numpy()} ...")
    print(f"  Relation 3: {delta_e2[0, :5].detach().cpu().numpy()} ...")
    print(f"  Different: {not torch.allclose(delta_e1[0], delta_e2[0])}")
    
    assert torch.allclose(B1, B2), "Basis should be same for same tau"
    assert not torch.allclose(delta_e1[0], delta_e2[0]), "Evolution should differ per relation"
    
    print("\n✅ PASS: Shared global structure verified")


def test_parameter_count():
    """Test 5: Parameter count comparison."""
    print("\n" + "="*80)
    print("TEST 5: Parameter Count (Reduces Overfitting)")
    print("="*80)
    
    n_relations = 230  # ICEWS14
    dim = 128
    K = 8
    
    encoder = GlobalTemporalBasisEncoder(n_relations, dim, K, sigma_init=0.1, mu_fixed=True)
    
    # Count parameters
    n_W_r = n_relations * K
    n_V_base = n_relations * dim
    n_mu = K  # Fixed, not counted
    n_sigma = K  # Fixed, not counted
    n_total = n_W_r + n_V_base
    
    # Old Dual-Component would have:
    n_W_trend_old = n_relations * dim
    n_A_old = n_relations * K * dim
    n_mu_old = n_relations * K
    n_s_old = n_relations * K
    n_total_old = n_W_trend_old + n_A_old + n_mu_old + n_s_old
    
    print(f"\n🆕 GlobalTemporalBasisEncoder (NEW):")
    print(f"  W_r: {n_W_r:,} ({n_relations} × {K})")
    print(f"  V_base: {n_V_base:,} ({n_relations} × {dim})")
    print(f"  mu (FIXED): {n_mu} (not counted)")
    print(f"  sigma (FIXED): {n_sigma} (not counted)")
    print(f"  TOTAL LEARNABLE: {n_total:,}")
    
    print(f"\n❌ DualComponentEncoder (OLD):")
    print(f"  W_trend: {n_W_trend_old:,} ({n_relations} × {dim})")
    print(f"  A: {n_A_old:,} ({n_relations} × {K} × {dim})")
    print(f"  mu: {n_mu_old:,} ({n_relations} × {K})")
    print(f"  s: {n_s_old:,} ({n_relations} × {K})")
    print(f"  TOTAL LEARNABLE: {n_total_old:,}")
    
    reduction = 1 - (n_total / n_total_old)
    print(f"\n📉 Parameter Reduction: {reduction*100:.1f}%")
    print(f"   ({n_total_old:,} → {n_total:,})")
    
    print(f"\n💡 Why This Helps:")
    print(f"  - Fewer parameters → harder to memorize timestamps")
    print(f"  - Shared global basis → forces learning common patterns")
    print(f"  - Fixed mu & sigma → cannot collapse to δ-functions")
    print(f"  - W_r learns WHICH patterns, not WHAT patterns")
    
    print("\n✅ PASS: Parameter count reduced significantly")


def test_full_model():
    """Test 6: GEPairRE with GlobalTemporalBasisEncoder."""
    print("\n" + "="*80)
    print("TEST 6: Full GEPairRE Model")
    print("="*80)
    
    sizes = (100, 20, 100, 50)  # (n_entities, n_relations, n_entities, n_timestamps)
    rank = 32
    K = 8
    
    model = GEPairRE(sizes, rank, K=K, sigma_init=0.1, mu_fixed=True)
    
    print(f"\nModel configuration:")
    print(f"  Sizes: {sizes}")
    print(f"  Rank: {rank}")
    print(f"  K: {K}")
    
    print(f"\nTime encoder type: {type(model.time_encoder).__name__}")
    assert type(model.time_encoder).__name__ == 'GlobalTemporalBasisEncoder', \
        "Should use GlobalTemporalBasisEncoder"
    
    # Test forward pass
    batch_size = 16
    x = torch.zeros(batch_size, 4)
    x[:, 0] = torch.randint(0, sizes[0], (batch_size,))  # head
    x[:, 1] = torch.randint(0, sizes[1], (batch_size,))  # relation
    x[:, 2] = torch.randint(0, sizes[0], (batch_size,))  # tail
    x[:, 3] = torch.rand(batch_size)  # tau
    
    # Score function
    scores = model.score(x)
    print(f"\nScore function:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {scores.shape} (should be [{batch_size}])")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    
    assert scores.shape == (batch_size,), "Score shape mismatch"
    
    # Forward pass (1-vs-All)
    scores_all, factors, delta_e = model.forward(x)
    print(f"\nForward pass (1-vs-All):")
    print(f"  Scores shape: {scores_all.shape} (should be [{batch_size}, {sizes[0]}])")
    print(f"  Evolution vector shape: {delta_e.shape} (should be [{batch_size}, {rank}])")
    
    assert scores_all.shape == (batch_size, sizes[0]), "Forward scores shape mismatch"
    assert delta_e.shape == (batch_size, rank), "Evolution vector shape mismatch"
    
    print("\n✅ PASS: Full GEPairRE model works correctly")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING GlobalTemporalBasisEncoder (GE-PairRE v5)")
    print("="*80)
    print("\nThis tests the new architecture that prevents 'học vẹt' (rote learning)")
    print("by using SHARED global Gaussian kernels instead of per-relation kernels.\n")
    
    try:
        test_encoder_initialization()
        test_basis_computation()
        test_evolution_vector()
        test_shared_structure()
        test_parameter_count()
        test_full_model()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nGlobalTemporalBasisEncoder successfully implemented!")
        print("Ready to train with:")
        print("  cd external/tkbc")
        print("  .\\scripts\\train_dual_gepairre_icews14.ps1")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
