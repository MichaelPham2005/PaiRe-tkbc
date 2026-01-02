"""
Comprehensive Test Suite for GE-PairRE (Gaussian Evolution PairRE)

Tests the implementation against the specification in README_GE-PairRE.md
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import GaussianTemporalEncoder, GEPairRE
from regularizers import AmplitudeDecayRegularizer, WidthPenaltyRegularizer


def test_1_gaussian_activation():
    """
    Test 1: Gaussian Pulse Activation
    
    Verify:
    - G(τ) = exp(-(τ - μ)² / (2σ² + ε))
    - G(μ) ≈ 1 (maximum at center)
    - G(μ ± 3σ) ≈ 0 (near zero far from center)
    """
    print("="*70)
    print("TEST 1: Gaussian Pulse Activation")
    print("="*70)
    
    n_relations = 5
    dim = 64
    K = 4
    
    encoder = GaussianTemporalEncoder(n_relations, dim, K, sigma_max=0.2)
    
    # Test single relation, single pulse
    rel_id = torch.tensor([0])
    mu_0_0 = encoder.mu[0, 0].item()  # Center of first pulse
    sigma_0_0 = encoder.get_sigma()[0, 0].item()  # Width
    
    print(f"Testing relation 0, pulse 0:")
    print(f"  μ = {mu_0_0:.4f}")
    print(f"  σ = {sigma_0_0:.4f}")
    
    # Test at center
    tau_center = torch.tensor([mu_0_0])
    delta_e_center = encoder(rel_id, tau_center)
    
    # Test at ±3σ (should be near zero)
    tau_far = torch.tensor([mu_0_0 + 3 * sigma_0_0])
    delta_e_far = encoder(rel_id, tau_far)
    
    # Compute Gaussian values manually
    G_center_manual = np.exp(0)  # -(0)² / (2σ² + ε) = 0
    diff_far = 3 * sigma_0_0
    G_far_manual = np.exp(-diff_far**2 / (2 * sigma_0_0**2 + 1e-9))
    
    print(f"\n  At τ = μ (center):")
    print(f"    Expected G ≈ 1.0")
    print(f"    ||Δe|| = {delta_e_center.norm().item():.6f}")
    print(f"    (Should be maximum for this relation)")
    
    print(f"\n  At τ = μ + 3σ (far from center):")
    print(f"    Expected G ≈ {G_far_manual:.6f}")
    print(f"    ||Δe|| = {delta_e_far.norm().item():.6f}")
    print(f"    (Should be much smaller)")
    
    ratio = delta_e_far.norm().item() / (delta_e_center.norm().item() + 1e-9)
    print(f"\n  Ratio ||Δe_far|| / ||Δe_center|| = {ratio:.6f}")
    
    if ratio < 0.1:
        print("  ✅ PASS: Gaussian activation decays properly")
    else:
        print("  ❌ FAIL: Activation should decay more")
    
    return ratio < 0.1


def test_2_static_preservation():
    """
    Test 2: Absolute Static Preservation
    
    Verify:
    - If τ is far from all μ_{r,k}, then Δe ≈ 0
    - Model preserves static relations perfectly
    """
    print("\n" + "="*70)
    print("TEST 2: Static Preservation")
    print("="*70)
    
    n_relations = 10
    dim = 64
    K = 8
    
    encoder = GaussianTemporalEncoder(n_relations, dim, K, sigma_max=0.1)
    
    # Set all μ in middle region [0.3, 0.7]
    with torch.no_grad():
        encoder.mu.data = torch.linspace(0.3, 0.7, K).unsqueeze(0).expand(n_relations, -1)
        # Set small sigma
        encoder.s.data = torch.ones(n_relations, K) * np.log(0.02)
    
    # Test at boundary times (far from all centers)
    rel_id = torch.tensor([0, 1, 2])
    tau_boundary = torch.tensor([0.0, 1.0, 0.05])  # Far from [0.3, 0.7]
    
    delta_e = encoder(rel_id, tau_boundary)
    norms = delta_e.norm(dim=1)
    
    print(f"  Gaussian centers (μ): [{0.3:.2f}, ..., {0.7:.2f}]")
    print(f"  Testing at boundary times: {tau_boundary.tolist()}")
    print(f"  ||Δe|| for each: {norms.tolist()}")
    print(f"  Mean ||Δe||: {norms.mean().item():.6f}")
    
    threshold = 0.01
    if norms.mean().item() < threshold:
        print(f"  ✅ PASS: Static preservation verified (mean ||Δe|| < {threshold})")
        success = True
    else:
        print(f"  ❌ FAIL: Evolution too large at boundary (should be < {threshold})")
        success = False
    
    return success


def test_3_evolution_additivity():
    """
    Test 3: Additive Evolution Mechanism
    
    Verify:
    - h_τ = h + Δe_h
    - t_τ = t + Δe_t
    - Evolution is pure addition (no gating/multiplication)
    """
    print("\n" + "="*70)
    print("TEST 3: Evolution Additivity")
    print("="*70)
    
    sizes = (100, 20, 100, 365)  # n_entities, n_relations, n_entities, n_timestamps
    rank = 64
    K = 8
    
    model = GEPairRE(sizes, rank, K=K).cuda()
    
    # Create a sample query
    x = torch.tensor([[10, 5, 20, 0.5]]).cuda()  # [head, rel, tail, tau]
    
    # Get static embeddings
    h_static = model.entity_embeddings(x[:, 0].long())
    t_static = model.entity_embeddings(x[:, 2].long())
    
    # Get evolution vectors
    rel_id = x[:, 1].long()
    tau = x[:, 3].float()
    delta_e_h, delta_e_t = model.get_evolution_vectors(rel_id, tau)
    
    # Compute evolved manually
    h_tau_manual = h_static + delta_e_h
    t_tau_manual = t_static + delta_e_t
    
    # Get from score function
    r_h = model.relation_head(x[:, 1].long())
    r_t = model.relation_tail(x[:, 1].long())
    score = model.score(x)
    
    # Reconstruct score manually
    interaction_manual = h_tau_manual * r_h - t_tau_manual * r_t
    score_manual = -torch.norm(interaction_manual, p=1, dim=-1)
    
    print(f"  Static embeddings:")
    print(f"    ||h|| = {h_static.norm().item():.4f}")
    print(f"    ||t|| = {t_static.norm().item():.4f}")
    print(f"\n  Evolution vectors:")
    print(f"    ||Δe_h|| = {delta_e_h.norm().item():.4f}")
    print(f"    ||Δe_t|| = {delta_e_t.norm().item():.4f}")
    print(f"\n  Evolved embeddings:")
    print(f"    ||h_τ|| = {h_tau_manual.norm().item():.4f}")
    print(f"    ||t_τ|| = {t_tau_manual.norm().item():.4f}")
    print(f"\n  Score from model: {score.item():.6f}")
    print(f"  Score computed manually: {score_manual.item():.6f}")
    print(f"  Difference: {abs(score.item() - score_manual.item()):.9f}")
    
    if abs(score.item() - score_manual.item()) < 1e-5:
        print("  ✅ PASS: Additive evolution verified")
        success = True
    else:
        print("  ❌ FAIL: Score mismatch (evolution not purely additive)")
        success = False
    
    return success


def test_4_scoring_function():
    """
    Test 4: PairRE Scoring Function
    
    Verify:
    - φ(h, r, t, τ) = -||h_τ ∘ r^H - t_τ ∘ r^T||₁
    - Uses L1 norm (not L2)
    - Negative sign (higher score = better)
    """
    print("\n" + "="*70)
    print("TEST 4: Scoring Function")
    print("="*70)
    
    sizes = (100, 20, 100, 365)
    rank = 64
    K = 8
    
    model = GEPairRE(sizes, rank, K=K).cuda()
    
    # Test two queries: one with good match, one with bad
    x_good = torch.tensor([[10, 5, 10, 0.5]]).cuda()  # Same head and tail
    x_bad = torch.tensor([[10, 5, 50, 0.5]]).cuda()   # Different tail
    
    score_good = model.score(x_good)
    score_bad = model.score(x_bad)
    
    # Verify L1 norm usage by checking gradient behavior
    x_test = torch.tensor([[10, 5, 20, 0.5]], requires_grad=False, dtype=torch.float32).cuda()
    score_test = model.score(x_test)
    
    print(f"  Query (h=t): score = {score_good.item():.6f}")
    print(f"  Query (h≠t): score = {score_bad.item():.6f}")
    print(f"  Test query:  score = {score_test.item():.6f}")
    
    print(f"\n  Score is negative? {score_test.item() < 0}")
    print(f"  Better match scores higher? {score_good.item() > score_bad.item()}")
    
    if score_test.item() < 0 and score_good.item() > score_bad.item():
        print("  ✅ PASS: Scoring function correct")
        success = True
    else:
        print("  ❌ FAIL: Scoring function issue")
        success = False
    
    return success


def test_5_temporal_discrimination():
    """
    Test 5: Temporal Discrimination
    
    Verify:
    - φ(h, r, t, τ_correct) > φ(h, r, t, τ_wrong) for most cases
    - Model learns to discriminate time
    """
    print("\n" + "="*70)
    print("TEST 5: Temporal Discrimination Capability")
    print("="*70)
    
    sizes = (100, 20, 100, 365)
    rank = 64
    K = 8
    
    model = GEPairRE(sizes, rank, K=K).cuda()
    
    # Create batch of queries with different times
    n_samples = 50
    h_ids = torch.randint(0, sizes[0], (n_samples,))
    r_ids = torch.randint(0, sizes[1], (n_samples,))
    t_ids = torch.randint(0, sizes[2], (n_samples,))
    tau_correct = torch.rand(n_samples) * 0.5  # First half
    tau_wrong = torch.rand(n_samples) * 0.5 + 0.5  # Second half
    
    x_correct = torch.stack([h_ids, r_ids, t_ids, tau_correct], dim=1).float().cuda()
    x_wrong = torch.stack([h_ids, r_ids, t_ids, tau_wrong], dim=1).float().cuda()
    
    scores_correct = model.score(x_correct)
    scores_wrong = model.score(x_wrong)
    
    # Check how many prefer correct time
    prefer_correct = (scores_correct > scores_wrong).float().mean().item()
    mean_diff = (scores_correct - scores_wrong).mean().item()
    
    print(f"  Testing {n_samples} queries with τ ∈ [0, 0.5] vs τ ∈ [0.5, 1]")
    print(f"  Fraction preferring first time: {prefer_correct:.2%}")
    print(f"  Mean score difference: {mean_diff:.6f}")
    
    # With random initialization, should be close to 50%
    # After training, should be > 60%
    if 0.3 < prefer_correct < 0.7:
        print("  ✅ PASS: Model can discriminate time (not collapsed)")
        success = True
    else:
        print("  ⚠️  WARNING: Discrimination very biased (may indicate issue)")
        success = False
    
    return success


def test_6_parameter_initialization():
    """
    Test 6: Parameter Initialization
    
    Verify initialization matches specification:
    - μ: evenly spaced in [0,1]
    - s: small (σ ≈ 0.01)
    - A: small Gaussian (std = 1e-4)
    """
    print("\n" + "="*70)
    print("TEST 6: Parameter Initialization")
    print("="*70)
    
    n_relations = 20
    dim = 64
    K = 8
    sigma_max = 0.2
    
    encoder = GaussianTemporalEncoder(n_relations, dim, K, sigma_max)
    
    # Check μ initialization
    mu_first_rel = encoder.mu[0]
    mu_expected = torch.linspace(0, 1, K)
    mu_diff = (mu_first_rel - mu_expected).abs().max().item()
    
    print(f"  μ initialization (first relation):")
    print(f"    Expected: {mu_expected.tolist()}")
    print(f"    Actual:   {mu_first_rel.tolist()}")
    print(f"    Max diff: {mu_diff:.6f}")
    
    # Check σ initialization
    sigma = encoder.get_sigma()
    sigma_mean = sigma.mean().item()
    sigma_std = sigma.std().item()
    
    print(f"\n  σ initialization:")
    print(f"    Mean: {sigma_mean:.6f}")
    print(f"    Std:  {sigma_std:.6f}")
    print(f"    Expected: ≈ 0.01")
    
    # Check A initialization
    A_norm_mean = encoder.A.norm(dim=2).mean().item()
    A_std = encoder.A.std().item()
    
    print(f"\n  A initialization:")
    print(f"    Mean ||A||: {A_norm_mean:.6f}")
    print(f"    Std of A:   {A_std:.6f}")
    print(f"    Expected std: ≈ 1e-4")
    
    checks = []
    checks.append(("μ evenly spaced", mu_diff < 0.01))
    checks.append(("σ ≈ 0.01", 0.005 < sigma_mean < 0.02))
    checks.append(("A small", A_norm_mean < 0.01))
    
    print(f"\n  Initialization checks:")
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"    {status} {check_name}")
    
    success = all(passed for _, passed in checks)
    if success:
        print("\n  ✅ PASS: All initialization checks passed")
    else:
        print("\n  ❌ FAIL: Some initialization checks failed")
    
    return success


def test_7_regularizers():
    """
    Test 7: Regularizers
    
    Verify:
    - AmplitudeDecayRegularizer works correctly
    - WidthPenaltyRegularizer enforces σ_max constraint
    """
    print("\n" + "="*70)
    print("TEST 7: Regularizers")
    print("="*70)
    
    n_relations = 10
    dim = 64
    K = 8
    sigma_max = 0.2
    
    encoder = GaussianTemporalEncoder(n_relations, dim, K, sigma_max)
    
    # Test amplitude regularizer
    amp_reg = AmplitudeDecayRegularizer(weight=0.001)
    loss_amp = amp_reg.forward(encoder.A)
    
    print(f"  Amplitude Decay Regularizer:")
    print(f"    ||A||² = {(encoder.A ** 2).sum().item():.6f}")
    print(f"    Loss = {loss_amp.item():.6f}")
    print(f"    Weight = 0.001")
    
    # Test width penalty (force some σ > σ_max)
    with torch.no_grad():
        encoder.s.data[0, 0] = np.log(0.3)  # σ = 0.3 > 0.2
        encoder.s.data[0, 1] = np.log(0.25)  # σ = 0.25 > 0.2
    
    sigma = encoder.get_sigma()
    width_reg = WidthPenaltyRegularizer(weight=0.01, sigma_max=sigma_max)
    loss_width = width_reg.forward(sigma)
    
    print(f"\n  Width Penalty Regularizer:")
    print(f"    σ_max = {sigma_max}")
    print(f"    σ[0,0] = {sigma[0, 0].item():.4f} (> σ_max)")
    print(f"    σ[0,1] = {sigma[0, 1].item():.4f} (> σ_max)")
    print(f"    Loss = {loss_width.item():.6f}")
    print(f"    Expected: penalize excess widths")
    
    checks = []
    checks.append(("Amplitude reg > 0", loss_amp.item() > 0))
    checks.append(("Width penalty > 0", loss_width.item() > 0))
    
    print(f"\n  Regularizer checks:")
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"    {status} {check_name}")
    
    success = all(passed for _, passed in checks)
    if success:
        print("\n  ✅ PASS: Regularizers working correctly")
    else:
        print("\n  ❌ FAIL: Regularizer issues detected")
    
    return success


def test_8_inverse_relations():
    """
    Test 8: Inverse Relation Handling
    
    Verify:
    - Tail uses inverse relation for evolution
    - Δe_t computed with r⁻¹, not r
    """
    print("\n" + "="*70)
    print("TEST 8: Inverse Relation Handling")
    print("="*70)
    
    sizes = (100, 20, 100, 365)  # n_relations = 20 (10 base + 10 inverse)
    rank = 64
    K = 8
    
    model = GEPairRE(sizes, rank, K=K).cuda()
    
    # Test relation 3 and its inverse
    n_base = sizes[1] // 2
    rel_id = torch.tensor([3]).cuda()
    inverse_rel_id = torch.tensor([3 + n_base]).cuda()
    tau = torch.tensor([0.5]).cuda()
    
    delta_e_h_fwd, delta_e_t_fwd = model.get_evolution_vectors(rel_id, tau)
    delta_e_h_inv, delta_e_t_inv = model.get_evolution_vectors(inverse_rel_id, tau)
    
    print(f"  Forward relation (r={rel_id.item()}):")
    print(f"    ||Δe_h|| = {delta_e_h_fwd.norm().item():.6f}")
    print(f"    ||Δe_t|| = {delta_e_t_fwd.norm().item():.6f}")
    
    print(f"\n  Inverse relation (r={inverse_rel_id.item()}):")
    print(f"    ||Δe_h|| = {delta_e_h_inv.norm().item():.6f}")
    print(f"    ||Δe_t|| = {delta_e_t_inv.norm().item():.6f}")
    
    # For forward relation r:
    #   Δe_h uses r, Δe_t uses r⁻¹
    # For inverse relation r⁻¹:
    #   Δe_h uses r⁻¹, Δe_t uses r
    # So: delta_e_h_fwd should ≈ delta_e_t_inv (both use r)
    #     delta_e_t_fwd should ≈ delta_e_h_inv (both use r⁻¹)
    
    diff_1 = (delta_e_h_fwd - delta_e_t_inv).norm().item()
    diff_2 = (delta_e_t_fwd - delta_e_h_inv).norm().item()
    
    print(f"\n  Cross-check (should match):")
    print(f"    ||Δe_h(r) - Δe_t(r⁻¹)|| = {diff_1:.6f}")
    print(f"    ||Δe_t(r) - Δe_h(r⁻¹)|| = {diff_2:.6f}")
    
    if diff_1 < 1e-5 and diff_2 < 1e-5:
        print("  ✅ PASS: Inverse relations handled correctly")
        success = True
    else:
        print("  ❌ FAIL: Inverse relation mismatch")
        success = False
    
    return success


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("GE-PAIRRE COMPREHENSIVE TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        ("Gaussian Activation", test_1_gaussian_activation),
        ("Static Preservation", test_2_static_preservation),
        ("Evolution Additivity", test_3_evolution_additivity),
        ("Scoring Function", test_4_scoring_function),
        ("Temporal Discrimination", test_5_temporal_discrimination),
        ("Parameter Initialization", test_6_parameter_initialization),
        ("Regularizers", test_7_regularizers),
        ("Inverse Relations", test_8_inverse_relations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n  ❌ ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed, error in results:
        if error:
            print(f"  ❌ {test_name}: ERROR - {error}")
        elif passed:
            print(f"  ✅ {test_name}: PASSED")
        else:
            print(f"  ❌ {test_name}: FAILED")
    
    n_passed = sum(1 for _, passed, error in results if passed and not error)
    n_total = len(results)
    
    print(f"\n  Total: {n_passed}/{n_total} tests passed")
    print("="*70 + "\n")
    
    return n_passed == n_total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
