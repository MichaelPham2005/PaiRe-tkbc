"""
Verification tests for Relation-Conditioned Continuous Time Model.
Tests that the new model implementation matches the spec in readme.

Run: python verify_relation_conditioned_model.py
"""

import sys
sys.path.append('external/tkbc')

import torch
import numpy as np
from models import ContinuousPairRE, RelationConditionedTimeEncoder

def test_relation_conditioned_time_encoder():
    """Test 1: RelationConditionedTimeEncoder structure and outputs."""
    print("\n" + "="*70)
    print("TEST 1: RelationConditionedTimeEncoder")
    print("="*70)
    
    n_relations = 10
    dim = 64
    K = 8
    batch_size = 5
    
    encoder = RelationConditionedTimeEncoder(n_relations, dim, K)
    
    # Check parameters exist
    assert hasattr(encoder, 'a_r'), "Missing trend parameter a_r"
    assert hasattr(encoder, 'A_r'), "Missing amplitude parameter A_r"
    assert hasattr(encoder, 'P_r'), "Missing phase parameter P_r"
    assert hasattr(encoder, 'omega'), "Missing global frequency omega"
    assert hasattr(encoder, 'W_proj'), "Missing projection weight W_proj"
    assert hasattr(encoder, 'b_proj'), "Missing projection bias b_proj"
    
    # Check shapes
    assert encoder.a_r.shape == (n_relations,), f"a_r shape {encoder.a_r.shape} != ({n_relations},)"
    assert encoder.A_r.shape == (n_relations, K), f"A_r shape {encoder.A_r.shape} != ({n_relations}, {K})"
    assert encoder.P_r.shape == (n_relations, K), f"P_r shape {encoder.P_r.shape} != ({n_relations}, {K})"
    assert encoder.omega.shape == (K,), f"omega shape {encoder.omega.shape} != ({K},)"
    assert encoder.W_proj.shape == (dim, 1+K), f"W_proj shape {encoder.W_proj.shape} != ({dim}, {1+K})"
    assert encoder.b_proj.shape == (dim,), f"b_proj shape {encoder.b_proj.shape} != ({dim},)"
    
    print(f"✓ All parameters present with correct shapes")
    
    # Test forward pass
    rel_ids = torch.randint(0, n_relations, (batch_size,))
    taus = torch.rand(batch_size)  # Random times in [0, 1]
    
    m = encoder(rel_ids, taus)
    
    assert m.shape == (batch_size, dim), f"Output shape {m.shape} != ({batch_size}, {dim})"
    assert torch.all(torch.abs(m) <= 1.0), "Output not in tanh range [-1, 1]"
    
    print(f"✓ Forward pass produces correct output shape: {m.shape}")
    print(f"✓ Output in tanh range: [{m.min():.4f}, {m.max():.4f}]")
    
    # Test that different times produce different outputs (temporal variation)
    tau1 = torch.zeros(batch_size)
    tau2 = torch.ones(batch_size)
    rel_ids_same = torch.zeros(batch_size, dtype=torch.long)
    
    m1 = encoder(rel_ids_same, tau1)
    m2 = encoder(rel_ids_same, tau2)
    
    diff = torch.norm(m1 - m2, dim=1).mean()
    assert diff > 0.01, f"Time embeddings don't vary with time: diff = {diff:.6f}"
    print(f"✓ Temporal variation confirmed: ||m(0) - m(1)|| = {diff:.4f}")
    
    # Test relation-specific patterns
    tau_same = torch.ones(batch_size) * 0.5
    rel_ids_diff = torch.arange(batch_size)
    
    m_diff = encoder(rel_ids_diff, tau_same)
    
    # Different relations should produce different embeddings
    for i in range(batch_size-1):
        diff_rel = torch.norm(m_diff[i] - m_diff[i+1])
        assert diff_rel > 0.001, f"Relations {i} and {i+1} produce same embedding"
    
    print(f"✓ Relation-specific embeddings confirmed")
    
    print("\n✅ TEST 1 PASSED\n")
    return True


def test_residual_gate():
    """Test 2: Residual gate properties."""
    print("\n" + "="*70)
    print("TEST 2: Residual Gate")
    print("="*70)
    
    # Test gate function
    n_entities = 100
    n_relations = 20
    n_times = 50
    rank = 64
    K = 8
    beta = 0.5  # Updated default
    
    sizes = (n_entities, n_relations, n_entities, n_times)
    model = ContinuousPairRE(sizes, rank, K=K, beta=beta)
    
    # Check beta parameter
    assert hasattr(model, 'beta'), "Missing beta parameter"
    assert model.beta == beta, f"Beta {model.beta} != {beta}"
    print(f"✓ Beta parameter: {model.beta}")
    
    # Test residual_gate method
    assert hasattr(model, 'residual_gate'), "Missing residual_gate method"
    
    # Test gate values
    batch_size = 10
    m = torch.randn(batch_size, rank)
    gate = model.residual_gate(m)
    
    assert gate.shape == (batch_size, rank), f"Gate shape {gate.shape} != ({batch_size}, {rank})"
    
    # Gate should always be positive if beta < 1
    # g = 1 + beta * tanh(m), tanh ∈ [-1, 1]
    # Min: 1 - beta = 1 - 0.2 = 0.8
    # Max: 1 + beta = 1 + 0.2 = 1.2
    min_gate = gate.min().item()
    max_gate = gate.max().item()
    
    assert min_gate >= (1 - beta) - 0.01, f"Gate min {min_gate} < {1-beta}"
    assert max_gate <= (1 + beta) + 0.01, f"Gate max {max_gate} > {1+beta}"
    assert torch.all(gate > 0), "Gate contains non-positive values!"
    
    print(f"✓ Gate range: [{min_gate:.4f}, {max_gate:.4f}] (expected [{1-beta:.1f}, {1+beta:.1f}])")
    print(f"✓ All gate values > 0")
    
    # Test that gate varies with input
    m1 = torch.zeros(batch_size, rank)
    m2 = torch.ones(batch_size, rank) * 2
    
    gate1 = model.residual_gate(m1)
    gate2 = model.residual_gate(m2)
    
    # gate1 should be close to 1 (tanh(0) = 0)
    assert torch.allclose(gate1, torch.ones_like(gate1), atol=1e-6), "Gate(0) should be 1"
    
    # gate2 should be different
    diff = torch.norm(gate1 - gate2).item()
    assert diff > 0.1, f"Gate doesn't vary: diff = {diff}"
    
    print(f"✓ Gate(0) ≈ 1.0: {gate1[0, 0]:.6f}")
    print(f"✓ Gate varies with input: ||gate(0) - gate(2)|| = {diff:.4f}")
    
    print("\n✅ TEST 2 PASSED\n")
    return True


def test_no_alpha_parameters():
    """Test 3: Verify no alpha gating parameters exist."""
    print("\n" + "="*70)
    print("TEST 3: No Alpha Parameters")
    print("="*70)
    
    sizes = (100, 20, 100, 50)
    model = ContinuousPairRE(sizes, rank=64, K=8, beta=0.5)
    
    # Check that alpha embedding does NOT exist
    assert not hasattr(model, 'alpha'), "ERROR: Model still has alpha parameter!"
    
    print(f"✓ No 'alpha' attribute in model")
    
    # Check all parameters
    param_names = [name for name, _ in model.named_parameters()]
    alpha_params = [name for name in param_names if 'alpha' in name.lower()]
    
    assert len(alpha_params) == 0, f"ERROR: Found alpha-related parameters: {alpha_params}"
    
    print(f"✓ No alpha-related parameters found")
    print(f"✓ Total parameters: {len(param_names)}")
    
    # List time encoder parameters
    time_params = [name for name in param_names if 'time_encoder' in name]
    print(f"\nTime encoder parameters:")
    for name in time_params:
        param = dict(model.named_parameters())[name]
        print(f"  {name}: {tuple(param.shape)}")
    
    print("\n✅ TEST 3 PASSED\n")
    return True


def test_score_varies_with_time():
    """Test 4: Scores change with different timestamps."""
    print("\n" + "="*70)
    print("TEST 4: Score Temporal Sensitivity")
    print("="*70)
    
    sizes = (100, 20, 100, 50)
    model = ContinuousPairRE(sizes, rank=64, K=16, beta=0.5)
    model.eval()
    
    batch_size = 20
    
    # Create test queries: [h, r, t, tau]
    h = torch.randint(0, sizes[0], (batch_size,))
    r = torch.randint(0, sizes[1], (batch_size,))
    t = torch.randint(0, sizes[2], (batch_size,))
    
    # Test with different times
    tau1 = torch.zeros(batch_size)  # t=0
    tau2 = torch.ones(batch_size) * 0.5  # t=0.5
    tau3 = torch.ones(batch_size)  # t=1
    
    queries1 = torch.stack([h.float(), r.float(), t.float(), tau1], dim=1)
    queries2 = torch.stack([h.float(), r.float(), t.float(), tau2], dim=1)
    queries3 = torch.stack([h.float(), r.float(), t.float(), tau3], dim=1)
    
    with torch.no_grad():
        scores1 = model.score(queries1)
        scores2 = model.score(queries2)
        scores3 = model.score(queries3)
    
    # Compute differences
    diff_12 = torch.abs(scores1 - scores2).mean().item()
    diff_23 = torch.abs(scores2 - scores3).mean().item()
    diff_13 = torch.abs(scores1 - scores3).mean().item()
    
    print(f"Mean |score(t=0) - score(t=0.5)|: {diff_12:.6f}")
    print(f"Mean |score(t=0.5) - score(t=1)|: {diff_23:.6f}")
    print(f"Mean |score(t=0) - score(t=1)|: {diff_13:.6f}")
    
    # Scores MUST change with time (no static shortcut)
    assert diff_12 > 0.001, f"Scores don't change enough with time: {diff_12}"
    assert diff_23 > 0.001, f"Scores don't change enough with time: {diff_23}"
    assert diff_13 > 0.001, f"Scores don't change enough with time: {diff_13}"
    
    print(f"✓ Scores vary significantly with time")
    
    # Test that score range is reasonable
    print(f"\nScore statistics:")
    print(f"  t=0:   mean={scores1.mean():.4f}, std={scores1.std():.4f}, range=[{scores1.min():.4f}, {scores1.max():.4f}]")
    print(f"  t=0.5: mean={scores2.mean():.4f}, std={scores2.std():.4f}, range=[{scores2.min():.4f}, {scores2.max():.4f}]")
    print(f"  t=1:   mean={scores3.mean():.4f}, std={scores3.std():.4f}, range=[{scores3.min():.4f}, {scores3.max():.4f}]")
    
    print("\n✅ TEST 4 PASSED\n")
    return True


def test_forward_pass():
    """Test 5: Full forward pass and output shapes."""
    print("\n" + "="*70)
    print("TEST 5: Forward Pass (1-vs-All)")
    print("="*70)
    
    n_entities = 100
    n_relations = 20
    n_times = 50
    rank = 64
    K = 16
    batch_size = 10
    
    sizes = (n_entities, n_relations, n_entities, n_times)
    model = ContinuousPairRE(sizes, rank, K=K, beta=0.5)
    model.eval()
    
    # Create batch
    h = torch.randint(0, n_entities, (batch_size,))
    r = torch.randint(0, n_relations, (batch_size,))
    t = torch.randint(0, n_entities, (batch_size,))
    tau = torch.rand(batch_size)
    
    queries = torch.stack([h.float(), r.float(), t.float(), tau], dim=1)
    
    with torch.no_grad():
        scores, factors, time_emb = model.forward(queries)
    
    # Check output shapes
    assert scores.shape == (batch_size, n_entities), f"Scores shape {scores.shape} != ({batch_size}, {n_entities})"
    assert len(factors) == 3, f"Expected 3 factors, got {len(factors)}"
    assert time_emb.shape == (batch_size, rank), f"Time embedding shape {time_emb.shape} != ({batch_size}, {rank})"
    
    print(f"✓ Scores shape: {scores.shape}")
    print(f"✓ Time embedding shape: {time_emb.shape}")
    print(f"✓ Factors: {len(factors)} tensors")
    
    # Check score statistics
    print(f"\nScore statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    
    # Scores should be negative (L1 norm with negative sign)
    assert scores.max() <= 0, "Scores should be negative (L1 distance)"
    
    print(f"✓ All scores are negative (L1-based)")
    
    print("\n✅ TEST 5 PASSED\n")
    return True


def test_time_discrimination_loss_gradient():
    """Test 6: Time discrimination loss produces gradients."""
    print("\n" + "="*70)
    print("TEST 6: Time Discrimination Loss Gradient")
    print("="*70)
    
    sizes = (50, 10, 50, 30)
    model = ContinuousPairRE(sizes, rank=32, K=8, beta=0.5)
    model.train()
    
    batch_size = 5
    
    # Create positive batch
    h = torch.randint(0, sizes[0], (batch_size,))
    r = torch.randint(0, sizes[1], (batch_size,))
    t = torch.randint(0, sizes[2], (batch_size,))
    tau_pos = torch.rand(batch_size)
    
    queries_pos = torch.stack([h.float(), r.float(), t.float(), tau_pos], dim=1)
    queries_pos.requires_grad = False
    
    # Create negative batch (wrong time)
    tau_neg = torch.rand(batch_size)
    queries_neg = torch.stack([h.float(), r.float(), t.float(), tau_neg], dim=1)
    queries_neg.requires_grad = False
    
    # Compute scores
    score_pos = model.score(queries_pos)
    score_neg = model.score(queries_neg)
    
    # Time discrimination loss
    loss_time_disc = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-9).mean()
    
    # Check gradient flow to time parameters
    loss_time_disc.backward()
    
    # Check that time encoder parameters have gradients
    time_params_with_grad = []
    for name, param in model.named_parameters():
        if 'time_encoder' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                time_params_with_grad.append((name, grad_norm))
    
    assert len(time_params_with_grad) > 0, "No gradients flowing to time encoder!"
    
    print(f"✓ Time discrimination loss: {loss_time_disc.item():.6f}")
    print(f"✓ Gradients flowing to {len(time_params_with_grad)} time parameters:")
    for name, grad_norm in time_params_with_grad:
        print(f"    {name}: {grad_norm:.6f}")
    
    # Check that entity/relation parameters also have gradients
    entity_grad = model.entity_embeddings.weight.grad
    assert entity_grad is not None and entity_grad.norm() > 0, "No gradients to entity embeddings"
    print(f"✓ Entity embeddings gradient norm: {entity_grad.norm():.6f}")
    
    print("\n✅ TEST 6 PASSED\n")
    return True


def test_comparison_with_spec():
    """Test 7: Compare with specification formulas."""
    print("\n" + "="*70)
    print("TEST 7: Comparison with Specification")
    print("="*70)
    
    n_relations = 5
    dim = 32
    K = 4
    beta = 0.3
    
    encoder = RelationConditionedTimeEncoder(n_relations, dim, K)
    
    # Manual computation for one sample
    rel_id = 2
    tau = 0.7
    
    # Get parameters
    a = encoder.a_r[rel_id].item()
    A = encoder.A_r[rel_id]  # (K,)
    P = encoder.P_r[rel_id]  # (K,)
    omega = encoder.omega  # (K,)
    
    # Compute trend: z_0 = a * tau
    z_trend_manual = a * tau
    
    # Compute periodic: z_k = A_k * sin(omega_k * tau + P_k)
    phase = omega * tau + P
    z_periodic_manual = A * torch.sin(phase)
    
    # Concat
    z_manual = torch.cat([torch.tensor([z_trend_manual]), z_periodic_manual])
    
    # Project
    m_manual = torch.tanh(z_manual @ encoder.W_proj.t() + encoder.b_proj)
    
    # Forward pass
    m_forward = encoder(torch.tensor([rel_id]), torch.tensor([tau]))
    
    # Compare
    diff = torch.norm(m_manual - m_forward[0]).item()
    
    print(f"Manual computation vs forward pass diff: {diff:.8f}")
    assert diff < 1e-5, f"Manual computation doesn't match forward: diff={diff}"
    
    print(f"✓ Manual formula matches forward pass")
    
    # Test residual gate formula
    sizes = (10, 5, 10, 10)
    model = ContinuousPairRE(sizes, rank=dim, K=K, beta=beta)
    
    m_test = torch.randn(3, dim)
    gate = model.residual_gate(m_test)
    gate_manual = 1.0 + beta * torch.tanh(m_test)
    
    diff_gate = torch.norm(gate - gate_manual).item()
    print(f"Gate formula diff: {diff_gate:.8f}")
    assert diff_gate < 1e-6, f"Gate formula doesn't match: diff={diff_gate}"
    
    print(f"✓ Residual gate formula correct: g = 1 + β·tanh(m)")
    
    print("\n✅ TEST 7 PASSED\n")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("RELATION-CONDITIONED CONTINUOUS TIME MODEL VERIFICATION")
    print("="*70)
    
    tests = [
        ("RelationConditionedTimeEncoder", test_relation_conditioned_time_encoder),
        ("Residual Gate", test_residual_gate),
        ("No Alpha Parameters", test_no_alpha_parameters),
        ("Score Temporal Sensitivity", test_score_varies_with_time),
        ("Forward Pass", test_forward_pass),
        ("Time Discrimination Loss", test_time_discrimination_loss_gradient),
        ("Specification Compliance", test_comparison_with_spec),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {name}")
            print(f"   Exception: {e}\n")
            failed += 1
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Model implementation is correct!")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Please review implementation")
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
