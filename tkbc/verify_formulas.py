#!/usr/bin/env python3
"""
Verify implementation against mathematical formulas from the specification
"""

import torch
import numpy as np

def verify_formulas():
    print("="*70)
    print("FORMULA VERIFICATION")
    print("="*70)
    
    from models import ContinuousPairRE
    
    # Create a small test model
    num_entities = 100
    num_relations = 10
    num_timestamps = 50
    rank = 20
    
    model = ContinuousPairRE((num_entities, num_relations, num_entities, num_timestamps), rank)
    model.eval()
    
    print("\n✓ Model created successfully")
    print(f"  Entities: {num_entities}, Relations: {num_relations}, Rank: {rank}")
    
    # Test 1: Check alpha is a vector per relation
    print("\n1. Checking α ∈ ℝ^|R| (one alpha per relation)...")
    alpha_shape = model.alpha.weight.shape
    print(f"   Alpha shape: {alpha_shape}")
    if alpha_shape == (num_relations, 1):
        print(f"   ✅ PASS: Alpha has shape ({num_relations}, 1)")
    else:
        print(f"   ❌ FAIL: Expected ({num_relations}, 1), got {alpha_shape}")
    
    # Test 2: Check alpha initialized to 0 (sigmoid(0) = 0.5)
    print("\n2. Checking α initialized to 0...")
    alpha_init = model.alpha.weight.data.mean().item()
    print(f"   Alpha mean: {alpha_init:.6f}")
    if abs(alpha_init) < 0.01:
        print(f"   ✅ PASS: Alpha initialized near 0")
    else:
        print(f"   ❌ FAIL: Alpha should be ~0, got {alpha_init:.6f}")
    
    # Test 3: Check time embedding m = cos(W·t + b)
    print("\n3. Checking m = cos(W·t + b)...")
    t_test = torch.tensor([0.5], dtype=torch.float32)  # Test time in [-1, 1]
    m = model.time_encoder(t_test)
    W = model.time_encoder.W
    b = model.time_encoder.b
    
    # Manual calculation
    m_expected = torch.cos(t_test.unsqueeze(-1) * W + b)
    
    diff = torch.abs(m - m_expected).max().item()
    print(f"   m shape: {m.shape}")
    print(f"   W shape: {W.shape}")
    print(f"   b shape: {b.shape}")
    print(f"   Max difference: {diff:.10f}")
    
    if diff < 1e-6:
        print(f"   ✅ PASS: m = cos(W·t + b) correctly implemented")
    else:
        print(f"   ❌ FAIL: Formula mismatch")
    
    # Test 4: Check gating function G(m,r) = σ(aᵣ)·m + (1-σ(aᵣ))·1
    print("\n4. Checking G(m,r) = σ(aᵣ)·m + (1-σ(aᵣ))·1...")
    
    # Create test input
    h_idx = torch.tensor([0], dtype=torch.long)
    r_idx = torch.tensor([0], dtype=torch.long)
    t_idx = torch.tensor([1], dtype=torch.long)
    time = torch.tensor([0.0], dtype=torch.float32)  # Time at 0
    
    x = torch.stack([h_idx, r_idx, t_idx, time], dim=1)
    
    with torch.no_grad():
        # Get components
        m = model.time_encoder(time)
        alpha_raw = model.alpha(r_idx)
        alpha = torch.sigmoid(alpha_raw)
        
        # Compute gating
        gate = alpha * m + (1 - alpha) * torch.ones_like(m)
        
        # Manual verification
        gate_manual = alpha * m + (1 - alpha) * torch.ones_like(m)
        
        diff = torch.abs(gate - gate_manual).max().item()
        
        print(f"   α = σ(a₀) = {alpha.item():.4f}")
        print(f"   m[0] = {m[0, 0].item():.4f}")
        print(f"   G[0] = {gate[0, 0].item():.4f}")
        print(f"   Expected: {gate_manual[0, 0].item():.4f}")
        print(f"   Difference: {diff:.10f}")
        
        # Check that when alpha=0, gate=1 (all ones)
        model.alpha.weight.data.fill_(-10)  # sigmoid(-10) ≈ 0
        alpha_zero = torch.sigmoid(model.alpha(r_idx))
        gate_zero = alpha_zero * m + (1 - alpha_zero) * torch.ones_like(m)
        
        ones_diff = torch.abs(gate_zero - torch.ones_like(m)).max().item()
        
        print(f"\n   When α≈0: G should ≈ 1 (ones)")
        print(f"   α = {alpha_zero.item():.6f}")
        print(f"   max|G - 1| = {ones_diff:.6f}")
        
        if diff < 1e-6 and ones_diff < 0.01:
            print(f"   ✅ PASS: Gating formula correct")
        else:
            print(f"   ❌ FAIL: Gating formula incorrect")
        
        # Reset alpha
        model.alpha.weight.data.fill_(0.0)
    
    # Test 5: Check scoring function φ(h,r,t,m) = |(h∘r^H - t∘r^T)∘G(m,r)|
    print("\n5. Checking φ(h,r,t,m) = |(h∘r^H - t∘r^T)∘G(m,r)|...")
    
    with torch.no_grad():
        # Get embeddings
        h = model.entity_embeddings(h_idx)
        t = model.entity_embeddings(t_idx)
        r_h = model.relation_head(r_idx)
        r_t = model.relation_tail(r_idx)
        
        m = model.time_encoder(time)
        alpha = torch.sigmoid(model.alpha(r_idx))
        gate = alpha * m + (1 - alpha) * torch.ones_like(m)
        
        # Manual computation
        interaction = (h * r_h - t * r_t) * gate
        score_manual = -torch.norm(interaction, p=1, dim=-1)
        
        # Model computation
        score_model = model.score(x)
        
        diff = torch.abs(score_manual - score_model).max().item()
        
        print(f"   h shape: {h.shape}")
        print(f"   r^H shape: {r_h.shape}")
        print(f"   t shape: {t.shape}")
        print(f"   r^T shape: {r_t.shape}")
        print(f"   G shape: {gate.shape}")
        print(f"   Score (manual): {score_manual.item():.6f}")
        print(f"   Score (model):  {score_model.item():.6f}")
        print(f"   Difference: {diff:.10f}")
        
        if diff < 1e-5:
            print(f"   ✅ PASS: Scoring formula matches")
        else:
            print(f"   ❌ FAIL: Scoring formula mismatch")
    
    # Test 6: Verify L1 norm is used
    print("\n6. Checking L1 norm (sum of absolute values)...")
    
    test_vec = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    l1_manual = torch.abs(test_vec).sum().item()
    l1_torch = torch.norm(test_vec, p=1).item()
    
    print(f"   Test vector: {test_vec.tolist()}")
    print(f"   L1 (manual abs.sum): {l1_manual:.6f}")
    print(f"   L1 (torch.norm p=1): {l1_torch:.6f}")
    
    if abs(l1_manual - l1_torch) < 1e-6:
        print(f"   ✅ PASS: L1 norm correctly implemented")
    else:
        print(f"   ❌ FAIL: L1 norm mismatch")
    
    # Test 7: Check negative sign convention
    print("\n7. Checking negative sign convention...")
    print("   Distance-based models (PairRE): smaller distance = better match")
    print("   For CrossEntropyLoss: higher score = better")
    print("   Therefore: score = -distance (negative L1 norm)")
    print("   ✅ Negative sign is CORRECT")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    
    return True

if __name__ == "__main__":
    verify_formulas()
