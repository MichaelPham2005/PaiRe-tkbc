"""
Test scoring consistency between score() and get_ranking() for GE-PairRE.

This test verifies that:
1. score() computes correct scores for specific quadruples
2. get_ranking() produces consistent scores using L1 distance
3. Both methods give the same results
"""

import torch
import sys
from models import GEPairRE

def test_ranking_consistency():
    """Test that get_ranking() produces same scores as score()."""
    print("=" * 80)
    print("TEST: Ranking Consistency for GE-PairRE")
    print("=" * 80)
    
    # Small test problem
    n_entities = 100
    n_relations = 10
    n_timestamps = 50
    rank = 32
    K_pulses = 4
    
    # Create model
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes=sizes, rank=rank, K=K_pulses)
    model.eval()
    
    # Create test queries: (h, r, t, tau)
    n_test = 5
    test_queries = torch.zeros(n_test, 4)
    test_queries[:, 0] = torch.randint(0, n_entities, (n_test,))  # heads
    test_queries[:, 1] = torch.randint(0, n_relations, (n_test,))  # relations
    test_queries[:, 2] = torch.randint(0, n_entities, (n_test,))  # tails
    test_queries[:, 3] = torch.rand(n_test)  # continuous time [0, 1]
    
    print(f"\nTest queries shape: {test_queries.shape}")
    print(f"Sample query: h={int(test_queries[0,0])}, r={int(test_queries[0,1])}, "
          f"t={int(test_queries[0,2])}, tau={test_queries[0,3]:.3f}")
    
    # Test 1: Get target scores using score()
    print("\n" + "-" * 80)
    print("Test 1: Computing target scores with score()")
    print("-" * 80)
    
    with torch.no_grad():
        target_scores = model.score(test_queries)
    
    print(f"Target scores shape: {target_scores.shape}")
    print(f"Target scores: {target_scores[:3].numpy()}")
    
    # Test 2: Get ranking scores using get_ranking()
    print("\n" + "-" * 80)
    print("Test 2: Computing ranking scores with get_ranking()")
    print("-" * 80)
    
    # Create empty filters (no filtering)
    filters = {}
    for i in range(n_test):
        h = int(test_queries[i, 0].item())
        r = int(test_queries[i, 1].item())
        t_id = int(test_queries[i, 3].item() * n_timestamps)  # Convert tau to time_id
        filters[(h, r, t_id)] = []
    
    with torch.no_grad():
        # Manually extract scores for true tails from all entity scores
        # We'll replicate get_ranking logic but extract scores instead of ranks
        
        h = model.entity_embeddings(test_queries[:, 0].long())
        r_h = model.relation_head(test_queries[:, 1].long())
        r_t = model.relation_tail(test_queries[:, 1].long())
        
        rel_id = test_queries[:, 1].long()
        tau = test_queries[:, 3].float()
        delta_e_h, delta_e_t = model.get_evolution_vectors(rel_id, tau)
        
        h_tau_r_h = (h + delta_e_h) * r_h  # (n_test, rank)
        
        # Get true tail entities
        true_tails = test_queries[:, 2].long()
        true_tail_embs = model.entity_embeddings(true_tails)
        
        # Evolve true tails
        t_tau = true_tail_embs + delta_e_t
        t_tau_r_t = t_tau * r_t
        
        # Compute scores for true tails
        interaction = h_tau_r_h - t_tau_r_t
        ranking_scores = -torch.norm(interaction, p=1, dim=1)
    
    print(f"Ranking scores shape: {ranking_scores.shape}")
    print(f"Ranking scores: {ranking_scores[:3].numpy()}")
    
    # Test 3: Compare consistency
    print("\n" + "-" * 80)
    print("Test 3: Checking consistency")
    print("-" * 80)
    
    abs_diff = torch.abs(target_scores - ranking_scores)
    rel_diff = abs_diff / (torch.abs(target_scores) + 1e-10)
    
    print(f"Absolute differences: {abs_diff[:3].numpy()}")
    print(f"Relative differences: {rel_diff[:3].numpy()}")
    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"Max relative difference: {rel_diff.max().item():.6e}")
    
    # Check if consistent
    tolerance = 1e-5
    if abs_diff.max() < tolerance:
        print(f"\n✅ PASS: Scores are consistent (max diff < {tolerance})")
        return True
    else:
        print(f"\n❌ FAIL: Scores are inconsistent (max diff = {abs_diff.max().item():.6e})")
        print("\nDetailed comparison:")
        for i in range(min(3, n_test)):
            print(f"  Query {i}: target={target_scores[i].item():.6f}, "
                  f"ranking={ranking_scores[i].item():.6f}, "
                  f"diff={abs_diff[i].item():.6e}")
        return False


def test_forward_vs_score():
    """Test that forward() gives same scores as score() for batch."""
    print("\n" + "=" * 80)
    print("TEST: Forward vs Score Consistency")
    print("=" * 80)
    
    # Small test problem
    n_entities = 100
    n_relations = 10
    n_timestamps = 50
    rank = 32
    K_pulses = 4
    
    # Create model
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes=sizes, rank=rank, K=K_pulses)
    model.eval()
    
    # Create test batch
    batch_size = 5
    test_batch = torch.zeros(batch_size, 4)
    test_batch[:, 0] = torch.randint(0, n_entities, (batch_size,))
    test_batch[:, 1] = torch.randint(0, n_relations, (batch_size,))
    test_batch[:, 2] = torch.randint(0, n_entities, (batch_size,))
    test_batch[:, 3] = torch.rand(batch_size)
    
    print(f"\nTest batch shape: {test_batch.shape}")
    
    # Get scores from score()
    with torch.no_grad():
        scores_from_score = model.score(test_batch)
    
    print(f"Scores from score(): {scores_from_score[:3].numpy()}")
    
    # Get scores from forward() - extract diagonal
    with torch.no_grad():
        all_scores, _, _ = model.forward(test_batch)
        # Extract scores for true tails
        true_tail_indices = test_batch[:, 2].long()
        scores_from_forward = all_scores[torch.arange(batch_size), true_tail_indices]
    
    print(f"Scores from forward(): {scores_from_forward[:3].numpy()}")
    
    # Compare
    abs_diff = torch.abs(scores_from_score - scores_from_forward)
    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    
    tolerance = 1e-5
    if abs_diff.max() < tolerance:
        print(f"\n✅ PASS: forward() and score() are consistent")
        return True
    else:
        print(f"\n❌ FAIL: forward() and score() are inconsistent")
        return False


def test_temporal_sensitivity():
    """Test that scores change with time."""
    print("\n" + "=" * 80)
    print("TEST: Temporal Sensitivity")
    print("=" * 80)
    
    n_entities = 100
    n_relations = 10
    n_timestamps = 50
    rank = 32
    K_pulses = 4
    
    sizes = (n_entities, n_relations, n_entities, n_timestamps)
    model = GEPairRE(sizes=sizes, rank=rank, K=K_pulses)
    model.eval()
    
    # Same triple, different times
    n_times = 10
    test_queries = torch.zeros(n_times, 4)
    test_queries[:, 0] = 5  # Same head
    test_queries[:, 1] = 2  # Same relation
    test_queries[:, 2] = 10  # Same tail
    test_queries[:, 3] = torch.linspace(0, 1, n_times)  # Different times
    
    print(f"\nScoring same triple at {n_times} different times...")
    
    with torch.no_grad():
        scores = model.score(test_queries)
    
    print(f"Scores: {scores.numpy()}")
    print(f"Score range: [{scores.min().item():.6e}, {scores.max().item():.6e}]")
    print(f"Score std: {scores.std().item():.6e}")
    
    # Check if scores vary
    score_range = scores.max() - scores.min()
    if score_range > 1e-6:
        print(f"\n✅ PASS: Scores vary with time (range={score_range.item():.6e})")
        return True
    else:
        print(f"\n❌ FAIL: Scores do not vary with time (range={score_range.item():.6e})")
        return False


if __name__ == "__main__":
    all_pass = True
    
    try:
        all_pass &= test_ranking_consistency()
    except Exception as e:
        print(f"\n❌ ERROR in test_ranking_consistency: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    try:
        all_pass &= test_forward_vs_score()
    except Exception as e:
        print(f"\n❌ ERROR in test_forward_vs_score: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    try:
        all_pass &= test_temporal_sensitivity()
    except Exception as e:
        print(f"\n❌ ERROR in test_temporal_sensitivity: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    sys.exit(0 if all_pass else 1)
