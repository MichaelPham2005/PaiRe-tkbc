"""
Quick diagnostic to check if gate values are actually varying
and if they're impacting scores meaningfully.
"""
import torch
import numpy as np
from models import ContinuousPairRE
from datasets import TemporalDataset
import argparse

def check_gate_impact():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ICEWS14')
    args = parser.parse_args()
    
    print("Loading model and data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load dataset
    dataset = TemporalDataset(args.dataset)
    
    # Extract model parameters from checkpoint
    # The checkpoint may have different structures
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Infer rank from entity embeddings
        rank = state_dict['entity_embeddings.weight'].shape[1]
        K_frequencies = state_dict['time_encoder.omega'].shape[0] if 'time_encoder.omega' in state_dict else 16
        beta = state_dict['beta'].item() if 'beta' in state_dict else 0.5
    else:
        # Checkpoint is the state dict itself
        state_dict = checkpoint
        rank = state_dict['entity_embeddings.weight'].shape[1]
        K_frequencies = state_dict['time_encoder.omega'].shape[0] if 'time_encoder.omega' in state_dict else 16
        beta = state_dict['beta'].item() if 'beta' in state_dict else 0.5
    
    # Create model
    model = ContinuousPairRE(
        dataset.get_shape(),
        rank,
        init_size=0.1,
        K=K_frequencies,
        beta=beta
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nModel config:")
    print(f"  Beta: {model.beta}")
    print(f"  K_frequencies: {model.time_encoder.K}")
    print(f"  Rank: {rank}")
    
    # Get validation data - use correct method
    val_examples = dataset.get_examples('valid')
    n_samples = min(100, len(val_examples))
    samples = val_examples[:n_samples].astype('int64')
    
    # Convert to continuous time if needed
    if dataset.use_continuous_time and dataset.ts_normalized is not None:
        samples_continuous = np.copy(samples).astype('float32')
        timestamp_ids = samples[:, 3].astype('int32')
        continuous_times = np.array([dataset.ts_normalized[int(tid)] for tid in timestamp_ids], dtype=np.float32)
        samples_continuous[:, 3] = continuous_times
        samples = samples_continuous
    
    samples = torch.from_numpy(samples).to(device)
    
    print(f"\nAnalyzing {n_samples} samples...")
    
    with torch.no_grad():
        # Get time embeddings and gates
        rel_ids = samples[:, 1].long()
        taus = samples[:, 3].float()
        
        m = model.time_encoder(rel_ids, taus)  # Time embeddings
        gate = model.residual_gate(m)  # Gate values
        
        # Get entity embeddings for scoring
        h = model.entity_embeddings(samples[:, 0].long())
        r_h = model.relation_head(samples[:, 1].long())
        t = model.entity_embeddings(samples[:, 2].long())
        r_t = model.relation_tail(samples[:, 1].long())
        
        # Compute interaction WITHOUT gate
        interaction_no_gate = h * r_h - t * r_t
        score_no_gate = -torch.norm(interaction_no_gate, p=1, dim=-1)
        
        # Compute interaction WITH gate
        interaction_with_gate = interaction_no_gate * gate
        score_with_gate = -torch.norm(interaction_with_gate, p=1, dim=-1)
        
        # Analyze gate statistics
        gate_mean = gate.mean().item()
        gate_std = gate.std().item()
        gate_min = gate.min().item()
        gate_max = gate.max().item()
        
        # Analyze score difference
        score_diff = (score_with_gate - score_no_gate).abs()
        score_diff_mean = score_diff.mean().item()
        score_diff_max = score_diff.max().item()
        
        # Analyze relative change
        relative_change = (score_diff / (score_no_gate.abs() + 1e-8)).abs()
        relative_change_mean = relative_change.mean().item()
        
        print("\n" + "="*80)
        print("GATE STATISTICS")
        print("="*80)
        print(f"Gate values (g = 1 + β×tanh(m)):")
        print(f"  Mean:  {gate_mean:.4f}")
        print(f"  Std:   {gate_std:.4f}")
        print(f"  Min:   {gate_min:.4f}")
        print(f"  Max:   {gate_max:.4f}")
        print(f"  Range: [{gate_min:.4f}, {gate_max:.4f}]")
        
        print("\n" + "="*80)
        print("TIME EMBEDDING (m) STATISTICS")
        print("="*80)
        m_mean = m.mean().item()
        m_std = m.std().item()
        m_min = m.min().item()
        m_max = m.max().item()
        print(f"  Mean:  {m_mean:.4f}")
        print(f"  Std:   {m_std:.4f}")
        print(f"  Min:   {m_min:.4f}")
        print(f"  Max:   {m_max:.4f}")
        
        print("\n" + "="*80)
        print("SCORE IMPACT ANALYSIS")
        print("="*80)
        print(f"Score magnitude (no gate):")
        print(f"  Mean: {score_no_gate.mean().item():.4f}")
        print(f"  Std:  {score_no_gate.std().item():.4f}")
        
        print(f"\nScore magnitude (with gate):")
        print(f"  Mean: {score_with_gate.mean().item():.4f}")
        print(f"  Std:  {score_with_gate.std().item():.4f}")
        
        print(f"\nAbsolute score difference:")
        print(f"  Mean:  {score_diff_mean:.4f}")
        print(f"  Max:   {score_diff_max:.4f}")
        
        print(f"\nRelative score change:")
        print(f"  Mean:  {relative_change_mean*100:.2f}%")
        
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        if gate_std < 0.01:
            print("❌ PROBLEM: Gate values barely vary (std < 0.01)")
            print("   → Time embeddings collapsed or beta too small")
        elif gate_std < 0.05:
            print("⚠️  WARNING: Gate variation weak (std < 0.05)")
            print("   → Consider increasing beta")
        else:
            print("✅ GOOD: Gate values vary significantly")
        
        if relative_change_mean < 0.01:
            print("❌ PROBLEM: Gate changes scores by <1% on average")
            print("   → Time has negligible impact on predictions")
        elif relative_change_mean < 0.05:
            print("⚠️  WARNING: Gate changes scores by <5% on average")
            print("   → Time impact weak but present")
        else:
            print("✅ GOOD: Gate has significant impact on scores")
        
        if gate_mean < 0.9 or gate_mean > 1.1:
            print(f"⚠️  INFO: Gate mean is {gate_mean:.2f} (not centered at 1.0)")
            print("   → Time embeddings have non-zero mean")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if gate_std < 0.05 or relative_change_mean < 0.05:
            current_beta = model.beta
            print(f"Current beta: {current_beta}")
            print(f"Recommended: Increase beta to {current_beta * 2:.1f} or {current_beta * 3:.1f}")
            print("This will amplify the gate effect:")
            print(f"  Current range: [{gate_min:.2f}, {gate_max:.2f}]")
            print(f"  With 2x beta:  [{1 + 2*current_beta*(gate_min-1):.2f}, {1 + 2*current_beta*(gate_max-1):.2f}]")
            print(f"  With 3x beta:  [{1 + 3*current_beta*(gate_min-1):.2f}, {1 + 3*current_beta*(gate_max-1):.2f}]")
        else:
            print("Gate strength appears adequate.")
            print("If time still not learning, check:")
            print("  1. Time discrimination weight (try 1.0-2.0)")
            print("  2. Time parameter regularization (try reducing to 0.00001)")
            print("  3. Learning rate for time parameters")

if __name__ == '__main__':
    check_gate_impact()
