"""
Debug script to check if layer normalization is actually working
and if time embeddings are centered at zero.
"""
import torch
import numpy as np
from models import ContinuousPairRE
from datasets import TemporalDataset
import argparse

def debug_time_encoder():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ICEWS14')
    args = parser.parse_args()
    
    print("="*80)
    print("DEBUG: Time Encoder Layer Normalization Check")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load dataset
    dataset = TemporalDataset(args.dataset)
    
    # Infer parameters
    rank = state_dict['entity_embeddings.weight'].shape[1]
    K = state_dict['time_encoder.omega'].shape[0]
    beta = state_dict['beta'].item() if 'beta' in state_dict else 0.5
    
    # Create model
    model = ContinuousPairRE(
        dataset.get_shape(),
        rank,
        init_size=0.1,
        K=K,
        beta=beta
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nModel parameters:")
    print(f"  Rank: {rank}")
    print(f"  K: {K}")
    print(f"  Beta: {beta}")
    
    # Get validation data
    val_examples = dataset.get_examples('valid')
    n_samples = 500
    samples = val_examples[:n_samples].astype('int64')
    
    # Convert to continuous time
    if dataset.use_continuous_time and dataset.ts_normalized is not None:
        samples_continuous = np.copy(samples).astype('float32')
        timestamp_ids = samples[:, 3].astype('int32')
        continuous_times = np.array([dataset.ts_normalized[int(tid)] for tid in timestamp_ids], dtype=np.float32)
        samples_continuous[:, 3] = continuous_times
        samples = samples_continuous
    
    samples = torch.from_numpy(samples).to(device)
    
    print(f"\nAnalyzing {n_samples} samples...")
    
    with torch.no_grad():
        rel_ids = samples[:, 1].long()
        taus = samples[:, 3].float()
        
        # Call time encoder
        m = model.time_encoder(rel_ids, taus)
        
        # Compute gate
        gate = model.residual_gate(m)
        
        print("\n" + "="*80)
        print("TIME EMBEDDING STATISTICS (AFTER FORWARD PASS)")
        print("="*80)
        
        m_mean = m.mean().item()
        m_std = m.std().item()
        m_min = m.min().item()
        m_max = m.max().item()
        
        print(f"m_mean: {m_mean:.6f}")
        print(f"m_std:  {m_std:.6f}")
        print(f"m_min:  {m_min:.6f}")
        print(f"m_max:  {m_max:.6f}")
        
        print("\n" + "="*80)
        print("GATE STATISTICS")
        print("="*80)
        
        gate_mean = gate.mean().item()
        gate_std = gate.std().item()
        gate_min = gate.min().item()
        gate_max = gate.max().item()
        
        print(f"gate_mean: {gate_mean:.6f}")
        print(f"gate_std:  {gate_std:.6f}")
        print(f"gate_min:  {gate_min:.6f}")
        print(f"gate_max:  {gate_max:.6f}")
        
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        if abs(m_mean) < 0.01:
            print("✅ GOOD: Time embeddings are centered (m_mean ≈ 0)")
            print("   → Layer normalization is working!")
        elif abs(m_mean) < 0.1:
            print("⚠️  WARNING: Time embeddings have small bias (|m_mean| < 0.1)")
            print(f"   → m_mean = {m_mean:.6f}")
        else:
            print("❌ PROBLEM: Time embeddings have large bias!")
            print(f"   → m_mean = {m_mean:.6f} (should be ≈ 0)")
            print("   → Layer normalization NOT working or NOT implemented!")
        
        if abs(gate_mean - 1.0) < 0.05:
            print("✅ GOOD: Gate centered at 1.0")
        else:
            print(f"⚠️  WARNING: Gate mean is {gate_mean:.4f} (should be ≈ 1.0)")
        
        print("\n" + "="*80)
        print("CHECK MODEL SOURCE CODE")
        print("="*80)
        
        # Try to check if normalization code exists
        import inspect
        forward_source = inspect.getsource(model.time_encoder.forward)
        
        if 'm.mean(' in forward_source or 'layer_norm' in forward_source.lower():
            print("✅ Layer normalization code FOUND in time_encoder.forward()")
            print("\nRelevant code snippet:")
            lines = forward_source.split('\n')
            for i, line in enumerate(lines):
                if 'm.mean(' in line or 'layer_norm' in line.lower():
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        print(f"{marker}{lines[j]}")
                    break
        else:
            print("❌ Layer normalization code NOT FOUND in time_encoder.forward()!")
            print("   → Did you forget to apply the fix to models.py?")
            print("   → Or is the checkpoint from before the fix?")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if abs(m_mean) > 0.1:
            print("1. ❌ Layer normalization NOT working")
            print("   → Check if you trained with the FIXED models.py")
            print("   → The checkpoint might be from BEFORE the fix")
            print("   → Re-train with updated models.py that has:")
            print("      m = m - m.mean(dim=0, keepdim=True)")
            print("")
            print("2. If you DID train with fixed code, check for bugs:")
            print("   → Maybe normalization applied incorrectly")
            print("   → Maybe batch size=1 causing mean=single value")
            print("   → Maybe keepdim=False instead of True")
        else:
            print("✅ Layer normalization working correctly")
            print("   → But time still not learning!")
            print("   → Problem is elsewhere:")
            print("      1. Time discrimination loss weight still too low")
            print("      2. Architecture issue (gate mechanism insufficient)")
            print("      3. Need different approach (e.g., additive instead of multiplicative)")

if __name__ == '__main__':
    debug_time_encoder()
