#!/usr/bin/env python3
# Test script for ContinuousPairRE model

import torch
import numpy as np
from datasets import TemporalDataset
from models import ContinuousPairRE

def test_continuous_pairre():
    """Test the ContinuousPairRE model on ICEWS14 dataset."""
    
    print("="*60)
    print("Testing ContinuousPairRE Implementation")
    print("="*60)
    
    # Load dataset with continuous time
    print("\n1. Loading ICEWS14 dataset with continuous time...")
    dataset = TemporalDataset('ICEWS14', use_continuous_time=True)
    
    print(f"   - Entities: {dataset.n_entities}")
    print(f"   - Relations: {dataset.n_predicates}")
    print(f"   - Timestamps: {dataset.n_timestamps}")
    print(f"   - Continuous time loaded: {dataset.ts_normalized is not None}")
    
    # Get dataset shape
    sizes = dataset.get_shape()
    print(f"   - Dataset shape: {sizes}")
    
    # Initialize model
    print("\n2. Initializing ContinuousPairRE model...")
    rank = 50
    model = ContinuousPairRE(sizes, rank)
    model = model.cuda()
    print(f"   - Rank: {rank}")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass with a small batch
    print("\n3. Testing forward pass...")
    train_data = dataset.get_train()
    batch_size = 10
    batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
    batch = train_data[batch_indices]
    
    # Convert to continuous time
    batch_continuous = batch.copy().astype(np.float32)
    for i in range(batch_size):
        ts_id = int(batch[i, 3])
        batch_continuous[i, 3] = dataset.ts_normalized[ts_id]
    
    batch_tensor = torch.from_numpy(batch_continuous).cuda()
    print(f"   - Batch shape: {batch_tensor.shape}")
    print(f"   - Sample timestamps (discrete): {batch[:3, 3]}")
    print(f"   - Sample timestamps (continuous): {batch_continuous[:3, 3]}")
    
    # Forward pass
    with torch.no_grad():
        scores, factors, time_embeddings = model.forward(batch_tensor)
    
    print(f"   - Output scores shape: {scores.shape}")
    print(f"   - Score range: [{scores.min().item():.2f}, {scores.max().item():.2f}]")
    factor_shapes = [f.shape for f in factors]
    factor_norms = [f.norm().item() for f in factors]
    print(f"   - Regularization factors shapes: {factor_shapes}")
    print(f"   - Regularization factors norms: {factor_norms}")
    if time_embeddings is not None:
        print(f"   - Time embeddings shape: {time_embeddings.shape}")
        print(f"   - Time embeddings norm: {time_embeddings.norm().item():.4f}")
    
    # Test score function
    print("\n4. Testing score function...")
    with torch.no_grad():
        point_scores = model.score(batch_tensor)
    print(f"   - Point scores shape: {point_scores.shape}")
    print(f"   - Point scores sample: {point_scores[:5].flatten().tolist()}")
    
    # Test time embedding
    print("\n5. Testing continuous time embedding...")
    test_times = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0]).cuda()
    with torch.no_grad():
        time_embeds = model.time_encoder(test_times)
    print(f"   - Time embeddings shape: {time_embeds.shape}")
    print(f"   - Time embedding sample at t=0: {time_embeds[0, :5].tolist()}")
    print(f"   - Time embedding sample at t=100: {time_embeds[-1, :5].tolist()}")
    
    # Test gating parameters
    print("\n6. Analyzing temporal gating...")
    with torch.no_grad():
        alphas = torch.sigmoid(model.alpha.weight)
    print(f"   - Alpha (gating) shape: {alphas.shape}")
    print(f"   - Alpha statistics:")
    print(f"     * Mean: {alphas.mean().item():.3f}")
    print(f"     * Std: {alphas.std().item():.3f}")
    print(f"     * Min: {alphas.min().item():.3f}")
    print(f"     * Max: {alphas.max().item():.3f}")
    print(f"   - Sample alpha values: {alphas[:5].flatten().tolist()}")
    
    # Analyze which relations are more static vs dynamic
    print("\n7. Relation temporal analysis...")
    alpha_values = alphas.cpu().numpy().flatten()
    static_relations = np.where(alpha_values < 0.3)[0]
    dynamic_relations = np.where(alpha_values > 0.7)[0]
    print(f"   - Static relations (alpha < 0.3): {len(static_relations)} / {len(alpha_values)}")
    print(f"   - Dynamic relations (alpha > 0.7): {len(dynamic_relations)} / {len(alpha_values)}")
    print(f"   - Mixed relations: {len(alpha_values) - len(static_relations) - len(dynamic_relations)}")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
    
if __name__ == "__main__":
    test_continuous_pairre()
