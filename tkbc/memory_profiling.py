"""
Memory profiling and estimation for GE-PairRE training.
Helps choose appropriate hyperparameters for available GPU memory.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import GEPairRE


def estimate_memory(n_entities, n_relations, rank, K_gaussians, batch_size):
    """
    Estimate GPU memory usage for GE-PairRE.
    
    Returns estimated memory in GB.
    """
    # Model parameters
    entity_emb = n_entities * rank * 4  # float32 = 4 bytes
    relation_head = n_relations * rank * 4
    relation_tail = n_relations * rank * 4
    gaussian_A = n_relations * K_gaussians * rank * 4
    gaussian_mu = n_relations * K_gaussians * 4
    gaussian_s = n_relations * K_gaussians * 4
    
    model_params = entity_emb + relation_head + relation_tail + gaussian_A + gaussian_mu + gaussian_s
    
    # Activations during forward pass (1-vs-All)
    # Main memory hog: scoring all entities
    h_tau = batch_size * rank * 4
    entity_chunk = 5000 * rank * 4  # Chunked processing
    t_tau_chunk = batch_size * 5000 * rank * 4
    interaction = batch_size * 5000 * rank * 4
    scores_chunk = batch_size * 5000 * 4
    
    forward_activations = h_tau + entity_chunk + t_tau_chunk + interaction + scores_chunk
    
    # Gradients (same size as parameters)
    gradients = model_params
    
    # Optimizer state (Adagrad: sum of squared gradients)
    optimizer_state = model_params
    
    # Total
    total_bytes = model_params + forward_activations + gradients + optimizer_state
    total_gb = total_bytes / (1024 ** 3)
    
    return {
        'model_params_gb': model_params / (1024 ** 3),
        'forward_activations_gb': forward_activations / (1024 ** 3),
        'gradients_gb': gradients / (1024 ** 3),
        'optimizer_state_gb': optimizer_state / (1024 ** 3),
        'total_gb': total_gb
    }


def print_memory_estimate(dataset_name, rank, K_gaussians, batch_size):
    """Print memory estimate for given configuration."""
    
    # Dataset sizes
    dataset_sizes = {
        'ICEWS14': (7128, 230, 7128, 365),
        'ICEWS05-15': (10488, 251, 10488, 4017),
    }
    
    if dataset_name not in dataset_sizes:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    sizes = dataset_sizes[dataset_name]
    n_entities, n_relations, _, n_timestamps = sizes
    
    mem = estimate_memory(n_entities, n_relations, rank, K_gaussians, batch_size)
    
    print("="*70)
    print(f"Memory Estimate for {dataset_name}")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Entities: {n_entities:,}")
    print(f"  - Relations: {n_relations}")
    print(f"  - Rank: {rank}")
    print(f"  - K Gaussians: {K_gaussians}")
    print(f"  - Batch size: {batch_size}")
    print()
    print(f"Estimated GPU Memory Usage:")
    print(f"  - Model parameters:     {mem['model_params_gb']:.2f} GB")
    print(f"  - Forward activations:  {mem['forward_activations_gb']:.2f} GB")
    print(f"  - Gradients:            {mem['gradients_gb']:.2f} GB")
    print(f"  - Optimizer state:      {mem['optimizer_state_gb']:.2f} GB")
    print(f"  - TOTAL (estimated):    {mem['total_gb']:.2f} GB")
    print()
    
    # Recommendations
    if mem['total_gb'] < 4:
        print("✅ Should work on 6GB GPUs (e.g., Colab free tier)")
    elif mem['total_gb'] < 8:
        print("✅ Should work on 8GB GPUs")
    elif mem['total_gb'] < 12:
        print("⚠️  Requires 12-16GB GPU")
    else:
        print("❌ Requires high-end GPU (16GB+)")
    
    print("="*70)
    print()


def recommend_config(available_memory_gb):
    """Recommend hyperparameters for given GPU memory."""
    print("="*70)
    print(f"Recommended Configurations for {available_memory_gb}GB GPU")
    print("="*70)
    
    configs = []
    
    if available_memory_gb >= 16:
        configs.append({
            'name': 'Full Model',
            'rank': 128,
            'K_gaussians': 8,
            'batch_size': 1000,
            'script': 'train_gepairre_icews14.ps1'
        })
    
    if available_memory_gb >= 8:
        configs.append({
            'name': 'Standard',
            'rank': 128,
            'K_gaussians': 8,
            'batch_size': 500,
            'script': 'train_gepairre_icews14.ps1 (batch_size 500)'
        })
    
    if available_memory_gb >= 6:
        configs.append({
            'name': 'Low Memory',
            'rank': 64,
            'K_gaussians': 4,
            'batch_size': 256,
            'script': 'train_gepairre_icews14_lowmem.ps1'
        })
    
    if available_memory_gb >= 4:
        configs.append({
            'name': 'Ultra Low Memory',
            'rank': 32,
            'K_gaussians': 2,
            'batch_size': 128,
            'script': 'train_gepairre_icews14_ultralowmem.ps1'
        })
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  --rank {config['rank']}")
        print(f"  --K_gaussians {config['K_gaussians']}")
        print(f"  --batch_size {config['batch_size']}")
        print(f"  Script: {config['script']}")
        
        mem = estimate_memory(7128, 230, config['rank'], config['K_gaussians'], config['batch_size'])
        print(f"  Estimated memory: {mem['total_gb']:.2f} GB")
    
    print("\n" + "="*70)


def main():
    print("\n" + "="*70)
    print("GE-PAIRRE MEMORY PROFILING")
    print("="*70)
    print()
    
    # Check current GPU
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"GPU Detected: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {total_mem:.2f} GB")
        print()
    else:
        print("No GPU detected. Using CPU mode.")
        print()
    
    # Example configurations
    print_memory_estimate('ICEWS14', rank=128, K_gaussians=8, batch_size=1000)
    print_memory_estimate('ICEWS14', rank=128, K_gaussians=8, batch_size=500)
    print_memory_estimate('ICEWS14', rank=64, K_gaussians=4, batch_size=256)
    print_memory_estimate('ICEWS14', rank=32, K_gaussians=2, batch_size=128)
    
    # Recommendations
    if torch.cuda.is_available():
        recommend_config(total_mem)


if __name__ == '__main__':
    main()
