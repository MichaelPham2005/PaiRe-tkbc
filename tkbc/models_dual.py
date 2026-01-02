"""
Dual-Component Gaussian Temporal Encoder for better extrapolation.

Key improvements:
1. Global Trend: W_trend · τ for linear extrapolation
2. Local Pulses: Gaussian RBF for event modeling
3. Sigma Lower Bound: Prevent overfitting to single timestamps
4. Temporal Smoothness: Encourage smooth evolution
"""

import torch
from torch import nn
import numpy as np
from typing import Tuple


class DualComponentEncoder(nn.Module):
    """
    Dual-component temporal encoder with global trend + local pulses.
    
    Formula:
        Δe_r(τ) = W_trend,r · τ + Σ_k A_r,k · G_r,k(τ)
        
    Where:
        - W_trend,r: Global linear trend (learnable per relation)
        - A_r,k: Amplitude of k-th Gaussian pulse
        - G_r,k(τ): Gaussian activation centered at μ_r,k with width σ_r,k
    """
    
    def __init__(self, n_relations: int, dim: int, K: int = 8, 
                 sigma_min: float = 0.02, sigma_max: float = 0.3):
        """
        Args:
            n_relations: Number of relations (includes inverse)
            dim: Embedding dimension
            K: Number of Gaussian pulses per relation
            sigma_min: Minimum width (e.g., 0.02 = 7 days in ICEWS14)
            sigma_max: Maximum width (prevent flat Gaussians)
        """
        super(DualComponentEncoder, self).__init__()
        self.n_relations = n_relations
        self.dim = dim
        self.K = K
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = 1e-9
        
        # GLOBAL TREND COMPONENT
        # W_trend: (n_relations, dim) - per-relation linear evolution
        self.W_trend = nn.Parameter(torch.randn(n_relations, dim) * 1e-4)
        
        # LOCAL PULSE COMPONENT
        # Amplitude: A_{r,k} ∈ R^d
        self.A = nn.Parameter(torch.randn(n_relations, K, dim) * 1e-3)
        
        # Mean: μ_{r,k} ∈ [0,1], evenly spaced
        mu_init = torch.linspace(0, 1, K).unsqueeze(0).expand(n_relations, -1)
        self.mu = nn.Parameter(mu_init)
        
        # Log-scale: s_{r,k} controls σ = exp(s)
        # Initialize to σ ≈ 0.1 (between sigma_min and sigma_max)
        s_init = torch.ones(n_relations, K) * np.log(0.1)
        self.s = nn.Parameter(s_init)
    
    def get_sigma(self):
        """
        Compute width σ with bounds: σ_min ≤ σ ≤ σ_max
        
        Implementation:
            σ = σ_min + (1 - σ_min) · σ_learned
        where σ_learned = sigmoid(s) to ensure [0,1]
        """
        # Sigmoid to [0, 1]
        sigma_normalized = torch.sigmoid(self.s)
        
        # Map to [sigma_min, sigma_max]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma_normalized
        
        return sigma
    
    def forward(self, rel_id: torch.Tensor, tau: torch.Tensor):
        """
        Compute dual-component evolution vectors.
        
        Args:
            rel_id: (batch,) relation indices
            tau: (batch,) normalized timestamps in [0,1]
        
        Returns:
            delta_e: (batch, dim) evolution vector
            components: dict with 'trend' and 'pulses' for analysis
        """
        batch_size = rel_id.shape[0]
        
        # === GLOBAL TREND COMPONENT ===
        W_trend = self.W_trend[rel_id]  # (batch, dim)
        delta_trend = W_trend * tau.unsqueeze(1)  # (batch, dim)
        
        # === LOCAL PULSE COMPONENT ===
        A = self.A[rel_id]  # (batch, K, dim)
        mu = self.mu[rel_id]  # (batch, K)
        sigma = self.get_sigma()[rel_id]  # (batch, K)
        
        # Gaussian activations
        tau_expanded = tau.unsqueeze(1)  # (batch, 1)
        diff = tau_expanded - mu  # (batch, K)
        G = torch.exp(-diff ** 2 / (2 * sigma ** 2 + self.eps))  # (batch, K)
        
        # Weighted sum of pulses
        delta_pulses = torch.sum(A * G.unsqueeze(2), dim=1)  # (batch, dim)
        
        # === TOTAL EVOLUTION ===
        delta_e = delta_trend + delta_pulses
        
        # Return components for analysis
        components = {
            'trend': delta_trend,
            'pulses': delta_pulses
        }
        
        return delta_e, components
    
    def compute_smoothness_loss(self, rel_id: torch.Tensor, 
                                tau: torch.Tensor, epsilon: float = 0.01):
        """
        Compute temporal smoothness regularization.
        
        L_smooth = Σ ||Δe_r(τ) - Δe_r(τ + ε)||²_2
        
        Args:
            rel_id: (batch,) relation indices
            tau: (batch,) timestamps
            epsilon: Small time step (default: 0.01 = ~3.6 days in ICEWS14)
        
        Returns:
            Smoothness loss (scalar)
        """
        # Evolution at current time
        delta_e_t, _ = self.forward(rel_id, tau)
        
        # Evolution at nearby time
        tau_next = torch.clamp(tau + epsilon, 0, 1)  # Stay in [0,1]
        delta_e_t_next, _ = self.forward(rel_id, tau_next)
        
        # L2 difference
        diff = delta_e_t - delta_e_t_next
        smoothness_loss = torch.mean(diff ** 2)
        
        return smoothness_loss


class TemporalSmoothnessRegularizer(nn.Module):
    """
    Regularizer for temporal smoothness.
    Penalizes large changes in evolution vectors over small time intervals.
    """
    
    def __init__(self, weight: float = 0.01, epsilon: float = 0.01):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
    
    def forward(self, encoder: DualComponentEncoder, 
                rel_id: torch.Tensor, tau: torch.Tensor):
        """
        Compute weighted smoothness loss.
        
        Args:
            encoder: DualComponentEncoder instance
            rel_id: (batch,) relation indices
            tau: (batch,) timestamps
        """
        if self.weight == 0:
            return torch.tensor(0.0)
        
        smoothness = encoder.compute_smoothness_loss(rel_id, tau, self.epsilon)
        return self.weight * smoothness


class SigmaLowerBoundRegularizer(nn.Module):
    """
    Regularizer to enforce σ_min lower bound.
    
    L_sigma = Σ ReLU(σ_min - σ)
    
    Prevents Gaussians from becoming too narrow (overfitting).
    """
    
    def __init__(self, weight: float = 0.1, sigma_min: float = 0.02):
        super().__init__()
        self.weight = weight
        self.sigma_min = sigma_min
    
    def forward(self, sigma: torch.Tensor):
        """
        Args:
            sigma: Width parameters (n_relations, K)
        """
        if self.weight == 0:
            return torch.tensor(0.0)
        
        # Penalize σ < σ_min
        violation = torch.relu(self.sigma_min - sigma)
        return self.weight * torch.sum(violation)
