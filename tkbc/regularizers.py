# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class ContinuousTimeLambda3(Regularizer):
    """L3 regularization for continuous time embeddings."""
    def __init__(self, weight: float):
        super(ContinuousTimeLambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        """Apply L3 norm to time embeddings."""
        if factor is None:
            return torch.tensor(0.0)
        
        norm = self.weight * torch.sum(torch.abs(factor) ** 3)
        return norm / factor.shape[0]


<<<<<<< Updated upstream
=======
class TimeParameterRegularizer(Regularizer):
    """
    Light regularization on relation-conditioned time parameters (a_r, A_r).
    Does NOT regularize output m_r(t) directly to avoid collapse.
    
    L_time_param = weight * (||a_r||_2^2 + ||A_r||_2^2)
    """
    def __init__(self, weight: float):
        super(TimeParameterRegularizer, self).__init__()
        self.weight = weight
    
    def forward(self, a_r: torch.Tensor, A_r: torch.Tensor):
        """
        Regularize trend and amplitude parameters.
        
        Args:
            a_r: Trend parameters, shape (n_relations,)
            A_r: Amplitude parameters, shape (n_relations, K)
        Returns:
            Regularization loss (scalar)
        """
        if self.weight == 0:
            return torch.tensor(0.0)
        
        trend_reg = torch.sum(a_r ** 2)
        amplitude_reg = torch.sum(A_r ** 2)
        
        return self.weight * (trend_reg + amplitude_reg)


class AmplitudeDecayRegularizer(Regularizer):
    """
    Regularization for Gaussian amplitude parameters.
    Suppresses unnecessary temporal energy.
    
    L_amp = weight * ||A||_2^2
    """
    def __init__(self, weight: float):
        super(AmplitudeDecayRegularizer, self).__init__()
        self.weight = weight
    
    def forward(self, A: torch.Tensor):
        """
        Regularize amplitude parameters.
        
        Args:
            A: Amplitude tensor, shape (n_relations, K, dim)
        Returns:
            Regularization loss (scalar)
        """
        if self.weight == 0:
            return torch.tensor(0.0)
        
        return self.weight * torch.sum(A ** 2)


class WidthPenaltyRegularizer(Regularizer):
    """
    Width constraint regularizer for Gaussian pulses.
    Prevents flat (static-like) Gaussians.
    
    L_σ = Σ ReLU(σ - σ_max)
    """
    def __init__(self, weight: float, sigma_max: float = 0.2):
        super(WidthPenaltyRegularizer, self).__init__()
        self.weight = weight
        self.sigma_max = sigma_max
    
    def forward(self, sigma: torch.Tensor):
        """
        Penalize widths exceeding sigma_max.
        
        Args:
            sigma: Width parameters, shape (n_relations, K)
        Returns:
            Regularization loss (scalar)
        """
        if self.weight == 0:
            return torch.tensor(0.0)
        
        excess = torch.relu(sigma - self.sigma_max)
        return self.weight * torch.sum(excess)


class SigmaLowerBoundRegularizer(Regularizer):
    """
    Regularizer to enforce σ_min lower bound.
    
    L_sigma = Σ ReLU(σ_min - σ)
    
    Prevents Gaussians from becoming too narrow (overfitting to single timestamps).
    """
    
    def __init__(self, weight: float = 0.1, sigma_min: float = 0.02):
        super(SigmaLowerBoundRegularizer, self).__init__()
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


class TemporalSmoothnessRegularizer(Regularizer):
    """
    Regularizer for temporal smoothness.
    Penalizes large changes in evolution vectors over small time intervals.
    
    L_smooth = Σ ||Δe_r(τ) - Δe_r(τ + ε)||²
    """
    
    def __init__(self, weight: float = 0.01, epsilon: float = 0.01):
        super(TemporalSmoothnessRegularizer, self).__init__()
        self.weight = weight
        self.epsilon = epsilon
    
    def forward(self, encoder, rel_id: torch.Tensor, tau: torch.Tensor):
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


# OLD REGULARIZERS - NOT USED IN NEW MODEL
# Kept for compatibility with old checkpoints only

>>>>>>> Stashed changes
class ContinuitySmoothness(Regularizer):
    """
    Continuity regularizer for time embeddings m = cos(W·t + b).
    Encourages W to have reasonable frequencies (not too high).
    High W → rapid oscillation → discontinuous-looking m.
    """
    def __init__(self, weight: float):
        super(ContinuitySmoothness, self).__init__()
        self.weight = weight
    
    def forward(self, W: torch.Tensor, b: torch.Tensor = None):
        """
        Regularize W to prevent extremely high frequencies.
        Args:
            W: frequency parameter (rank,)
            b: phase parameter (rank,) - optional
        """
        if W is None:
            return torch.tensor(0.0)
        
        # L2 regularization on W to keep frequencies moderate
        # Larger W → faster oscillation → less smooth
        freq_reg = self.weight * torch.sum(W ** 2)
        
        # Optionally regularize b as well (usually not needed)
        if b is not None and self.weight > 0:
            phase_reg = self.weight * 0.1 * torch.sum(b ** 2)  # smaller weight for b
            return freq_reg + phase_reg
        
        return freq_reg
