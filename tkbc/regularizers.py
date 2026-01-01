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


# OLD REGULARIZERS - NOT USED IN NEW MODEL
# Kept for compatibility with old checkpoints only

class ContinuitySmoothness(Regularizer):
    """
    Continuity regularizer for Fourier-based time embeddings.
    Encourages W to have reasonable frequencies (not too high).
    High W → rapid oscillation → discontinuous-looking m.
    """
    def __init__(self, weight: float):
        super(ContinuitySmoothness, self).__init__()
        self.weight = weight
    
    def forward(self, W: torch.Tensor, b: torch.Tensor = None, linear_weight: torch.Tensor = None):
        """
        Regularize W, b, and optionally the linear layer weights.
        Args:
            W: frequency parameter (k,)
            b: phase parameter (k,) - optional
            linear_weight: linear layer weights (d, 2k) - optional
        """
        if W is None:
            return torch.tensor(0.0)
        
        # L2 regularization on W to keep frequencies moderate
        # Larger W → faster oscillation → less smooth
        freq_reg = self.weight * torch.sum(W ** 2)
        
        # Optionally regularize b as well (usually not needed)
        if b is not None and self.weight > 0:
            phase_reg = self.weight * 0.1 * torch.sum(b ** 2)  # smaller weight for b
            freq_reg = freq_reg + phase_reg
        
        # Optionally regularize linear layer to prevent extreme weights
        if linear_weight is not None and self.weight > 0:
            linear_reg = self.weight * 0.01 * torch.sum(linear_weight ** 2)
            freq_reg = freq_reg + linear_reg
        
        return freq_reg


class AlphaPolarization(Regularizer):
    """
    Polarization regularizer for alpha parameters.
    Encourages alpha_r to be close to 0 (static) or 1 (dynamic).
    
    Uses entropy-based penalty: H(alpha) = -alpha*log(alpha) - (1-alpha)*log(1-alpha)
    Minimum entropy at alpha=0 or alpha=1 (polarized).
    Maximum entropy at alpha=0.5 (uncertain).
    """
    def __init__(self, weight: float):
        super(AlphaPolarization, self).__init__()
        self.weight = weight
    
    def forward(self, alpha_raw: torch.Tensor):
        """
        Apply polarization penalty to push alphas toward 0 or 1.
        
        Args:
            alpha_raw: Raw alpha logits (before sigmoid), shape (num_relations, 1)
        Returns:
            Polarization loss (scalar)
        """
        if alpha_raw is None or self.weight == 0:
            return torch.tensor(0.0)
        
        # Apply sigmoid to get alpha in (0, 1)
        alpha = torch.sigmoid(alpha_raw)
        
        # Clip to avoid log(0)
        eps = 1e-7
        alpha = torch.clamp(alpha, eps, 1 - eps)
        
        # Binary entropy: H = -alpha*log(alpha) - (1-alpha)*log(1-alpha)
        # We want to MINIMIZE entropy (maximize polarization)
        entropy = -alpha * torch.log(alpha) - (1 - alpha) * torch.log(1 - alpha)
        
        # Average over all relations
        polarization_loss = self.weight * torch.mean(entropy)
        
        return polarization_loss
