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
