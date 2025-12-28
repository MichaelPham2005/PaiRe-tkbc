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
    """
    Regularizer for continuous time embeddings.
    Applies N3 regularization to time embeddings to prevent overfitting.
    """
    def __init__(self, weight: float):
        super(ContinuousTimeLambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        """
        Args:
            factor: Time embeddings of shape (batch, rank)
        Returns:
            Regularization loss (scalar)
        """
        if factor is None:
            return torch.tensor(0.0)
        
        # Apply L3 norm (cubic norm) to time embeddings
        # Similar to N3 regularizer but specifically for time
        norm = self.weight * torch.sum(torch.abs(factor) ** 3)
        return norm / factor.shape[0]
