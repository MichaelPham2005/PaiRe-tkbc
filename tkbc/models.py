# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1,
            timestamp_ids: torch.Tensor = None
    ):
        """Returns filtered ranking for queries."""
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)  # (batch,)
                    targets = targets.unsqueeze(1)  # (batch, 1) for broadcasting
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        # Use original timestamp ID for filtering if provided (continuous time case)
                        if timestamp_ids is not None:
                            ts_for_filter = int(timestamp_ids[b_begin + i].item())
                        else:
                            ts_for_filter = int(query[3].item())
                        filter_out = filters[(int(query[0].item()), int(query[1].item()), ts_for_filter)]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)


class RelationConditionedTimeEncoder(nn.Module):
    """
    Relation-conditioned continuous time encoder.
    Each relation has its own temporal dynamics: trend (a_r), amplitude (A_r), phase (P_r).
    Global frequencies (omega) are shared across relations.
    
    Formula:
        z_r(tau) = [a_r * tau, A_r * sin(omega * tau + P_r)]
        m_r(tau) = tanh(W_proj @ z_r + b_proj)
    """
    def __init__(self, n_relations: int, dim: int, K: int = 16):
        """
        Args:
            n_relations: Number of relations
            dim: Embedding dimension
            K: Number of frequencies (default 16)
        """
        super(RelationConditionedTimeEncoder, self).__init__()
        self.n_relations = n_relations
        self.dim = dim
        self.K = K
        
        # Per-relation parameters
        # Trend: a_r ∈ R for each relation
        self.a_r = nn.Parameter(torch.randn(n_relations) * 0.01)
        
        # Amplitude: A_r ∈ R^K for each relation
        self.A_r = nn.Parameter(torch.randn(n_relations, K) * 0.1)
        
        # Phase: P_r ∈ R^K for each relation
        self.P_r = nn.Parameter(torch.rand(n_relations, K) * 2 * np.pi - np.pi)  # Uniform[-π, π]
        
        # Global frequencies: omega ∈ R^K (shared across all relations)
        # Initialize with log-spaced frequencies for better coverage
        # FIXED (not learnable as per spec section 11)
        omega_init = torch.logspace(0, np.log10(10), K)  # [1, ..., 10]
        self.register_buffer('omega', omega_init)
        
        # Projection layer: (1 + K) -> dim
        self.W_proj = nn.Parameter(torch.empty(dim, 1 + K))
        self.b_proj = nn.Parameter(torch.zeros(dim))
        
        # Xavier initialization for projection with 3x scale to amplify time signal
        nn.init.xavier_uniform_(self.W_proj)
        self.W_proj.data *= 3.0
    
    def forward(self, rel_id: torch.Tensor, tau: torch.Tensor):
        """
        Compute relation-conditioned time embedding.
        
        Args:
            rel_id: (batch,) relation indices
            tau: (batch,) normalized continuous timestamps in [0, 1]
        
        Returns:
            m: (batch, dim) time embedding
        """
        batch_size = rel_id.shape[0]
        
        # Lookup per-relation parameters
        a = self.a_r[rel_id]  # (batch,)
        A = self.A_r[rel_id]  # (batch, K)
        P = self.P_r[rel_id]  # (batch, K)
        
        # Compute trend: z_0 = a_r * tau
        z_trend = a * tau  # (batch,)
        
        # Compute periodic: z_k = A_r * sin(omega_k * tau + P_r)
        # tau: (batch,) -> (batch, 1)
        # omega: (K,) -> (1, K)
        # Result: (batch, K)
        phase = self.omega.unsqueeze(0) * tau.unsqueeze(1) + P  # (batch, K)
        z_periodic = A * torch.sin(phase)  # (batch, K)
        
        # Concatenate: [z_trend, z_periodic]
        z = torch.cat([z_trend.unsqueeze(1), z_periodic], dim=1)  # (batch, 1+K)
        
        # Project: m = tanh(W_proj @ z + b_proj)
        m = torch.tanh(z @ self.W_proj.t() + self.b_proj)  # (batch, dim)
        
        # CRITICAL FIX: Center time embeddings to prevent bias
        # Without this, m tends to have non-zero mean (observed: 0.8046)
        # This causes gate to be constantly shifted, not time-dependent
        m = m - m.mean(dim=0, keepdim=True)  # Zero-mean per dimension
        
        return m


class GaussianTemporalEncoder(nn.Module):
    """
    Gaussian Evolution Temporal Encoder for GE-PairRE.
    Each relation has K Gaussian pulses that activate at specific times.
    
    Formula:
        G_{r,k}(τ) = exp(-(τ - μ_{r,k})² / (2σ_{r,k}² + ε))
        Δe(r, τ) = Σ_k A_{r,k} · G_{r,k}(τ)
    """
    def __init__(self, n_relations: int, dim: int, K: int = 8, sigma_max: float = 0.2):
        """
        Args:
            n_relations: Number of relations (includes inverse relations)
            dim: Embedding dimension
            K: Number of Gaussian pulses per relation
            sigma_max: Maximum allowed width
        """
        super(GaussianTemporalEncoder, self).__init__()
        self.n_relations = n_relations
        self.dim = dim
        self.K = K
        self.sigma_max = sigma_max
        self.eps = 1e-9  # For numerical stability
        
        # Per-relation Gaussian parameters
        # Amplitude: A_{r,k} ∈ R^d for each relation and pulse
        self.A = nn.Parameter(torch.randn(n_relations, K, dim) * 1e-4)
        
        # Mean: μ_{r,k} ∈ [0,1] for each pulse
        # Initialize with evenly spaced centers
        mu_init = torch.linspace(0, 1, K).unsqueeze(0).expand(n_relations, -1)
        self.mu = nn.Parameter(mu_init)
        
        # Log-scale: s_{r,k} controls width σ = exp(s)
        # Initialize small: σ ≈ 0.01 (s ≈ log(0.01) = -4.6)
        s_init = torch.ones(n_relations, K) * np.log(0.01)
        self.s = nn.Parameter(s_init)
    
    def get_sigma(self):
        """Compute width σ = exp(s), ensuring σ > 0."""
        return torch.exp(self.s)
    
    def forward(self, rel_id: torch.Tensor, tau: torch.Tensor):
        """
        Compute evolution vectors for given relations and times.
        
        Args:
            rel_id: (batch,) relation indices
            tau: (batch,) normalized timestamps in [0,1]
        
        Returns:
            delta_e: (batch, dim) evolution vector
        """
        batch_size = rel_id.shape[0]
        
        # Lookup per-relation parameters
        A = self.A[rel_id]  # (batch, K, dim)
        mu = self.mu[rel_id]  # (batch, K)
        sigma = self.get_sigma()[rel_id]  # (batch, K)
        
        # Compute Gaussian activations: G_{r,k}(τ) = exp(-(τ - μ)² / (2σ² + ε))
        # tau: (batch,) -> (batch, 1)
        # mu: (batch, K)
        tau_expanded = tau.unsqueeze(1)  # (batch, 1)
        diff = tau_expanded - mu  # (batch, K)
        G = torch.exp(-diff ** 2 / (2 * sigma ** 2 + self.eps))  # (batch, K)
        
        # Compute evolution: Δe = Σ_k A_{r,k} · G_{r,k}(τ)
        # G: (batch, K) -> (batch, K, 1) for broadcasting
        # A: (batch, K, dim)
        delta_e = torch.sum(A * G.unsqueeze(2), dim=1)  # (batch, dim)
        
        return delta_e


class GEPairRE(TKBCModel):
    """
    Gaussian Evolution PairRE (GE-PairRE).
    
    Additive temporal evolution:
        h_τ = h + Δe_h(r, τ)
        t_τ = t + Δe_t(r⁻¹, τ)
        φ(h, r, t, τ) = -||h_τ ∘ r^H - t_τ ∘ r^T||₁
    
    Key properties:
    - Absolute static preservation (Δe → 0 when τ far from all μ)
    - Event-driven dynamics (local Gaussian pulses)
    - No oscillation artifacts
    """
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, 
                 init_size: float = 1e-3, K: int = 8, sigma_max: float = 0.2):
        super(GEPairRE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.K = K
        self.sigma_max = sigma_max
        
        # Static embeddings
        self.entity_embeddings = nn.Embedding(sizes[0], rank, sparse=True)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.entity_embeddings.weight.data *= init_size
        
        # Relation projections (PairRE style: r^H and r^T)
        self.relation_head = nn.Embedding(sizes[1], rank, sparse=True)
        self.relation_tail = nn.Embedding(sizes[1], rank, sparse=True)
        nn.init.xavier_normal_(self.relation_head.weight)
        nn.init.xavier_normal_(self.relation_tail.weight)
        self.relation_head.weight.data *= init_size
        self.relation_tail.weight.data *= init_size
        
        # Gaussian temporal encoder
        self.time_encoder = GaussianTemporalEncoder(
            n_relations=sizes[1], 
            dim=rank, 
            K=K,
            sigma_max=sigma_max
        )
    
    @staticmethod
    def has_time():
        return True
    
    def get_evolution_vectors(self, rel_id: torch.Tensor, tau: torch.Tensor):
        """
        Get evolution vectors for head and tail entities.
        
        For inverse relations: if rel_id >= n_relations/2, it's an inverse.
        Head evolution uses forward relation, tail uses inverse.
        
        Args:
            rel_id: (batch,) relation indices
            tau: (batch,) timestamps
        
        Returns:
            delta_e_h: (batch, dim) head evolution
            delta_e_t: (batch, dim) tail evolution
        """
        # Forward relation for head evolution
        delta_e_h = self.time_encoder(rel_id, tau)
        
        # For tail, we need the inverse relation
        # In TKBC convention: inverse_id = rel_id + n_relations/2
        n_base_relations = self.sizes[1] // 2
        inverse_rel_id = torch.where(
            rel_id < n_base_relations,
            rel_id + n_base_relations,
            rel_id - n_base_relations
        )
        delta_e_t = self.time_encoder(inverse_rel_id, tau)
        
        return delta_e_h, delta_e_t
    
    def score(self, x: torch.Tensor):
        """
        Score a batch of (h, r, t, tau) quadruples.
        
        Args:
            x: (batch, 4) tensor [head_id, rel_id, tail_id, tau]
        Returns:
            scores: (batch,) L1-norm based scores
        """
        # Static embeddings
        h = self.entity_embeddings(x[:, 0].long())
        r_h = self.relation_head(x[:, 1].long())
        t = self.entity_embeddings(x[:, 2].long())
        r_t = self.relation_tail(x[:, 1].long())
        
        # Get evolution vectors
        rel_id = x[:, 1].long()
        tau = x[:, 3].float()
        delta_e_h, delta_e_t = self.get_evolution_vectors(rel_id, tau)
        
        # Evolved entities
        h_tau = h + delta_e_h
        t_tau = t + delta_e_t
        
        # PairRE scoring: -||h_τ ∘ r^H - t_τ ∘ r^T||₁
        interaction = h_tau * r_h - t_tau * r_t
        score = -torch.norm(interaction, p=1, dim=-1)
        return score
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for 1-vs-All prediction.
        
        Args:
            x: (batch, 4) tensor [head_id, rel_id, tail_id, tau]
        Returns:
            scores: (batch, n_entities) scores for all entities
            factors: tuple of embeddings for regularization
            delta_e: (batch, dim) evolution vectors for visualization
        """
        batch_size = x.shape[0]
        
        # Static embeddings
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        
        # Get evolution vectors
        rel_id = x[:, 1].long()
        tau = x[:, 3].float()
        delta_e_h, delta_e_t = self.get_evolution_vectors(rel_id, tau)
        
        # Evolved head
        h_tau = h + delta_e_h  # (batch, rank)
        
        # Get all entity embeddings
        all_entities = self.entity_embeddings.weight  # (n_entities, rank)
        
        # VECTORIZED: Compute scores for all possible tails
        # For each entity e_i as tail candidate:
        #   score = -||h_τ ∘ r^H - (e_i + Δe_t) ∘ r^T||₁
        
        h_tau_r_h = (h_tau * r_h).unsqueeze(1)  # (batch, 1, rank)
        r_t_expanded = r_t.unsqueeze(1)  # (batch, 1, rank)
        delta_e_t_expanded = delta_e_t.unsqueeze(1)  # (batch, 1, rank)
        all_entities_expanded = all_entities.unsqueeze(0)  # (1, n_entities, rank)
        
        # Evolved tail candidates
        all_tails_tau = all_entities_expanded + delta_e_t_expanded  # (batch, n_entities, rank)
        
        # Compute interaction
        interaction = h_tau_r_h - all_tails_tau * r_t_expanded  # (batch, n_entities, rank)
        
        # Compute L1 norm
        scores = -torch.norm(interaction, p=1, dim=2)  # (batch, n_entities)
        
        # Factors for regularization
        t = self.entity_embeddings(x[:, 2].long())
        factors = (
            torch.sqrt(h ** 2 + 1e-10),
            torch.sqrt(r_h ** 2 + r_t ** 2 + 1e-10),
            torch.sqrt(t ** 2 + 1e-10)
        )
        
        return scores, factors, delta_e_h
    
    def forward_over_time(self, x: torch.Tensor):
        """
        Score (h, r, t) queries over all timestamps.
        
        Args:
            x: (batch, 3) tensor [head, relation, tail] WITHOUT time
        Returns:
            scores: (batch, n_timestamps) scores for each timestamp
        """
        batch_size = x.shape[0]
        
        # Static embeddings
        h = self.entity_embeddings(x[:, 0].long())
        r_h = self.relation_head(x[:, 1].long())
        t = self.entity_embeddings(x[:, 2].long())
        r_t = self.relation_tail(x[:, 1].long())
        rel_id = x[:, 1].long()
        
        # Get all normalized timestamps
        n_timestamps = self.sizes[3]
        all_taus = torch.linspace(0, 1, n_timestamps).to(h.device)
        
        # Expand for all timestamps
        rel_id_exp = rel_id.unsqueeze(1).expand(-1, n_timestamps).reshape(-1)
        all_taus_exp = all_taus.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        
        # Compute evolution vectors for all times
        delta_e_h_all, delta_e_t_all = self.get_evolution_vectors(rel_id_exp, all_taus_exp)
        delta_e_h_all = delta_e_h_all.view(batch_size, n_timestamps, self.rank)
        delta_e_t_all = delta_e_t_all.view(batch_size, n_timestamps, self.rank)
        
        # Evolved entities
        h_tau_all = h.unsqueeze(1) + delta_e_h_all  # (batch, n_timestamps, rank)
        t_tau_all = t.unsqueeze(1) + delta_e_t_all  # (batch, n_timestamps, rank)
        
        # Compute scores
        h_tau_r_h = h_tau_all * r_h.unsqueeze(1)  # (batch, n_timestamps, rank)
        t_tau_r_t = t_tau_all * r_t.unsqueeze(1)  # (batch, n_timestamps, rank)
        interaction = h_tau_r_h - t_tau_r_t  # (batch, n_timestamps, rank)
        scores = -torch.norm(interaction, p=1, dim=2)  # (batch, n_timestamps)
        
        return scores
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """Get entity embeddings for ranking."""
        return self.entity_embeddings.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
    
    def get_queries(self, queries: torch.Tensor):
        """Compute query embeddings (left side of PairRE interaction)."""
        h = self.entity_embeddings(queries[:, 0].long())
        r_h = self.relation_head(queries[:, 1].long())
        
        rel_id = queries[:, 1].long()
        tau = queries[:, 3].float()
        delta_e_h, _ = self.get_evolution_vectors(rel_id, tau)
        
        h_tau = h + delta_e_h
        return h_tau * r_h


class ContinuousPairRE(TKBCModel):
    """
    Continuous-time PairRE with Relation-Conditioned Time Encoding and Residual Gate.
    
    Key changes from old version:
    - NO alpha gating (no mixing with vector 1)
    - Residual gate: g_r(tau) = 1 + beta * tanh(m_r(tau))
    - Relation-conditioned time encoder with trend + periodic components
    """
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, 
                 init_size: float = 0.1, K: int = 16, beta: float = 0.5):
        super(ContinuousPairRE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.K = K
        self.beta = beta
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(sizes[0], rank, sparse=True)
        self.entity_embeddings.weight.data *= init_size
        
        # Relation projections (PairRE style: r^H and r^T)
        self.relation_head = nn.Embedding(sizes[1], rank, sparse=True)
        self.relation_tail = nn.Embedding(sizes[1], rank, sparse=True)
        self.relation_head.weight.data *= init_size
        self.relation_tail.weight.data *= init_size
        
        # Relation-conditioned time encoder
        self.time_encoder = RelationConditionedTimeEncoder(
            n_relations=sizes[1], 
            dim=rank, 
            K=K
        )
    
    @staticmethod
    def has_time():
        return True
    
    def residual_gate(self, m: torch.Tensor):
        """
        Compute residual gate: g = 1 + beta * tanh(m)
        
        Args:
            m: (batch, rank) time embedding
        Returns:
            g: (batch, rank) gate values (always > 0 if beta < 1)
        """
        return 1.0 + self.beta * torch.tanh(m)
    
    def score(self, x: torch.Tensor):
        """
        Score a batch of (h, r, t, tau) quadruples.
        
        Args:
            x: (batch, 4) tensor [head_id, rel_id, tail_id, tau]
        Returns:
            scores: (batch,) L1-norm based scores
        """
        h = self.entity_embeddings(x[:, 0].long())
        r_h = self.relation_head(x[:, 1].long())
        t = self.entity_embeddings(x[:, 2].long())
        r_t = self.relation_tail(x[:, 1].long())
        
        # Get relation-conditioned time embedding
        rel_id = x[:, 1].long()
        tau = x[:, 3].float()
        m = self.time_encoder(rel_id, tau)
        
        # Compute residual gate
        gate = self.residual_gate(m)
        
        # PairRE interaction with time gate
        interaction = (h * r_h - t * r_t) * gate
        score = -torch.norm(interaction, p=1, dim=-1)
        return score
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for 1-vs-All prediction.
        
        Args:
            x: (batch, 4) tensor [head_id, rel_id, tail_id, tau]
        Returns:
            scores: (batch, n_entities) scores for all entities
            factors: tuple of embeddings for regularization
            time_emb: (batch, rank) time embeddings
        """
        batch_size = x.shape[0]
        
        # Extract components
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        
        # Get relation-conditioned time embedding
        rel_id = x[:, 1].long()
        tau = x[:, 3].float()
        m = self.time_encoder(rel_id, tau)  # (batch, rank)
        
        # Compute residual gate
        gate = self.residual_gate(m)  # (batch, rank)
        
        # Get all entity embeddings
        all_entities = self.entity_embeddings.weight  # (n_entities, rank)
        
        # VECTORIZED: Compute scores for all entities at once
        # Expand dimensions for broadcasting
        h_r_h = (h * r_h).unsqueeze(1)  # (batch, 1, rank)
        r_t_expanded = r_t.unsqueeze(1)  # (batch, 1, rank)
        gate_expanded = gate.unsqueeze(1)  # (batch, 1, rank)
        all_entities_expanded = all_entities.unsqueeze(0)  # (1, n_entities, rank)
        
        # Compute interaction: (h * r_h - entity * r_t) * gate
        # Broadcasting: (batch, 1, rank) - (1, n_entities, rank) * (batch, 1, rank) -> (batch, n_entities, rank)
        interaction = (h_r_h - all_entities_expanded * r_t_expanded) * gate_expanded
        
        # Compute L1 norm along rank dimension
        scores = -torch.norm(interaction, p=1, dim=2)  # (batch, n_entities)
        
        t = self.entity_embeddings(x[:, 2].long())
        
        factors = (
            torch.sqrt(h ** 2 + 1e-10),
            torch.sqrt(r_h ** 2 + r_t ** 2 + 1e-10),
            torch.sqrt(t ** 2 + 1e-10)
        )
        return scores, factors, m
    
    def get_ranking(self, queries: torch.Tensor, filters: Dict[Tuple[int, int, int], List[int]],
                    batch_size: int = 1000, chunk_size: int = -1, timestamp_ids: torch.Tensor = None):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                all_scores, _, _ = self.forward(these_queries)
                targets = self.score(these_queries).unsqueeze(1)
                
                for i, query in enumerate(these_queries):
                    if timestamp_ids is not None:
                        ts_for_filter = int(timestamp_ids[b_begin + i].item())
                    else:
                        ts_for_filter = int(query[3].item())
                    
                    # Get entities to filter (but DON'T include target - it's added below)
                    filter_out = filters[(int(query[0].item()), int(query[1].item()), ts_for_filter)]
                    # Filter out known true entities (except the target we're testing)
                    target_idx = int(queries[b_begin + i, 2].item())
                    filter_out = [x for x in filter_out if x != target_idx]
                    if len(filter_out) > 0:
                        all_scores[i, torch.LongTensor(filter_out)] = -1e6
                
                # Compute ranks
                ranks[b_begin:b_begin + batch_size] = torch.sum(
                    (all_scores >= targets).float(), dim=1
                ).cpu()
                
                b_begin += batch_size
        
        return ranks
    
    def forward_over_time(self, x: torch.Tensor):
        """
        Score (h, r, t) queries over all timestamps for time prediction.
        
        Args:
            x: (batch, 3) tensor [head, relation, tail] WITHOUT time
        Returns:
            scores: (batch, n_timestamps) scores for each timestamp
        """
        batch_size = x.shape[0]
        
        # Extract components (no time in input)
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        t = self.entity_embeddings(x[:, 2].long())   # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        rel_id = x[:, 1].long()  # (batch,)
        
        # Get all normalized timestamps in [0, 1]
        n_timestamps = self.sizes[3]
        all_taus = torch.linspace(0, 1, n_timestamps).to(h.device)  # (n_timestamps,)
        
        # Expand rel_id for all timestamps
        # rel_id: (batch,) -> (batch, n_timestamps)
        rel_id_exp = rel_id.unsqueeze(1).expand(-1, n_timestamps).reshape(-1)  # (batch * n_timestamps,)
        all_taus_exp = all_taus.unsqueeze(0).expand(batch_size, -1).reshape(-1)  # (batch * n_timestamps,)
        
        # Compute time embeddings for all (relation, time) pairs
        all_m = self.time_encoder(rel_id_exp, all_taus_exp)  # (batch * n_timestamps, rank)
        all_m = all_m.view(batch_size, n_timestamps, self.rank)  # (batch, n_timestamps, rank)
        
        # Compute gates
        all_gates = self.residual_gate(all_m)  # (batch, n_timestamps, rank)
        
        # Compute interaction for each timestamp
        h_r_h = (h * r_h).unsqueeze(1)  # (batch, 1, rank)
        t_r_t = (t * r_t).unsqueeze(1)  # (batch, 1, rank)
        
        # interaction: (h * r_h - t * r_t) * gate
        interaction = (h_r_h - t_r_t) * all_gates  # (batch, n_timestamps, rank)
        
        # Compute scores: -||interaction||_1
        scores = -torch.norm(interaction, p=1, dim=2)  # (batch, n_timestamps)
        
        return scores
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """Get entity embeddings for ranking."""
        return self.entity_embeddings.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
    
    def get_queries(self, queries: torch.Tensor):
        """Compute query embeddings (left side of PairRE interaction)."""
        h = self.entity_embeddings(queries[:, 0].long())
        r_h = self.relation_head(queries[:, 1].long())
        
        rel_id = queries[:, 1].long()
        tau = queries[:, 3].float()
        m = self.time_encoder(rel_id, tau)
        gate = self.residual_gate(m)
        
        return h * r_h * gate

