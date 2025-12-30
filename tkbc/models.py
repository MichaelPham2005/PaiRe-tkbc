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


class ContinuousTimeEmbedding(nn.Module):
    """Fourier-based time embedding: m = Linear([cos(W₁t+b₁), sin(W₁t+b₁), ..., cos(Wₖt+bₖ), sin(Wₖt+bₖ)])"""
    def __init__(self, dim: int):
        super(ContinuousTimeEmbedding, self).__init__()
        self.dim = dim
        # k is number of frequencies (usually k = d/2)
        self.k = dim // 2
        
        # W and b for each frequency component
        self.W = nn.Parameter(torch.randn(self.k) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.k))
        
        # Linear layer to project Fourier features to embedding dimension
        # Input: 2k (cos and sin for each frequency)
        # Output: d (embedding dimension)
        self.linear = nn.Linear(2 * self.k, dim, bias=False)
    
    def forward(self, t: torch.Tensor, interval: bool = False):
        """
        Compute Fourier-based time embedding.
        
        Args:
            t: Normalized timestamps in [-1, 1]
               - If interval=False: (batch,) tensor of point timestamps
               - If interval=True: (batch, 2) tensor with [begin, end] timestamps
            interval: Whether t represents intervals or points
        Returns:
            m: (batch, dim) Fourier-based time embedding
        """
        if interval and t.dim() == 2 and t.shape[1] == 2:
            # Handle interval: use midpoint of [begin, end]
            t_begin = t[:, 0]  # (batch,)
            t_end = t[:, 1]    # (batch,)
            t_mid = (t_begin + t_end) / 2.0  # (batch,)
            t_scalar = t_mid
        else:
            # Handle point timestamp
            if t.dim() == 2:
                t_scalar = t.squeeze(-1)  # (batch,)
            else:
                t_scalar = t  # (batch,)
        
        # Compute phase: W_i * t + b_i for each frequency
        phase = t_scalar.unsqueeze(-1) * self.W + self.b  # (batch, k)
        
        # Compute cos and sin for each frequency
        cos_features = torch.cos(phase)  # (batch, k)
        sin_features = torch.sin(phase)  # (batch, k)
        
        # Concatenate: [cos(W₁t+b₁), sin(W₁t+b₁), ..., cos(Wₖt+bₖ), sin(Wₖt+bₖ)]
        fourier_features = torch.cat([cos_features, sin_features], dim=-1)  # (batch, 2k)
        
        # Linear projection to embedding dimension
        m = self.linear(fourier_features)  # (batch, dim)
        
        return m


class ContinuousPairRE(TKBCModel):
    """Continuous-time PairRE with relation-wise temporal gating"""
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, init_size: float = 1e-3):
        super(ContinuousPairRE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(sizes[0], rank, sparse=True)
        self.entity_embeddings.weight.data *= init_size
        
        # Relation projections (PairRE style: r^H and r^T)
        self.relation_head = nn.Embedding(sizes[1], rank, sparse=True)
        self.relation_tail = nn.Embedding(sizes[1], rank, sparse=True)
        self.relation_head.weight.data *= init_size
        self.relation_tail.weight.data *= init_size
        
        # Continuous time embedding module
        self.time_encoder = ContinuousTimeEmbedding(rank)
        
        # Relation-wise temporal gating parameter (alpha)
        # alpha close to 0 = static, alpha close to 1 = dynamic
        self.alpha = nn.Embedding(sizes[1], 1)
        nn.init.constant_(self.alpha.weight, 0.5)  # sigmoid(0.5) ≈ 0.62 (bias towards dynamic)
    
    @staticmethod
    def has_time():
        return True
    
    def score(self, x: torch.Tensor):
        h = self.entity_embeddings(x[:, 0].long())
        r_h = self.relation_head(x[:, 1].long())
        t = self.entity_embeddings(x[:, 2].long())
        r_t = self.relation_tail(x[:, 1].long())
        
        # Check if input has interval (5 columns) or point (4 columns)
        if x.shape[1] >= 5:
            # Interval mode: [h, r, t, begin, end]
            time_begin = x[:, 3].float()
            time_end = x[:, 4].float()
            time_interval = torch.stack([time_begin, time_end], dim=1)  # (batch, 2)
            m_fourier = self.time_encoder(time_interval, interval=True)
        else:
            # Point mode: [h, r, t, time]
            time_continuous = x[:, 3].float()
            m_fourier = self.time_encoder(time_continuous, interval=False)
        
        alpha = torch.sigmoid(self.alpha(x[:, 1].long()))
        gate = alpha * m_fourier + (1 - alpha) * torch.ones_like(m_fourier)
        
        interaction = (h * r_h - t * r_t) * gate
        score = -torch.norm(interaction, p=1, dim=-1)
        return score
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        
        # Extract components
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        
        # Get continuous time embedding (handle both point and interval)
        if x.shape[1] >= 5:
            # Interval mode: [h, r, t, begin, end]
            time_begin = x[:, 3].float()
            time_end = x[:, 4].float()
            time_interval = torch.stack([time_begin, time_end], dim=1)  # (batch, 2)
            m_fourier = self.time_encoder(time_interval, interval=True)
        else:
            # Point mode: [h, r, t, time]
            time_continuous = x[:, 3].float()
            m_fourier = self.time_encoder(time_continuous, interval=False)
        
        # Get relation-specific gating coefficient
        alpha = torch.sigmoid(self.alpha(x[:, 1].long()))
        
        # Compute gated time modulation: G(m,r) = alpha_r * m_fourier + (1 - alpha_r) * 1
        gate = alpha * m_fourier + (1 - alpha) * torch.ones_like(m_fourier)
        
        # Get all entity embeddings
        all_entities = self.entity_embeddings.weight  # (n_entities, rank)
        
        # VECTORIZED: Compute scores for all entities at once
        # h * r_h: (batch, rank)
        # all_entities: (n_entities, rank)
        # r_t: (batch, rank)
        # gate: (batch, rank)
        
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
        return scores, factors, m_fourier
    
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
        
        # Get all normalized timestamps
        # Assuming dataset provides ts_normalized_array or we compute it
        # For now, create linearly spaced timestamps in [-1, 1]
        n_timestamps = self.sizes[3]  # Number of timestamps in dataset
        all_times = torch.linspace(-1, 1, n_timestamps).to(h.device)  # (n_timestamps,)
        
        # Compute time embeddings for all timestamps
        all_m_fourier = self.time_encoder(all_times)  # (n_timestamps, rank)
        
        # Get relation-specific alpha
        alpha = torch.sigmoid(self.alpha(x[:, 1].long()))  # (batch, 1)
        
        # Expand for broadcasting
        # h, r_h, t, r_t: (batch, rank)
        # all_m_fourier: (n_timestamps, rank)
        # alpha: (batch, 1)
        
        # Compute interaction for each timestamp
        h_r_h = h * r_h  # (batch, rank)
        t_r_t = t * r_t  # (batch, rank)
        
        # Expand dimensions
        h_r_h_exp = h_r_h.unsqueeze(1)  # (batch, 1, rank)
        t_r_t_exp = t_r_t.unsqueeze(1)  # (batch, 1, rank)
        alpha_exp = alpha.unsqueeze(1)  # (batch, 1, 1)
        all_m_exp = all_m_fourier.unsqueeze(0)  # (1, n_timestamps, rank)
        
        # Compute gate for all timestamps: G(m,r) = alpha * m + (1-alpha) * 1
        # Broadcasting: (batch, 1, 1) * (1, n_timestamps, rank) + (batch, 1, 1)
        gate_all = alpha_exp * all_m_exp + (1 - alpha_exp) * torch.ones_like(all_m_exp)
        # gate_all: (batch, n_timestamps, rank)
        
        # Compute interaction: (h * r_h - t * r_t) * gate
        # (batch, 1, rank) - (batch, 1, rank) = (batch, 1, rank)
        # Then broadcast with gate_all: (batch, n_timestamps, rank)
        interaction = (h_r_h_exp - t_r_t_exp) * gate_all
        
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
        
        time_continuous = queries[:, 3].float()
        m_fourier = self.time_encoder(time_continuous)
        
        alpha = torch.sigmoid(self.alpha(queries[:, 1].long()))
        gate = alpha * m_fourier + (1 - alpha) * torch.ones_like(m_fourier)
        
        return h * r_h * gate

