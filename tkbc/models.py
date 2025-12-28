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
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :param timestamp_ids: original discrete timestamp IDs (for continuous time models)
        :return:
        """
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
                    targets = self.score(these_queries)
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
    """
    Continuous time embedding using learnable cosine-based regression.
    
    Implements: m = cos(W * tau + b)
    where tau is the normalized continuous timestamp.
    """
    def __init__(self, dim: int):
        super(ContinuousTimeEmbedding, self).__init__()
        self.dim = dim
        # Learnable parameters for continuous time embedding
        self.W = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))
    
    def forward(self, t: torch.Tensor):
        """
        Args:
            t: Normalized continuous timestamps of shape (batch_size,)
        Returns:
            Time embeddings of shape (batch_size, dim)
        """
        # Expand t to (batch_size, dim) and compute cos(W * t + b)
        return torch.cos(t.unsqueeze(-1) * self.W + self.b)


class ContinuousPairRE(TKBCModel):
    """
    Continuous-time PairRE model with relation-wise temporal gating.
    
    Scoring function: ||( (h * r^H - t * r^T) * (alpha * m + (1-alpha) * 1) )||
    where:
        - h, t: head and tail entity embeddings
        - r^H, r^T: relation-specific projection vectors (PairRE style)
        - m: continuous time embedding
        - alpha: relation-specific temporal gating coefficient
    """
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
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
        nn.init.constant_(self.alpha.weight, 0.5)  # Initialize to 0.5
    
    @staticmethod
    def has_time():
        return True
    
    def score(self, x: torch.Tensor):
        """
        Compute scores for given quadruples.
        
        Args:
            x: Tensor of shape (batch_size, 4) containing [head, relation, tail, time]
               where time is a normalized continuous value
        Returns:
            Scores of shape (batch_size, 1)
        """
        # Extract components - convert first 3 columns to long for embedding lookup
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        t = self.entity_embeddings(x[:, 2].long())   # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        
        # Get continuous time embedding (keep as float)
        time_continuous = x[:, 3].float()     # (batch,)
        m = self.time_encoder(time_continuous) # (batch, rank)
        
        # Get relation-specific gating coefficient
        alpha = torch.sigmoid(self.alpha(x[:, 1].long()))  # (batch, 1)
        
        # Compute gated time modulation: alpha * m + (1 - alpha) * 1
        gate = alpha * m + (1 - alpha)  # (batch, rank)
        
        # PairRE interaction with temporal gating
        interaction = (h * r_h - t * r_t) * gate  # (batch, rank)
        
        # Score is negative L1 norm (higher score = better)
        score = -torch.abs(interaction).sum(dim=-1, keepdim=True)
        
        return score
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for training.
        
        Args:
            x: Tensor of shape (batch_size, 4) containing [head, relation, tail, time]
        Returns:
            - scores: (batch, n_entities) scoring all possible tails
            - factors: tuple of regularization factors
            - time_embedding: for compatibility
        """
        batch_size = x.shape[0]
        
        # Extract components
        h = self.entity_embeddings(x[:, 0].long())  # (batch, rank)
        r_h = self.relation_head(x[:, 1].long())     # (batch, rank)
        r_t = self.relation_tail(x[:, 1].long())     # (batch, rank)
        
        # Get continuous time embedding
        time_continuous = x[:, 3].float()
        m = self.time_encoder(time_continuous)
        
        # Get relation-specific gating coefficient
        alpha = torch.sigmoid(self.alpha(x[:, 1].long()))
        
        # Compute gated time modulation
        gate = alpha * m + (1 - alpha)
        
        # Get all entity embeddings
        all_entities = self.entity_embeddings.weight  # (n_entities, rank)
        
        # For each query, compute scores against all entities
        # Score: -||(h * r_h - entity * r_t) * gate||_1
        scores = []
        for i in range(batch_size):
            h_i = h[i:i+1]  # (1, rank)
            r_h_i = r_h[i:i+1]  # (1, rank)
            r_t_i = r_t[i:i+1]  # (1, rank)
            gate_i = gate[i:i+1]  # (1, rank)
            
            # Broadcast computation
            interaction = (h_i * r_h_i - all_entities * r_t_i) * gate_i  # (n_entities, rank)
            score = -torch.abs(interaction).sum(dim=1)  # (n_entities,)
            scores.append(score)
        
        scores = torch.stack(scores)  # (batch, n_entities)
        
        # Get tail embeddings for regularization
        t = self.entity_embeddings(x[:, 2].long())  # (batch, rank)
        
        # Return regularization factors as individual tensors (not aggregated)
        # This allows N3 regularizer to compute proper cubic norm
        # Factors: (lhs, rel, rhs) where rel combines r_h and r_t
        factors = (
            torch.sqrt(h ** 2 + 1e-10),      # (batch, rank) - head entities
            torch.sqrt(r_h ** 2 + r_t ** 2 + 1e-10),  # (batch, rank) - relations
            torch.sqrt(t ** 2 + 1e-10)       # (batch, rank) - tail entities
        )
        
        # Return time embeddings for temporal regularization (Lambda3)
        # m is the continuous time embedding: (batch, rank)
        return scores, factors, m
    
    def forward_over_time(self, x: torch.Tensor):
        """
        Forward pass over all timestamps (for temporal evaluation).
        
        Args:
            x: Tensor of shape (batch_size, 3) containing [head, relation, tail]
        Returns:
            scores: (batch, n_timestamps) scoring across all timestamps
        """
        # This would require all normalized timestamps
        # For now, raise NotImplementedError
        raise NotImplementedError("forward_over_time requires normalized timestamp array")
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """Get right-hand side entities for ranking."""
        return self.entity_embeddings.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
    
    def get_queries(self, queries: torch.Tensor):
        """
        Compute query embeddings for ranking.
        
        Args:
            queries: (batch, 4) tensor [head, relation, tail, time]
        Returns:
            Query embeddings for scoring against all entities
        """
        h = self.entity_embeddings(queries[:, 0].long())
        r_h = self.relation_head(queries[:, 1].long())
        r_t = self.relation_tail(queries[:, 1].long())
        
        # Get continuous time embedding
        time_continuous = queries[:, 3].float()
        m = self.time_encoder(time_continuous)
        
        # Get gating
        alpha = torch.sigmoid(self.alpha(queries[:, 1].long()))
        gate = alpha * m + (1 - alpha)
        
        # For ranking, we need to return query that will be compared with all entities
        # This is simplified - may need adjustment based on exact ranking protocol
        query = (h * r_h) * gate
        
        return query

