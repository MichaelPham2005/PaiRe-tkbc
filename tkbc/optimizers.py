# Copyright (c) Facebook, Inc. and its affiliates.

import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset


class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}'
                )


class IKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                time_range = actual_examples[b_begin:b_begin + self.batch_size].cuda()

                ## RHS Prediction loss
                sampled_time = (
                        torch.rand(time_range.shape[0]).cuda() * (time_range[:, 4] - time_range[:, 3]).float() +
                        time_range[:, 3].float()
                ).round().long()
                with_time = torch.cat((time_range[:, 0:3], sampled_time.unsqueeze(1)), 1)

                predictions, factors, time = self.model.forward(with_time)
                truth = with_time[:, 2]

                l_fit = loss(predictions, truth)

                ## Time prediction loss (ie cross entropy over time)
                time_loss = 0.
                if self.model.has_time():
                    filtering = ~(
                        (time_range[:, 3] == 0) *
                        (time_range[:, 4] == (self.dataset.n_timestamps - 1))
                    ) # NOT no begin and no end
                    these_examples = time_range[filtering, :]
                    truth = (
                            torch.rand(these_examples.shape[0]).cuda() * (these_examples[:, 4] - these_examples[:, 3]).float() +
                            these_examples[:, 3].float()
                    ).round().long()
                    time_predictions = self.model.forward_over_time(these_examples[:, :3].cuda().long())
                    time_loss = loss(time_predictions, truth.cuda())

                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time + time_loss

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(with_time.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    loss_time=f'{time_loss if type(time_loss) == float else time_loss.item() :.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.4f}'
                )


class ContinuousTimeOptimizer(object):
    """Optimizer for continuous time models."""
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset,
            batch_size: int = 256, verbose: bool = True,
            smoothness_regularizer: Regularizer = None,
            alpha_regularizer: Regularizer = None
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.smoothness_regularizer = smoothness_regularizer
        self.alpha_regularizer = alpha_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        
        if not hasattr(dataset, 'ts_normalized') or dataset.ts_normalized is None:
            raise ValueError("Dataset must have continuous time mappings loaded")

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ]
                
                # Convert timestamp IDs to continuous normalized values
                batch_with_continuous_time = input_batch.clone().float()
                timestamp_ids = input_batch[:, 3].cpu().numpy()
                continuous_times = torch.tensor(
                    [self.dataset.ts_normalized[int(tid)] for tid in timestamp_ids],
                    dtype=torch.float32
                )
                batch_with_continuous_time[:, 3] = continuous_times
                batch_with_continuous_time = batch_with_continuous_time.cuda()
                
                # Forward pass
                predictions, factors, time = self.model.forward(batch_with_continuous_time)
                truth = input_batch[:, 2].cuda()

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                
                # Add smoothness regularization on W, b, and linear layer
                l_smooth = torch.zeros_like(l_reg)
                if self.smoothness_regularizer is not None and hasattr(self.model, 'time_encoder'):
                    W = self.model.time_encoder.W
                    b = self.model.time_encoder.b
                    linear_weight = self.model.time_encoder.linear.weight if hasattr(self.model.time_encoder, 'linear') else None
                    l_smooth = self.smoothness_regularizer.forward(W, b, linear_weight)
                
                # Add alpha polarization regularization
                l_alpha = torch.zeros_like(l_reg)
                if self.alpha_regularizer is not None and hasattr(self.model, 'alpha'):
                    alpha_weights = self.model.alpha.weight
                    l_alpha = self.alpha_regularizer.forward(alpha_weights)
                
                l = l_fit + l_reg + l_time + l_smooth + l_alpha

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}',
                    smooth=f'{l_smooth.item():.4f}',
                    alpha=f'{l_alpha.item():.4f}'
                )
