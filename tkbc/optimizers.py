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
    """Optimizer for continuous time models with relation-conditioned time encoding."""
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset,
            batch_size: int = 256, verbose: bool = True,
            time_param_regularizer: Regularizer = None,
            loss_type: str = 'cross_entropy',
            margin: float = 6.0,
            adversarial_temperature: float = 1.0,
            time_discrimination_weight: float = 0.1
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.time_param_regularizer = time_param_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_type = loss_type
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        self.time_discrimination_weight = time_discrimination_weight
        self.current_epoch = 0  # Track current epoch
        
        if not hasattr(dataset, 'ts_normalized') or dataset.ts_normalized is None:
            raise ValueError("Dataset must have continuous time mappings loaded")

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        
        # Define loss function based on type
        if self.loss_type == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
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

                if self.loss_type == 'cross_entropy':
                    l_fit = loss_fn(predictions, truth)
                elif self.loss_type == 'self_adversarial':
                    # Self-Adversarial Negative Sampling Loss
                    pos_scores = predictions.gather(1, truth.view(-1, 1))
                    
                    # Create mask for negatives
                    batch_size_local, n_entities = predictions.shape
                    neg_mask = torch.ones((batch_size_local, n_entities), dtype=torch.bool, device=predictions.device)
                    neg_mask.scatter_(1, truth.view(-1, 1), 0)
                    
                    # Negative scores
                    neg_scores = predictions[neg_mask].view(batch_size_local, -1)
                    
                    # Self-Adversarial Weights for negatives
                    neg_weights = torch.softmax(neg_scores * self.adversarial_temperature, dim=1).detach()
                    
                    # Compute Loss
                    loss_pos = -torch.log(torch.sigmoid(self.margin + pos_scores) + 1e-9).mean()
                    loss_neg = -torch.sum(
                        neg_weights * torch.log(torch.sigmoid(-neg_scores - self.margin) + 1e-9),
                        dim=1
                    ).mean()
                    
                    l_fit = (loss_pos + loss_neg) / 2
                
                # Time Discrimination Loss (CRITICAL FOR NEW MODEL)
                l_time_disc = torch.zeros_like(l_fit)
                if self.time_discrimination_weight > 0 and hasattr(self.model, 'score'):
                    # Sample negative times for each positive
                    n_times = len(self.dataset.ts_normalized)
                    neg_time_ids = torch.randint(0, n_times, (batch_with_continuous_time.shape[0],))
                    neg_continuous_times = torch.tensor(
                        [self.dataset.ts_normalized[int(tid)] for tid in neg_time_ids.cpu().numpy()],
                        dtype=torch.float32,
                        device=batch_with_continuous_time.device
                    )
                    
                    # Create negative batch with wrong times
                    batch_neg_time = batch_with_continuous_time.clone()
                    batch_neg_time[:, 3] = neg_continuous_times
                    
                    # Compute scores
                    score_pos = self.model.score(batch_with_continuous_time)  # (batch,)
                    score_neg_time = self.model.score(batch_neg_time)  # (batch,)
                    
                    # Time discrimination loss: -log σ(φ(+) - φ(-))
                    l_time_disc = -torch.log(torch.sigmoid(score_pos - score_neg_time) + 1e-9).mean()
                    l_time_disc = self.time_discrimination_weight * l_time_disc
                
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                
                # Time parameter regularization (light)
                l_time_param = torch.zeros_like(l_reg)
                if self.time_param_regularizer is not None and hasattr(self.model, 'time_encoder'):
                    a_r = self.model.time_encoder.a_r
                    A_r = self.model.time_encoder.A_r
                    l_time_param = self.time_param_regularizer.forward(a_r, A_r)
                
                l = l_fit + l_reg + l_time + l_time_param + l_time_disc

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.4f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}',
                    t_param=f'{l_time_param.item():.4f}',
                    t_disc=f'{l_time_disc.item():.4f}'
                )
        
        # Increment epoch counter after processing all batches
        self.current_epoch += 1
