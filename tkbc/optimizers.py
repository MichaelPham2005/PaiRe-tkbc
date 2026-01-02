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
            smoothness_regularizer: Regularizer = None
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.smoothness_regularizer = smoothness_regularizer
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
                
                # Add smoothness regularization on W and b
                l_smooth = torch.zeros_like(l_reg)
                if self.smoothness_regularizer is not None and hasattr(self.model, 'time_encoder'):
                    W = self.model.time_encoder.W
                    b = self.model.time_encoder.b
                    l_smooth = self.smoothness_regularizer.forward(W, b)
                
                l = l_fit + l_reg + l_time + l_smooth

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
<<<<<<< Updated upstream
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}',
                    smooth=f'{l_smooth.item():.4f}'
                )
=======
                
                # Bước A & B: Detailed diagnostics every 50 batches
                batch_idx = (b_begin - self.batch_size) // self.batch_size
                if batch_idx % 50 == 0 and self.time_discrimination_weight > 0:
                    # Step A: Check raw vs weighted
                    raw_value = raw_t_disc.item() if isinstance(raw_t_disc, torch.Tensor) else 0.0
                    weighted_value = l_time_disc.item() if isinstance(l_time_disc, torch.Tensor) else 0.0
                    
                    # Step B: Check backprop
                    requires_grad = raw_t_disc.requires_grad if isinstance(raw_t_disc, torch.Tensor) else False
                    has_grad_fn = raw_t_disc.grad_fn is not None if isinstance(raw_t_disc, torch.Tensor) else False
                    ratio = weighted_value / (l.item() + 1e-9)
                    
                    print(f"\n[Batch {batch_idx}] TIME DISCRIMINATION DIAGNOSTICS:")
                    print(f"  Step A - Loss values:")
                    print(f"    raw_t_disc (before weight): {raw_value:.6f}")
                    print(f"    weighted_t_disc (after x lambda): {weighted_value:.6f}")
                    print(f"    lambda_time: {self.time_discrimination_weight}")
                    print(f"    Expected weighted: {raw_value * self.time_discrimination_weight:.6f}")
                    print(f"  Step B - Backprop check:")
                    print(f"    requires_grad: {requires_grad}")
                    print(f"    has_grad_fn: {has_grad_fn}")
                    print(f"    ratio (t_disc/total_loss): {ratio:.4f}")
                    print(f"  Step C - Score difference:")
                    print(f"    mean(score_pos - score_neg): {score_diff_mean:.6f}")
                    print(f"    std(score_pos - score_neg): {score_diff_std:.6f}")
                    print(f"    tau collision rate: {tau_collision_rate:.4f}")
                    print(f"  Total loss: {l.item():.6f} = fit:{l_fit.item():.4f} + reg:{l_reg.item():.4f} + t_param:{l_time_param.item():.4f} + t_disc:{weighted_value:.4f}")
                
                bar.set_postfix(postfix_dict)
        
        # Increment epoch counter after processing all batches
        self.current_epoch += 1


class GEPairREOptimizer(object):
    """
    Optimizer for GE-PairRE (Gaussian Evolution PairRE).
    
    Implements 2-stage training:
    - Stage 1 (Warm-up, epochs 1-5): Freeze Gaussian params, train static only, λ_time = 0
    - Stage 2 (Dynamic, epochs 6+): Unfreeze Gaussian, enable temporal loss with gradual increase
    
    Hard temporal discrimination: τ⁻ = τ⁺ ± Unif({1,2}) · T⁻¹
    """
    def __init__(
            self, 
            model, 
            emb_regularizer: Regularizer,
            temporal_regularizer: Regularizer,
            amplitude_regularizer: Regularizer,
            width_regularizer: Regularizer,
            smoothness_regularizer: Regularizer,  # NEW!
            sigma_bound_regularizer: Regularizer,  # NEW!
            optimizer: optim.Optimizer,
            dataset: TemporalDataset,
            batch_size: int = 256,
            margin: float = 0.3,
            warmup_epochs: int = 5,
            max_lambda_time: float = 1.0,
            gradient_accumulation_steps: int = 1,
            verbose: bool = True
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.amplitude_regularizer = amplitude_regularizer
        self.width_regularizer = width_regularizer
        self.smoothness_regularizer = smoothness_regularizer  # NEW!
        self.sigma_bound_regularizer = sigma_bound_regularizer  # NEW!
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.max_lambda_time = max_lambda_time
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.verbose = verbose
        self.current_epoch = 0
        
        if not hasattr(dataset, 'ts_normalized') or dataset.ts_normalized is None:
            raise ValueError("Dataset must have continuous time mappings loaded")
        
        # DON'T freeze here - let epoch() handle freeze/unfreeze based on stage
        # This allows optimizer to be recreated each epoch without breaking training
    
    def _freeze_gaussian_params(self):
        """Freeze Gaussian parameters (A, mu, s, W_trend) during warm-up."""
        if hasattr(self.model, 'time_encoder'):
            self.model.time_encoder.A.requires_grad = False
            self.model.time_encoder.mu.requires_grad = False
            self.model.time_encoder.s.requires_grad = False
            # Also freeze global trend
            if hasattr(self.model.time_encoder, 'W_trend'):
                self.model.time_encoder.W_trend.requires_grad = False
    
    def _unfreeze_gaussian_params(self):
        """Unfreeze Gaussian parameters for dynamic training."""
        if hasattr(self.model, 'time_encoder'):
            self.model.time_encoder.A.requires_grad = True
            self.model.time_encoder.mu.requires_grad = True
            self.model.time_encoder.s.requires_grad = True
            # Also unfreeze global trend
            if hasattr(self.model.time_encoder, 'W_trend'):
                self.model.time_encoder.W_trend.requires_grad = True
    
    def get_lambda_time(self):
        """
        Get current λ_time based on training stage.
        Stage 1 (epochs 1-warmup_epochs): 0
        Stage 2: Gradually increase from 0.1 to max_lambda_time
        """
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        else:
            # Linear ramp: 0.1 → max_lambda_time over next 10 epochs
            progress = min((self.current_epoch - self.warmup_epochs) / 10.0, 1.0)
            return 0.1 + progress * (self.max_lambda_time - 0.1)
    
    def sample_negative_time(self, pos_tau: torch.Tensor, pos_time_ids: torch.Tensor):
        """
        Sample negative times using hard temporal discrimination strategy.
        τ⁻ = τ⁺ ± Unif({1,2}) · T⁻¹
        
        Args:
            pos_tau: (batch,) positive continuous times
            pos_time_ids: (batch,) positive timestamp IDs
        
        Returns:
            neg_tau: (batch,) negative continuous times
            neg_time_ids: (batch,) negative timestamp IDs
        """
        batch_size = pos_tau.shape[0]
        n_times = len(self.dataset.ts_normalized)
        
        # Sample offset: ±1 or ±2 timesteps
        offsets = torch.randint(1, 3, (batch_size,)).cpu()  # {1, 2}
        signs = torch.randint(0, 2, (batch_size,)).cpu() * 2 - 1  # {-1, 1}
        time_offsets = offsets * signs
        
        # Compute negative time IDs
        neg_time_ids = pos_time_ids.cpu() + time_offsets
        
        # Handle boundary cases: resample if out of bounds
        out_of_bounds = (neg_time_ids < 0) | (neg_time_ids >= n_times)
        n_resample = out_of_bounds.sum().item()
        if n_resample > 0:
            neg_time_ids[out_of_bounds] = torch.randint(0, n_times, (n_resample,))
        
        # Convert to continuous times
        neg_tau = torch.tensor(
            [self.dataset.ts_normalized[int(tid)] for tid in neg_time_ids.numpy()],
            dtype=torch.float32,
            device=pos_tau.device
        )
        
        return neg_tau, neg_time_ids
    
    def epoch(self, examples: torch.LongTensor, epoch_num: int = None):
        # Update current epoch if provided (for when optimizer is recreated each epoch)
        if epoch_num is not None:
            self.current_epoch = epoch_num
        
        # Freeze/unfreeze based on current stage
        if self.current_epoch < self.warmup_epochs:
            # Warm-up stage: freeze Gaussian parameters
            self._freeze_gaussian_params()
        else:
            # Dynamic stage: unfreeze Gaussian parameters
            self._unfreeze_gaussian_params()
            # Print transition message only once
            if self.current_epoch == self.warmup_epochs and self.verbose:
                print(f"\n[Epoch {self.current_epoch + 1}] TRANSITIONING TO DYNAMIC TRAINING")
                print(f"  - Unfroze Gaussian parameters (A, μ, s)")
                print(f"  - Enabling temporal discrimination loss")
        
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        lambda_time = self.get_lambda_time()
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            stage = "WARMUP" if self.current_epoch < self.warmup_epochs else "DYNAMIC"
            bar.set_description(f'[{stage}] train loss')
            b_begin = 0
            accumulation_step = 0
            
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size]
                
                # Convert to continuous time
                batch_with_continuous_time = input_batch.clone().float()
                timestamp_ids = input_batch[:, 3].cpu().numpy()
                continuous_times = torch.tensor(
                    [self.dataset.ts_normalized[int(tid)] for tid in timestamp_ids],
                    dtype=torch.float32
                )
                batch_with_continuous_time[:, 3] = continuous_times
                batch_with_continuous_time = batch_with_continuous_time.cuda()
                
                # Forward pass
                predictions, factors, delta_e = self.model.forward(batch_with_continuous_time)
                truth = input_batch[:, 2].cuda()
                
                # Primary loss (link prediction)
                l_fit = loss_fn(predictions, truth)
                
                # Hard temporal discrimination loss
                l_time_disc = torch.zeros_like(l_fit)
                accuracy = 0.0
                margin_mean = 0.0
                
                if lambda_time > 0 and hasattr(self.model, 'score'):
                    pos_tau = batch_with_continuous_time[:, 3]
                    pos_time_ids = input_batch[:, 3]
                    
                    # Sample hard negatives
                    neg_tau, neg_time_ids = self.sample_negative_time(pos_tau, pos_time_ids)
                    
                    # Create negative batch
                    batch_neg_time = batch_with_continuous_time.clone()
                    batch_neg_time[:, 3] = neg_tau
                    
                    # Compute scores
                    score_pos = self.model.score(batch_with_continuous_time)
                    score_neg = self.model.score(batch_neg_time)
                    
                    # Hard temporal discrimination loss: ReLU(γ - (φ⁺ - φ⁻))
                    margin_values = score_pos - score_neg
                    raw_t_disc = torch.relu(self.margin - margin_values).mean()
                    l_time_disc = lambda_time * raw_t_disc
                    
                    # Metrics
                    accuracy = (score_pos > score_neg).float().mean().item()
                    margin_mean = margin_values.mean().item()
                
                # Regularization
                l_reg = self.emb_regularizer.forward(factors)
                
                # Amplitude decay regularization
                l_amp = torch.zeros_like(l_reg)
                if hasattr(self.model, 'time_encoder'):
                    l_amp = self.amplitude_regularizer.forward(self.model.time_encoder.A)
                
                # Width penalty regularization (upper bound)
                l_width = torch.zeros_like(l_reg)
                if hasattr(self.model, 'time_encoder'):
                    sigma = self.model.time_encoder.get_sigma()
                    l_width = self.width_regularizer.forward(sigma)
                
                # Sigma lower bound regularization (NEW!)
                l_sigma_bound = torch.zeros_like(l_reg)
                if hasattr(self.model, 'time_encoder'):
                    sigma = self.model.time_encoder.get_sigma()
                    l_sigma_bound = self.sigma_bound_regularizer.forward(sigma)
                
                # Temporal smoothness regularization (NEW!)
                l_smooth = torch.zeros_like(l_reg)
                if self.current_epoch >= self.warmup_epochs and hasattr(self.model, 'time_encoder'):
                    # Only apply during dynamic stage
                    rel_id = input_batch[:, 1].long()
                    tau = pos_tau if self.current_epoch >= self.warmup_epochs else input_batch[:, 3].float()
                    l_smooth = self.smoothness_regularizer.forward(
                        self.model.time_encoder, rel_id, tau
                    )
                
                # Total loss (scaled by accumulation steps)
                l = (l_fit + l_reg + l_amp + l_width + l_sigma_bound + l_smooth + l_time_disc) / self.gradient_accumulation_steps
                
                # Backward
                l.backward()
                
                # Update weights only after accumulating gradients
                accumulation_step += 1
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                
                # Show metrics with proper scaling
                actual_loss = l.item() * self.gradient_accumulation_steps
                bar.set_postfix({
                    'loss': f'{l_fit.item():.4f}',
                    'reg': f'{l_reg.item():.0f}',
                    'amp': f'{l_amp.item():.4f}',
                    'width': f'{l_width.item():.4f}',
                    'σ_bound': f'{l_sigma_bound.item():.4f}',  # NEW!
                    'smooth': f'{l_smooth.item():.4f}',  # NEW!
                    't_disc': f'{l_time_disc.item():.4f}',
                    'λ_t': f'{lambda_time:.2f}',
                    'acc': f'{accuracy:.3f}',
                    'Δφ': f'{margin_mean:.3f}'
                })
            
            # Final optimizer step if there are remaining gradients
            if accumulation_step % self.gradient_accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.current_epoch += 1
>>>>>>> Stashed changes
