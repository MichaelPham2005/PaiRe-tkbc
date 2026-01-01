# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim
import os
import json
from pathlib import Path

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer, ContinuousTimeOptimizer, GEPairREOptimizer
from models import ComplEx, TComplEx, TNTComplEx, ContinuousPairRE, GEPairRE
from regularizers import N3, Lambda3, ContinuousTimeLambda3, TimeParameterRegularizer, AmplitudeDecayRegularizer, WidthPenaltyRegularizer

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'ContinuousPairRE', 'GEPairRE'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--time_param_reg', default=0., type=float,
    help="Time parameter regularizer for a_r and A_r (ContinuousPairRE only)"
)
parser.add_argument(
    '--time_discrimination_weight', default=0.1, type=float,
    help="Weight for time discrimination loss (ContinuousPairRE only)"
)
parser.add_argument(
    '--K_frequencies', default=16, type=int,
    help="Number of frequency components for relation-conditioned time encoder (ContinuousPairRE only)"
)
parser.add_argument(
    '--beta', default=0.5, type=float,
    help="Beta parameter for residual gate (ContinuousPairRE only)"
)
parser.add_argument(
    '--time_scale', default=1.0, type=float,
    help="Time normalization scale (1.0 for [0,1], 10.0 for [0,10])"
)
parser.add_argument(
    '--loss', default='cross_entropy', choices=['cross_entropy', 'self_adversarial'],
    help="Loss function to use. 'cross_entropy' is standard Softmax. 'self_adversarial' is PairRE/RotatE style."
)
parser.add_argument(
    '--margin', default=6.0, type=float,
    help="Margin gamma for Self-Adversarial Loss"
)
parser.add_argument(
    '--adversarial_temperature', default=1.0, type=float,
    help="Temperature alpha for self-adversarial negative sampling"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
# GE-PairRE specific arguments
parser.add_argument(
    '--K_gaussians', default=8, type=int,
    help="Number of Gaussian pulses per relation (GEPairRE only)"
)
parser.add_argument(
    '--sigma_max', default=0.2, type=float,
    help="Maximum width for Gaussian pulses (GEPairRE only)"
)
parser.add_argument(
    '--warmup_epochs', default=5, type=int,
    help="Number of warm-up epochs to freeze Gaussian params (GEPairRE only)"
)
parser.add_argument(
    '--amplitude_reg', default=0.001, type=float,
    help="Amplitude decay regularization weight (GEPairRE only)"
)
parser.add_argument(
    '--width_reg', default=0.01, type=float,
    help="Width penalty regularization weight (GEPairRE only)"
)
parser.add_argument(
    '--max_lambda_time', default=1.0, type=float,
    help="Maximum value for λ_time in dynamic stage (GEPairRE only)"
)


args = parser.parse_args()

# Use continuous time for ContinuousPairRE and GEPairRE models
use_continuous = (args.model in ['ContinuousPairRE', 'GEPairRE'])
dataset = TemporalDataset(args.dataset, use_continuous_time=use_continuous)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'ContinuousPairRE': ContinuousPairRE(sizes, args.rank, K=args.K_frequencies, beta=args.beta),
    'GEPairRE': GEPairRE(sizes, args.rank, K=args.K_gaussians, sigma_max=args.sigma_max),
}[args.model]
model = model.cuda()

# Setup checkpoint directory
checkpoint_dir = Path('models') / args.dataset / args.model
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save configuration
config = vars(args)
with open(checkpoint_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Setup training log file
log_file = checkpoint_dir / 'training_log.txt'
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"Training Log - {args.model} on {args.dataset}\n")
    f.write(f"{'='*70}\n")
    f.write(f"Configuration:\n")
    for key, value in config.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"{'='*70}\n\n")

# Print training configuration
print("\n" + "="*70)
print(f"TRAINING CONFIGURATION - {args.model} on {args.dataset}")
print("="*70)
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Rank: {args.rank}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.learning_rate}")
print(f"Max epochs: {args.max_epochs}")
print(f"Validation frequency: every {args.valid_freq} epochs")
print(f"Embedding regularization (N3): {args.emb_reg}")
print(f"Time regularization: {args.time_reg}")
if args.model == 'ContinuousPairRE':
    print(f"K frequencies: {args.K_frequencies}")
    print(f"Beta (residual gate): {args.beta}")
    print(f"Time parameter regularization: {args.time_param_reg}")
    print(f"Time discrimination weight: {args.time_discrimination_weight}")
    print(f"Time scale: [0, {args.time_scale}]")
elif args.model == 'GEPairRE':
    print(f"K Gaussians: {args.K_gaussians}")
    print(f"Sigma max: {args.sigma_max}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Amplitude regularization: {args.amplitude_reg}")
    print(f"Width regularization: {args.width_reg}")
    print(f"Max λ_time: {args.max_lambda_time}")
    print(f"Margin (hard temporal discrimination): {args.margin}")
print(f"Checkpoint directory: {checkpoint_dir}")
print("="*70 + "\n")

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
# Use ContinuousTimeLambda3 for ContinuousPairRE, Lambda3 for others
time_reg = ContinuousTimeLambda3(args.time_reg) if args.model == 'ContinuousPairRE' else Lambda3(args.time_reg)
# Add time parameter regularizer for continuous time models
from regularizers import TimeParameterRegularizer
time_param_reg = TimeParameterRegularizer(args.time_param_reg) if args.model == 'ContinuousPairRE' else None

# GE-PairRE specific regularizers
amplitude_reg = AmplitudeDecayRegularizer(args.amplitude_reg) if args.model == 'GEPairRE' else None
width_reg = WidthPenaltyRegularizer(args.width_reg, args.sigma_max) if args.model == 'GEPairRE' else None

# Track best validation MRR for saving best model
best_valid_mrr = 0.0

print("Starting training...")
print(f"Total training examples: {dataset.get_train().shape[0]:,}\n")

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)
    elif args.model == 'GEPairRE':
        # Use GE-PairRE optimizer with 2-stage training
        optimizer = GEPairREOptimizer(
            model, emb_reg, time_reg, amplitude_reg, width_reg, opt, dataset,
            batch_size=args.batch_size,
            margin=args.margin,
            warmup_epochs=args.warmup_epochs,
            max_lambda_time=args.max_lambda_time
        )
        optimizer.epoch(examples, epoch_num=epoch)
    elif args.model == 'ContinuousPairRE':
        # Use continuous time optimizer for ContinuousPairRE
        optimizer = ContinuousTimeOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size,
            time_param_regularizer=time_param_reg,
            loss_type=args.loss,
            margin=args.margin,
            adversarial_temperature=args.adversarial_temperature,
            time_discrimination_weight=args.time_discrimination_weight
        )
        optimizer.epoch(examples)
        
        # TEST B2: Track parameter updates every epoch
        if args.model == 'ContinuousPairRE' and hasattr(model, 'time_encoder'):
            with torch.no_grad():
                if hasattr(model, '_prev_params'):
                    delta_a_r = (model.time_encoder.a_r - model._prev_params['a_r']).norm().item()
                    delta_A_r = (model.time_encoder.A_r - model._prev_params['A_r']).norm().item()
                    delta_P_r = (model.time_encoder.P_r - model._prev_params['P_r']).norm().item()
                    delta_W_proj = (model.time_encoder.W_proj - model._prev_params['W_proj']).norm().item()
                    
                    print(f"[Epoch {epoch+1}] Parameter Updates: Δa_r={delta_a_r:.2e}, ΔA_r={delta_A_r:.2e}, ΔP_r={delta_P_r:.2e}, ΔW_proj={delta_W_proj:.2e}")
                    
                    if delta_a_r < 1e-6 and delta_A_r < 1e-6:
                        print("  ⚠️  WARNING: Time parameters barely changing!")
                
                # Save current for next comparison
                model._prev_params = {
                    'a_r': model.time_encoder.a_r.clone(),
                    'A_r': model.time_encoder.A_r.clone(),
                    'P_r': model.time_encoder.P_r.clone(),
                    'W_proj': model.time_encoder.W_proj.clone()
                }
    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    # Run validation every valid_freq epochs (skip epoch 0)
    # Note: epoch starts from 0, so this runs at: 5, 10, 15, 20... and last epoch
    if (epoch > 0 and epoch % args.valid_freq == 0) or epoch == args.max_epochs - 1:
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{args.max_epochs} - VALIDATION (every {args.valid_freq} epochs)")
        print(f"{'='*70}")
        
        # Print time encoder statistics for ContinuousPairRE
        if args.model == 'ContinuousPairRE' and hasattr(model, 'time_encoder'):
            with torch.no_grad():
                a_r = model.time_encoder.a_r.cpu()
                A_r = model.time_encoder.A_r.cpu()
                print(f"\nTime Encoder Statistics:")
                print(f"  Beta (gate strength): {model.beta:.4f}")
                print(f"  Trend (a_r):")
                print(f"    Mean: {a_r.mean().item():.4f}, Std: {a_r.std().item():.4f}")
                print(f"    Range: [{a_r.min().item():.4f}, {a_r.max().item():.4f}]")
                print(f"  Amplitude (A_r):")
                print(f"    Mean: {A_r.mean().item():.4f}, Std: {A_r.std().item():.4f}")
                print(f"    Range: [{A_r.min().item():.4f}, {A_r.max().item():.4f}]")
        
        model.eval()
        with torch.no_grad():
            # Only evaluate on validation and train sets during training
            # Test set is reserved for final evaluation only
            if dataset.has_intervals():
                valid = dataset.eval(model, 'valid', -1)
                train = dataset.eval(model, 'train', 50000)
                print("\nMetrics:")
                print(f"  Valid: {valid}")
                print(f"  Train: {train}")
                valid_mrr = valid.get('MRR', 0.0)
                train_mrr = train.get('MRR', 0.0)
            else:
                valid = avg_both(*dataset.eval(model, 'valid', -1))
                train = avg_both(*dataset.eval(model, 'train', 50000))
                print("\nMetrics:")
                print(f"  Valid MRR: {valid['MRR']:.4f}, Hits@[1,3,10]: {valid['hits@[1,3,10]'].tolist()}")
                print(f"  Train MRR: {train['MRR']:.4f}, Hits@[1,3,10]: {train['hits@[1,3,10]'].tolist()}")
                valid_mrr = valid['MRR']
                train_mrr = train['MRR']
        
        # Log results to file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nEpoch {epoch + 1}/{args.max_epochs}\n")
            f.write(f"{'-'*70}\n")
            if args.model == 'ContinuousPairRE' and hasattr(model, 'time_encoder'):
                with torch.no_grad():
                    a_r = model.time_encoder.a_r.cpu()
                    A_r = model.time_encoder.A_r.cpu()
                    f.write(f"Time Encoder Statistics:\n")
                    f.write(f"  Beta: {model.beta:.4f}\n")
                    f.write(f"  Trend (a_r): Mean={a_r.mean().item():.4f}, Std={a_r.std().item():.4f}, ")
                    f.write(f"Range=[{a_r.min().item():.4f}, {a_r.max().item():.4f}]\n")
                    f.write(f"  Amplitude (A_r): Mean={A_r.mean().item():.4f}, Std={A_r.std().item():.4f}, ")
                    f.write(f"Range=[{A_r.min().item():.4f}, {A_r.max().item():.4f}]\n")
            
            if dataset.has_intervals():
                f.write(f"Valid: {valid}\n")
                f.write(f"Train: {train}\n")
            else:
                f.write(f"Valid MRR: {valid['MRR']:.4f}, Hits@[1,3,10]: {valid['hits@[1,3,10]'].tolist()}\n")
                f.write(f"Train MRR: {train['MRR']:.4f}, Hits@[1,3,10]: {train['hits@[1,3,10]'].tolist()}\n")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'valid_mrr': valid_mrr,
            'train_mrr': train_mrr,
            'config': config
        }
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        print(f"\n💾 Saved latest checkpoint to: {checkpoint_dir / 'latest.pt'}")
        
        # Save best model if validation improves
        if valid_mrr > best_valid_mrr:
            improvement = valid_mrr - best_valid_mrr
            best_valid_mrr = valid_mrr
            torch.save(checkpoint, checkpoint_dir / 'best_valid.pt')
            print(f"✨ New best validation MRR: {best_valid_mrr:.4f} (+{improvement:.4f})")
            print(f"💾 Saved best checkpoint to: {checkpoint_dir / 'best_valid.pt'}")
            
            # Log best model update to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f">>> NEW BEST MODEL at epoch {epoch + 1}: Valid MRR = {best_valid_mrr:.4f} <<<\n")
        
        print(f"{'='*70}\n")

# Final test evaluation with best model
print("\n" + "="*70)
print("TRAINING COMPLETED - FINAL TEST EVALUATION")
print("="*70)
print(f"\nLoading best model from: {checkpoint_dir / 'best_valid.pt'}")

# Load best model
best_checkpoint = torch.load(checkpoint_dir / 'best_valid.pt')
model.load_state_dict(best_checkpoint['model_state_dict'])
best_epoch = best_checkpoint['epoch']
best_valid_mrr = best_checkpoint['valid_mrr']

print(f"Best model from epoch {best_epoch} with Valid MRR: {best_valid_mrr:.4f}")
print("\nEvaluating on Test set (final evaluation)...")

model.eval()
with torch.no_grad():
    if dataset.has_intervals():
        test = dataset.eval(model, 'test', -1)
        valid = dataset.eval(model, 'valid', -1)
        print("\nFinal Results:")
        print(f"  Valid: {valid}")
        print(f"  Test:  {test}")
        test_mrr = test.get('MRR', 0.0)
    else:
        test = avg_both(*dataset.eval(model, 'test', -1))
        valid = avg_both(*dataset.eval(model, 'valid', -1))
        print("\nFinal Results:")
        print(f"  Valid MRR: {valid['MRR']:.4f}, Hits@[1,3,10]: {valid['hits@[1,3,10]'].tolist()}")
        print(f"  Test MRR:  {test['MRR']:.4f}, Hits@[1,3,10]: {test['hits@[1,3,10]'].tolist()}")
        test_mrr = test['MRR']

# Log final test results
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(f"\n{'='*70}\n")
    f.write(f"FINAL TEST EVALUATION (Best model from epoch {best_epoch})\n")
    f.write(f"{'='*70}\n")
    if dataset.has_intervals():
        f.write(f"Valid: {valid}\n")
        f.write(f"Test:  {test}\n")
    else:
        f.write(f"Valid MRR: {valid['MRR']:.4f}, Hits@[1,3,10]: {valid['hits@[1,3,10]'].tolist()}\n")
        f.write(f"Test MRR:  {test['MRR']:.4f}, Hits@[1,3,10]: {test['hits@[1,3,10]'].tolist()}\n")
    f.write(f"{'='*70}\n")

# Update best checkpoint with test results
best_checkpoint['test_mrr'] = test_mrr
torch.save(best_checkpoint, checkpoint_dir / 'best_valid.pt')

print(f"\n✅ Training and evaluation completed!")
print(f"📊 Final Test MRR: {test_mrr:.4f}")
print(f"📁 All results saved to: {checkpoint_dir}")
print("="*70)
