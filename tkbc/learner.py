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
from optimizers import TKBCOptimizer, IKBCOptimizer, ContinuousTimeOptimizer
from models import ComplEx, TComplEx, TNTComplEx, ContinuousPairRE
from regularizers import N3, Lambda3, ContinuousTimeLambda3, ContinuitySmoothness, AlphaPolarization

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'ContinuousPairRE'
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
    '--smoothness_reg', default=0.0, type=float,
    help="Continuity smoothness regularizer for W and b (ContinuousPairRE only) - DISABLED by default"
)
parser.add_argument(
    '--alpha_reg', default=0., type=float,
    help="Alpha polarization regularizer (push toward 0 or 1, ContinuousPairRE only)"
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


args = parser.parse_args()

# Use continuous time for ContinuousPairRE model
use_continuous = (args.model == 'ContinuousPairRE')
dataset = TemporalDataset(args.dataset, use_continuous_time=use_continuous)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'ContinuousPairRE': ContinuousPairRE(sizes, args.rank),
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
    print(f"Smoothness regularization (W/b): {args.smoothness_reg}")
    print(f"Alpha polarization regularization: {args.alpha_reg}")
print(f"Checkpoint directory: {checkpoint_dir}")
print("="*70 + "\n")

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
# Use ContinuousTimeLambda3 for ContinuousPairRE, Lambda3 for others
time_reg = ContinuousTimeLambda3(args.time_reg) if args.model == 'ContinuousPairRE' else Lambda3(args.time_reg)
# Add smoothness regularizer for continuous time models
smoothness_reg = ContinuitySmoothness(args.smoothness_reg) if args.model == 'ContinuousPairRE' else None
# Add alpha polarization regularizer
alpha_reg = AlphaPolarization(args.alpha_reg) if args.model == 'ContinuousPairRE' else None

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
    elif args.model == 'ContinuousPairRE':
        # Use continuous time optimizer for ContinuousPairRE
        optimizer = ContinuousTimeOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size,
            smoothness_regularizer=smoothness_reg,
            alpha_regularizer=alpha_reg,
            loss_type=args.loss,
            margin=args.margin,
            adversarial_temperature=args.adversarial_temperature
        )
        optimizer.epoch(examples)
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
        
        # Print alpha statistics for ContinuousPairRE to monitor temporal gating
        if args.model == 'ContinuousPairRE':
            with torch.no_grad():
                alphas = torch.sigmoid(model.alpha.weight).cpu()
                print(f"\nAlpha Statistics:")
                print(f"  Mean: {alphas.mean().item():.4f}")
                print(f"  Std:  {alphas.std().item():.4f}")
                print(f"  Min:  {alphas.min().item():.4f}")
                print(f"  Max:  {alphas.max().item():.4f}")
                # Count static vs dynamic relations
                static_count = (alphas < 0.3).sum().item()
                dynamic_count = (alphas > 0.7).sum().item()
                print(f"  Static relations (α<0.3): {static_count}/{len(alphas)}")
                print(f"  Dynamic relations (α>0.7): {dynamic_count}/{len(alphas)}")
        
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
            if args.model == 'ContinuousPairRE':
                with torch.no_grad():
                    alphas = torch.sigmoid(model.alpha.weight).cpu()
                    f.write(f"Alpha Statistics:\n")
                    f.write(f"  Mean: {alphas.mean().item():.4f}, ")
                    f.write(f"Std: {alphas.std().item():.4f}, ")
                    f.write(f"Min: {alphas.min().item():.4f}, ")
                    f.write(f"Max: {alphas.max().item():.4f}\n")
                    static_count = (alphas < 0.3).sum().item()
                    dynamic_count = (alphas > 0.7).sum().item()
                    f.write(f"  Static (a<0.3): {static_count}/{len(alphas)}, ")
                    f.write(f"Dynamic (a>0.7): {dynamic_count}/{len(alphas)}\n")
            
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
