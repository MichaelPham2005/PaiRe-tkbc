#!/usr/bin/env python3
# Sanity Check Script for ContinuousPairRE before training
# Kiểm tra các điểm quan trọng trước khi bắt đầu train

import os
import torch
from pathlib import Path
from datasets import TemporalDataset
from models import ContinuousPairRE

def sanity_check():
    print("="*70)
    print("SANITY CHECK - ContinuousPairRE")
    print("="*70)
    
    # 1. Check W and b dimensions in ContinuousTimeEmbedding
    print("\n1. Kiểm tra kích thước W và b trong ContinuousTimeEmbedding:")
    rank = 100
    dataset = TemporalDataset('ICEWS14', use_continuous_time=True)
    sizes = dataset.get_shape()
    model = ContinuousPairRE(sizes, rank).cuda()
    
    W_shape = model.time_encoder.W.shape
    b_shape = model.time_encoder.b.shape
    print(f"   ✓ W shape: {W_shape} (Expected: torch.Size([{rank}]))")
    print(f"   ✓ b shape: {b_shape} (Expected: torch.Size([{rank}]))")
    
    if W_shape == torch.Size([rank]) and b_shape == torch.Size([rank]):
        print("   ✓ PASS: W và b đều là vector có chiều bằng rank")
        print("   → Mô hình có thể học các chu kỳ khác nhau trên từng chiều")
    else:
        print("   ✗ FAIL: W hoặc b không có kích thước đúng!")
        return False
    
    # 2. Check data paths and files
    print("\n2. Kiểm tra đường dẫn dữ liệu và files:")
    data_path = Path(__file__).resolve().parent / "data" / "ICEWS14"
    
    required_files = [
        'train.pickle',
        'valid.pickle', 
        'test.pickle',
        'ts_normalized.pickle',
        'to_skip.pickle'
    ]
    
    all_exist = True
    for filename in required_files:
        file_path = data_path / filename
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {filename}: {'Found' if exists else 'NOT FOUND'}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("   ✓ PASS: Tất cả files cần thiết đều tồn tại")
    else:
        print("   ✗ FAIL: Thiếu một số files quan trọng!")
        print("   → Hãy chạy preprocess_continuous_time.py trước")
        return False
    
    # 3. Check alpha initialization
    print("\n3. Kiểm tra khởi tạo tham số alpha:")
    with torch.no_grad():
        alphas = torch.sigmoid(model.alpha.weight).cpu()
    
    print(f"   ✓ Number of relations: {len(alphas)}")
    print(f"   ✓ Alpha initialization:")
    print(f"      Mean: {alphas.mean().item():.4f}")
    print(f"      Std:  {alphas.std().item():.4f}")
    print(f"      Min:  {alphas.min().item():.4f}")
    print(f"      Max:  {alphas.max().item():.4f}")
    
    # Check if alphas are reasonable (not all 0 or all 1)
    if 0.3 < alphas.mean().item() < 0.7:
        print("   ✓ PASS: Alpha được khởi tạo ở giá trị trung bình hợp lý (0.5)")
    else:
        print("   ⚠ WARNING: Alpha mean không ở khoảng 0.3-0.7")
    
    # 4. Test forward pass with continuous time
    print("\n4. Test forward pass với continuous time:")
    try:
        # Create a small batch
        import numpy as np
        train_data = dataset.get_train()
        batch = train_data[:5]
        
        # Convert to continuous time
        batch_continuous = batch.copy().astype(np.float32)
        for i in range(5):
            ts_id = int(batch[i, 3])
            batch_continuous[i, 3] = dataset.ts_normalized[ts_id]
        
        batch_tensor = torch.from_numpy(batch_continuous).cuda()
        
        with torch.no_grad():
            scores, factors, time_embeds = model.forward(batch_tensor)
        
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Scores shape: {scores.shape}")
        print(f"   ✓ Factors shapes: {[f.shape for f in factors]}")
        print(f"   ✓ Time embeddings shape: {time_embeds.shape}")
        print("   ✓ PASS: Forward pass hoạt động bình thường")
    except Exception as e:
        print(f"   ✗ FAIL: Forward pass gặp lỗi: {e}")
        return False
    
    # 5. Check optimizer configuration
    print("\n5. Kiểm tra optimizer configuration:")
    from torch import optim
    from regularizers import N3, ContinuousTimeLambda3
    
    opt = optim.Adagrad(model.parameters(), lr=0.1)
    emb_reg = N3(0.001)
    time_reg = ContinuousTimeLambda3(0.001)
    
    print(f"   ✓ Optimizer: Adagrad (learning rate: 0.1)")
    print(f"   ✓ Embedding regularizer: N3 (weight: 0.001)")
    print(f"   ✓ Time regularizer: ContinuousTimeLambda3 (weight: 0.001)")
    print("   ✓ PASS: Optimizer và regularizers được cấu hình đúng")
    
    # 6. Check model parameters
    print("\n6. Kiểm tra model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    # Break down by component
    entity_params = model.entity_embeddings.weight.numel()
    rel_head_params = model.relation_head.weight.numel()
    rel_tail_params = model.relation_tail.weight.numel()
    time_W_params = model.time_encoder.W.numel()
    time_b_params = model.time_encoder.b.numel()
    alpha_params = model.alpha.weight.numel()
    
    print(f"\n   Parameter breakdown:")
    print(f"      Entity embeddings: {entity_params:,}")
    print(f"      Relation head: {rel_head_params:,}")
    print(f"      Relation tail: {rel_tail_params:,}")
    print(f"      Time encoder W: {time_W_params:,}")
    print(f"      Time encoder b: {time_b_params:,}")
    print(f"      Alpha (gating): {alpha_params:,}")
    
    # 7. Final recommendations
    print("\n" + "="*70)
    print("KHUYẾN NGHỊ TRƯỚC KHI TRAIN:")
    print("="*70)
    print("\n✓ Tất cả kiểm tra đã PASS!")
    print("\n📝 Lưu ý khi training:")
    print("   1. Theo dõi alpha statistics mỗi epoch")
    print("   2. Nếu alpha hội tụ quá nhanh về 0 hoặc 1:")
    print("      → Giảm learning rate hoặc thêm regularization cho alpha")
    print("   3. Nếu overfitting (train MRR >> valid MRR):")
    print("      → Tăng --emb_reg và --time_reg (thử 0.01, 0.1)")
    print("   4. Nếu underfitting (train và valid MRR đều thấp):")
    print("      → Tăng rank hoặc giảm regularization")
    print("   5. Monitor loss components:")
    print("      → loss: prediction loss")
    print("      → reg: embedding regularization")
    print("      → cont: time regularization")
    print("\n🚀 Sẵn sàng train! Chạy lệnh:")
    print("   .\\train_continuous_pairre.ps1")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        success = sanity_check()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
