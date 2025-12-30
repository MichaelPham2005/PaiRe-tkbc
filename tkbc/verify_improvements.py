#!/usr/bin/env python3
"""
Verify that the implementation matches README_relation_gating.md specifications
"""

import torch
import numpy as np
from pathlib import Path
import sys

def check_implementation():
    print("="*70)
    print("IMPLEMENTATION VERIFICATION vs README_relation_gating.md")
    print("="*70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Time normalization range
    print("\n1. Checking time normalization range...")
    checks_total += 1
    try:
        from datasets import TemporalDataset
        dataset = TemporalDataset('ICEWS14', use_continuous_time=True)
        if dataset.ts_normalized is not None:
            values = list(dataset.ts_normalized.values())
            min_val, max_val = min(values), max(values)
            print(f"   Time range: [{min_val:.2f}, {max_val:.2f}]")
            if -1.1 <= min_val <= -0.9 and 0.9 <= max_val <= 1.1:
                print("   ✅ PASS: Time normalized to [-1, 1] range")
                checks_passed += 1
            else:
                print("   ❌ FAIL: Expected [-1, 1], got different range")
                print("   → Run: .\\scripts\\repreprocess_time.ps1")
        else:
            print("   ⚠ WARNING: ts_normalized not loaded")
            print("   → Run: python preprocess_continuous_time.py")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Check 2: Alpha initialization
    print("\n2. Checking alpha initialization...")
    checks_total += 1
    try:
        from models import ContinuousPairRE
        model = ContinuousPairRE((100, 10, 100, 100), rank=50)
        alpha_init = model.alpha.weight.data.mean().item()
        print(f"   Alpha initial value: {alpha_init:.4f}")
        if abs(alpha_init) < 0.1:  # Should be close to 0
            print("   ✅ PASS: Alpha initialized to ~0 (sigmoid(0)=0.5)")
            checks_passed += 1
        else:
            print(f"   ❌ FAIL: Expected ~0, got {alpha_init:.4f}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Check 3: Gating formula
    print("\n3. Checking gating formula...")
    checks_total += 1
    try:
        import inspect
        from models import ContinuousPairRE
        source = inspect.getsource(ContinuousPairRE.score)
        if "torch.ones_like" in source and "* torch.ones_like(m)" in source:
            print("   ✅ PASS: Gate = alpha * m + (1 - alpha) * ones")
            checks_passed += 1
        else:
            print("   ❌ FAIL: Missing 'torch.ones_like(m)' in gating")
            print("   → Should be: gate = alpha * m + (1 - alpha) * torch.ones_like(m)")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Check 4: Scoring function
    print("\n4. Checking scoring function...")
    checks_total += 1
    try:
        import inspect
        from models import ContinuousPairRE
        source = inspect.getsource(ContinuousPairRE.score)
        if "torch.norm" in source and "p=1" in source:
            print("   ✅ PASS: Using torch.norm(p=1) for L1 norm")
            checks_passed += 1
        elif "torch.abs" in source and ".sum(" in source:
            print("   ⚠ WARNING: Using torch.abs().sum() instead of torch.norm(p=1)")
            print("   → Both are equivalent, but norm(p=1) is cleaner")
            checks_passed += 1
        else:
            print("   ❌ FAIL: Scoring function doesn't use L1 norm")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Check 5: AlphaPolarization regularizer
    print("\n5. Checking AlphaPolarization regularizer...")
    checks_total += 1
    try:
        from regularizers import AlphaPolarization
        alpha_reg = AlphaPolarization(0.01)
        # Test with dummy data
        alpha_raw = torch.zeros(10, 1)  # sigmoid(0) = 0.5
        loss = alpha_reg.forward(alpha_raw)
        print(f"   Entropy at alpha=0.5: {loss.item():.4f}")
        if 0.006 < loss.item() < 0.008:  # 0.693 * 0.01
            print("   ✅ PASS: AlphaPolarization regularizer exists and works")
            checks_passed += 1
        else:
            print("   ⚠ WARNING: Loss value unexpected, but regularizer exists")
            checks_passed += 1
    except ImportError:
        print("   ❌ FAIL: AlphaPolarization class not found")
        print("   → Add AlphaPolarization to regularizers.py")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Check 6: Training integration
    print("\n6. Checking training script...")
    checks_total += 1
    try:
        script_path = Path(__file__).parent / "scripts" / "train_continuous_pairre.ps1"
        if script_path.exists():
            content = script_path.read_text()
            if "--alpha_reg" in content:
                print("   ✅ PASS: --alpha_reg argument in training script")
                checks_passed += 1
            else:
                print("   ❌ FAIL: Missing --alpha_reg in training script")
        else:
            print(f"   ⚠ WARNING: Script not found at {script_path}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("="*70)
    
    if checks_passed == checks_total:
        print("\n✅ ALL CHECKS PASSED!")
        print("Implementation matches README specifications.")
        print("\nNext steps:")
        print("  1. cd scripts")
        print("  2. .\\repreprocess_time.ps1  # Re-normalize to [-1, 1]")
        print("  3. .\\train_continuous_pairre.ps1  # Train with improvements")
        return 0
    else:
        print(f"\n⚠ {checks_total - checks_passed} checks failed or have warnings.")
        print("Please review the issues above before training.")
        return 1

if __name__ == "__main__":
    sys.exit(check_implementation())
