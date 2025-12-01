#!/usr/bin/env python3
"""
Test script for generator parameter in rand*_like functions.

This script tests the implementation of generator parameter support
for randn_like, rand_like, and randint_like functions.

Run this after building PyTorch with the updated native_functions.yaml
"""

import torch
import sys

def test_randn_like_with_generator():
    """Test randn_like with generator parameter"""
    print("Testing randn_like with generator...")
    
    # Test 1: Basic functionality with generator
    gen = torch.Generator()
    gen.manual_seed(42)
    x = torch.randn(3, 4)
    y1 = torch.randn_like(x, generator=gen)
    
    gen.manual_seed(42)
    y2 = torch.randn_like(x, generator=gen)
    
    assert torch.allclose(y1, y2), "randn_like should be reproducible with same generator seed"
    assert y1.shape == x.shape, "randn_like should preserve input shape"
    print("  ✓ Basic functionality works")
    
    # Test 2: Backward compatibility (no generator)
    z = torch.randn_like(x)
    assert z.shape == x.shape, "randn_like should work without generator"
    print("  ✓ Backward compatibility works")
    
    # Test 3: Different generators produce different results
    gen1 = torch.Generator().manual_seed(1)
    gen2 = torch.Generator().manual_seed(2)
    y1 = torch.randn_like(x, generator=gen1)
    y2 = torch.randn_like(x, generator=gen2)
    assert not torch.allclose(y1, y2), "Different generators should produce different results"
    print("  ✓ Different generators work correctly")
    
    print("  ✅ randn_like tests passed\n")
    return True

def test_rand_like_with_generator():
    """Test rand_like with generator parameter"""
    print("Testing rand_like with generator...")
    
    # Test 1: Basic functionality with generator
    gen = torch.Generator()
    gen.manual_seed(42)
    x = torch.randn(3, 4)
    y1 = torch.rand_like(x, generator=gen)
    
    gen.manual_seed(42)
    y2 = torch.rand_like(x, generator=gen)
    
    assert torch.allclose(y1, y2), "rand_like should be reproducible with same generator seed"
    assert y1.shape == x.shape, "rand_like should preserve input shape"
    assert (y1 >= 0).all() and (y1 < 1).all(), "rand_like should produce values in [0, 1)"
    print("  ✓ Basic functionality works")
    
    # Test 2: Backward compatibility (no generator)
    z = torch.rand_like(x)
    assert z.shape == x.shape, "rand_like should work without generator"
    print("  ✓ Backward compatibility works")
    
    print("  ✅ rand_like tests passed\n")
    return True

def test_randint_like_with_generator():
    """Test randint_like with generator parameter"""
    print("Testing randint_like with generator...")
    
    # Test 1: Basic functionality with generator (high only)
    gen = torch.Generator()
    gen.manual_seed(42)
    x = torch.randn(3, 4)
    y1 = torch.randint_like(x, 10, generator=gen)
    
    gen.manual_seed(42)
    y2 = torch.randint_like(x, 10, generator=gen)
    
    assert torch.equal(y1, y2), "randint_like should be reproducible with same generator seed"
    assert y1.shape == x.shape, "randint_like should preserve input shape"
    assert (y1 >= 0).all() and (y1 < 10).all(), "randint_like should produce values in [0, 10)"
    print("  ✓ Basic functionality (high only) works")
    
    # Test 2: With low and high
    gen.manual_seed(42)
    y1 = torch.randint_like(x, 5, 15, generator=gen)
    gen.manual_seed(42)
    y2 = torch.randint_like(x, 5, 15, generator=gen)
    
    assert torch.equal(y1, y2), "randint_like should be reproducible with same generator seed"
    assert (y1 >= 5).all() and (y1 < 15).all(), "randint_like should produce values in [5, 15)"
    print("  ✓ Basic functionality (low and high) works")
    
    # Test 3: Backward compatibility (no generator)
    z = torch.randint_like(x, 10)
    assert z.shape == x.shape, "randint_like should work without generator"
    print("  ✓ Backward compatibility works")
    
    print("  ✅ randint_like tests passed\n")
    return True

def test_device_matching():
    """Test that generator device must match tensor device"""
    print("Testing device matching...")
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping device mismatch test")
        return True
    
    try:
        gen = torch.Generator(device='cpu')
        x = torch.randn(3, 4, device='cuda')
        y = torch.randn_like(x, generator=gen)
        print("  ⚠ Device mismatch should have raised an error")
        return False
    except RuntimeError as e:
        if 'device' in str(e).lower():
            print("  ✓ Device mismatch correctly raises error")
            return True
        else:
            print(f"  ✗ Unexpected error: {e}")
            return False

def test_reproducibility():
    """Test that same generator seed produces same results"""
    print("Testing reproducibility...")
    
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    x = torch.randn(3, 4)
    
    y1 = torch.randn_like(x, generator=gen1)
    y2 = torch.randn_like(x, generator=gen2)
    
    assert torch.allclose(y1, y2), "Same seed should produce same results"
    print("  ✓ Reproducibility works correctly")
    
    print("  ✅ Reproducibility tests passed\n")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Generator Parameter for rand*_like Functions")
    print("=" * 60)
    print()
    
    tests = [
        ("randn_like", test_randn_like_with_generator),
        ("rand_like", test_rand_like_with_generator),
        ("randint_like", test_randint_like_with_generator),
        ("device_matching", test_device_matching),
        ("reproducibility", test_reproducibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_name} failed\n")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} failed with error: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

