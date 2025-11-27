# Setup and Testing Guide: Generator Parameter

## Prerequisites

- Python 3.8+
- CMake 3.18+
- C++ compiler (GCC 7+ or Clang 8+)
- Git
- (Optional) CUDA toolkit for GPU support

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
```

### 2. Install Dependencies

```bash
# Install PyTorch dependencies
pip install -r requirements.txt

# Install build dependencies
pip install -r requirements-build.txt
```

### 3. Build PyTorch

```bash
# Development build (recommended for testing)
python setup.py develop

# Or regular build
python setup.py build
```

**Note**: This will:
- Parse `native_functions.yaml`
- Generate Python bindings
- Compile C++ code
- Create the `torch` module

Build time: ~30-60 minutes depending on hardware.

---

## Testing Generator Parameter

### Quick Test (No Build Required)

The YAML syntax can be validated:

```bash
cd pytorch-repo
python -c "import yaml; yaml.safe_load(open('aten/src/ATen/native/native_functions.yaml'))"
```

### Full Test (Requires Build)

```bash
cd pytorch-repo

# Run generator parameter tests
python -m pytest test/test_tensor_creation_ops.py::test_rand_like -v
python -m pytest test/test_tensor_creation_ops.py::test_randn_like -v
python -m pytest test/test_tensor_creation_ops.py::test_randint_like -v

# Or run custom test suite
cd /Users/alexho/Million-line\ Codebase/pytorch
python test_generator_parameter.py
```

### Manual Testing

```python
import torch

# Create a generator
gen = torch.Generator()
gen.manual_seed(42)

# Test rand_like with generator
x = torch.randn(5, 5)
y1 = torch.rand_like(x, generator=gen)
gen.manual_seed(42)  # Reset
y2 = torch.rand_like(x, generator=gen)

# Should be identical (reproducible)
assert torch.equal(y1, y2), "Generator not working!"

# Test backward compatibility (no generator)
z = torch.rand_like(x)  # Should work without generator

print("✅ All tests passed!")
```

### Test Reproducibility

```python
import torch

# Test that generator produces reproducible results
gen1 = torch.Generator()
gen1.manual_seed(123)

gen2 = torch.Generator()
gen2.manual_seed(123)

x = torch.randn(3, 3)
y1 = torch.randn_like(x, generator=gen1)
y2 = torch.randn_like(x, generator=gen2)

assert torch.equal(y1, y2), "Results should be identical with same seed"
print("✅ Reproducibility test passed!")
```

---

## Troubleshooting

### Build Issues

**Problem**: CMake not found  
**Solution**: Install CMake: `brew install cmake` (macOS) or `apt-get install cmake` (Linux)

**Problem**: C++ compiler not found  
**Solution**: Install build tools: `xcode-select --install` (macOS) or `apt-get install build-essential` (Linux)

**Problem**: Python bindings not generated  
**Solution**: Ensure you ran `python setup.py develop` after modifying `native_functions.yaml`

### Test Issues

**Problem**: Tests fail with "Module not found"  
**Solution**: Ensure PyTorch is built and installed: `python setup.py develop`

**Problem**: Generator tests fail  
**Solution**: Ensure you're using a built version of PyTorch, not pip-installed version

---

## Performance Testing

### Generator Performance

```python
import torch
import time

def test_generator_performance():
    gen = torch.Generator()
    x = torch.randn(1000, 1000)
    
    # Without generator
    start = time.time()
    for _ in range(100):
        _ = torch.rand_like(x)
    time_without = time.time() - start
    
    # With generator
    start = time.time()
    for _ in range(100):
        _ = torch.rand_like(x, generator=gen)
    time_with = time.time() - start
    
    overhead = ((time_with - time_without) / time_without) * 100
    print(f"Generator overhead: {overhead:.2f}%")
    assert overhead < 1, "Generator overhead should be < 1%"

test_generator_performance()
```

---

## Next Steps

1. ✅ Build PyTorch from source
2. ✅ Run unit tests
3. ✅ Run integration tests
4. ✅ Generate test coverage reports
5. ✅ Verify feature works end-to-end
6. ✅ Create pull request

