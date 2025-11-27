# Setup and Testing Guide: Compile Diagnostics

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

## Testing Compile Diagnostics

### Quick Test (No Build Required)

```bash
cd pytorch-repo
python3 test_diagnostics_unit.py
python3 test_diagnostics_extended.py
```

These tests work without building PyTorch by mocking imports.

### Full Integration Test (Requires Build)

```bash
cd pytorch-repo

# Run diagnostic tests
python -m pytest test/dynamo/test_diagnostics.py -v

# Run with coverage
python -m pytest test/dynamo/test_diagnostics.py --cov=torch._dynamo.diagnostics --cov-report=html
```

### Manual Testing

```python
import torch
from torch._dynamo.diagnostics import CompileDiagnostics

def my_model(x):
    print("This will cause a graph break")
    if x.sum() > 0:
        return x * 2
    return x

# Test diagnostics
with CompileDiagnostics(function_name="my_model") as diag:
    compiled = torch.compile(my_model)
    result = compiled(torch.randn(10))

# Get report
report = diag.get_report()
print(f"Graph breaks: {report.graph_break_count}")

# Generate HTML dashboard
report.to_html("diagnostics.html")
```

### Verify HTML Dashboard

```bash
# Open the generated HTML file
open diagnostics.html  # macOS
# or
xdg-open diagnostics.html  # Linux
# or just double-click the file
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

**Problem**: Diagnostics not capturing breaks  
**Solution**: Ensure hooks are properly integrated in `output_graph.py` and `convert_frame.py`

---

## Performance Testing

### Diagnostics Overhead

```python
import torch
import time
from torch._dynamo.diagnostics import CompileDiagnostics

def test_overhead():
    def model(x):
        return x * 2
    
    # Without diagnostics
    start = time.time()
    for _ in range(100):
        compiled = torch.compile(model)
        _ = compiled(torch.randn(10))
    time_without = time.time() - start
    
    # With diagnostics
    start = time.time()
    for _ in range(100):
        with CompileDiagnostics():
            compiled = torch.compile(model)
            _ = compiled(torch.randn(10))
    time_with = time.time() - start
    
    overhead = ((time_with - time_without) / time_without) * 100
    print(f"Overhead: {overhead:.2f}%")
    assert overhead < 5, "Overhead should be < 5%"

test_overhead()
```

---

## Next Steps

1. ✅ Build PyTorch from source
2. ✅ Run unit tests
3. ✅ Run integration tests
4. ✅ Generate test coverage reports
5. ✅ Verify feature works end-to-end
6. ✅ Create pull request

