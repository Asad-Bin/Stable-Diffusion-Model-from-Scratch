# Utils Module

Utility functions for debugging, training progress, and gradient analysis.

## Files

| File | Description |
|------|-------------|
| `debug.py` | Debugging utilities and tensor inspection |
| `epoch_progress.py` | Training progress tracking and display |
| `gradient_descent.py` | Gradient analysis and monitoring tools |

## Debug Utilities

### `debug.py`

Tools for inspecting tensors and debugging training issues:

```python
from utils.debug import *

# Inspect tensor statistics
print_tensor_stats(tensor)
# Output: shape, dtype, min, max, mean, std, nan count
```

## Progress Tracking

### `epoch_progress.py`

Utilities for displaying and tracking training progress:

```python
from utils.epoch_progress import EpochProgress

progress = EpochProgress(total_epochs=3000)
progress.update(current_epoch=100, loss=0.045)
```

## Gradient Analysis

### `gradient_descent.py`

Tools for monitoring gradient flow and detecting training issues:

```python
from utils.gradient_descent import (
    log_gradient_norms,
    check_gradient_health
)

# Log gradient norms for each layer
log_gradient_norms(model)

# Check for vanishing/exploding gradients
health_report = check_gradient_health(model)
```

## Common Debugging Patterns

### Tensor Inspection

```python
def debug_tensor(name, tensor):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
```

### Gradient Monitoring

```python
def log_gradient_norms(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
```

## Usage in Training

```python
from utils.debug import *
from utils.gradient_descent import log_gradient_norms

# In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        loss.backward()
        
        # Log gradients periodically
        if epoch % 100 == 0:
            grad_norm = log_gradient_norms(model)
            print(f"Epoch {epoch}: Gradient norm = {grad_norm:.4f}")
        
        optimizer.step()
```
