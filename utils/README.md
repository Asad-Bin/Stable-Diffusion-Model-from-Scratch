# utils

Small helpers. Only `gradient_descent.py` is used by the training script.

## `debug.py`

Imports `config.config` (for `device`).

| Symbol | Description |
|---|---|
| `is_local_test = False` | Master switch for memory printing |
| `print_gpu_memory(tag)` | If `is_local_test` is true, prints `torch.cuda.memory_allocated/reserved` (MB) for `device`. Otherwise returns immediately. `main/main.py` calls it in many places. |
| `clear_memory()` | Resets peak stats, `gc.collect()`, `cuda.empty_cache()`, `cuda.ipc_collect()`. Not called anywhere. |

Set `is_local_test = True` to see memory numbers (CUDA required).

## `gradient_descent.py`

`plot_and_log_grad_norms(run_id, weight_metrics, bias_metrics, client)`: for each metric name it fetches the history with `MlflowClient.get_metric_history`, plots the metrics on one figure (legend labels with the `grad_norm_weight_` or `grad_norm_bias_` prefix removed), and logs the figure to the active run with `mlflow.log_image`:

- `gradient_norms_-_weights_(first_conv_layers).png`
- `gradient_norms_-_biases_(first_conv_layers).png`

In `main/main.py` (`log_grad_plots`, called once per epoch) the lists are built from metric names starting with `grad_norm_weight_` / `grad_norm_bias_`. The metrics actually logged are `grad_norm_<parameter name>` (e.g. `grad_norm_enc1.conv.conv1.weight`) for parameters whose names contain both `weight` and `conv1`, and those never start with `grad_norm_weight_`. In practice both lists are empty, so both logged plots have no curves. Bias gradients are never logged.

## `epoch_progress.py`

`print_epoch_progress(current_epoch, total_epochs, bar_length=30)`: writes `\rEpoch n/N [====   ]` to stdout. Not used by the current scripts (the call in `vae_train/train.py` is commented out).
