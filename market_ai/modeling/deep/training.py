from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from market_ai.data.deep_dataset import DeepDataset
from market_ai.modeling.deep.losses import deep_forecast_loss
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.llm_seq_moe import LLMContextSeqMoE
from market_ai.modeling.deep.oil_context_fusion import OilContextFusion


@dataclass(frozen=True)
class TrainingResult:
    model: torch.nn.Module
    train_loss: float
    validation_loss: float | None
    epochs_ran: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


def _make_model(model_name: str, dataset: DeepDataset, *, hidden_dim: int = 48, dropout: float = 0.1) -> torch.nn.Module:
    tensors = dataset.tensors(dataset.train_indices[:1] if len(dataset.train_indices) else [0])
    kwargs = {
        "price_feature_dim": tensors["x_price"].shape[-1],
        "cross_asset_dim": tensors["x_cross_asset"].shape[-1],
        "event_context_dim": tensors["x_event_context"].shape[-1],
        "static_dim": tensors["x_static"].shape[-1],
        "horizon": tensors["y_vol_scaled_cum_return"].shape[-1],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    if model_name == "oil_context_fusion":
        return OilContextFusion(**{**kwargs, "hidden_dim": max(hidden_dim, 72), "dropout": max(dropout, 0.12)})
    if model_name == "deep_lstm_tcn_fusion":
        return DeepLstmTcnFusion(**kwargs)
    if model_name == "llm_context_seq_moe":
        return LLMContextSeqMoE(**kwargs)
    raise ValueError(f"Unsupported deep model: {model_name}")


def _loader(dataset: DeepDataset, indices: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    tensors = dataset.tensors(indices)
    tensor_dataset = TensorDataset(
        torch.tensor(tensors["x_price"], dtype=torch.float32),
        torch.tensor(tensors["x_cross_asset"], dtype=torch.float32),
        torch.tensor(tensors["x_event_context"], dtype=torch.float32),
        torch.tensor(tensors["x_static"], dtype=torch.float32),
        torch.tensor(tensors["y_vol_scaled_cum_return"], dtype=torch.float32),
        torch.tensor(tensors["y_direction"], dtype=torch.float32),
        torch.tensor(tensors["y_future_volatility"], dtype=torch.float32),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def _eval_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, xc, xe, xs, y, yd, yv in loader:
            xb, xc, xe, xs, y, yd, yv = xb.to(device), xc.to(device), xe.to(device), xs.to(device), y.to(device), yd.to(device), yv.to(device)
            out = model(xb, xc, xe, xs)
            losses.append(float(deep_forecast_loss(out, y_vol_scaled_cum_return=y, y_direction=yd, y_future_volatility=yv).item()))
    return float(np.mean(losses))


def _point_metrics(
    *,
    pred_price: np.ndarray,
    actual_price: np.ndarray,
    true_log_path: np.ndarray,
    pred_log_path: np.ndarray,
) -> dict[str, float]:
    pred_price = np.asarray(pred_price, dtype=np.float64).reshape(-1)
    actual_price = np.asarray(actual_price, dtype=np.float64).reshape(-1)
    true_log_path = np.asarray(true_log_path, dtype=np.float64).reshape(-1)
    pred_log_path = np.asarray(pred_log_path, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred_price) & np.isfinite(actual_price) & (actual_price > 0.0)
    if not np.any(mask):
        return {
            "sse": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
            "smape": float("nan"),
            "directional_accuracy": float("nan"),
        }
    pred = pred_price[mask]
    actual = actual_price[mask]
    errors = pred - actual
    abs_errors = np.abs(errors)
    sse = float(np.sum(errors**2))
    mse = float(np.mean(errors**2))
    actual_mean = float(np.mean(actual))
    denom = float(np.sum((actual - actual_mean) ** 2))
    smape_denom = np.maximum((np.abs(pred) + np.abs(actual)) / 2.0, 1e-8)
    log_mask = np.isfinite(pred_log_path) & np.isfinite(true_log_path)
    if np.any(log_mask):
        direction = float(np.mean(np.sign(pred_log_path[log_mask]) == np.sign(true_log_path[log_mask])))
    else:
        direction = float("nan")
    return {
        "sse": sse,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_errors)),
        "r2": float(1.0 - sse / denom) if denom > 1e-12 else float("nan"),
        "mape": float(np.mean(abs_errors / np.maximum(np.abs(actual), 1e-8)) * 100.0),
        "smape": float(np.mean(abs_errors / smape_denom) * 100.0),
        "directional_accuracy": direction,
    }


def _eval_point_metrics(
    model: torch.nn.Module,
    dataset: DeepDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if len(indices) == 0:
        return {}
    loader = _loader(dataset, indices, batch_size, shuffle=False)
    medians: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, xc, xe, xs, _y, _yd, _yv in loader:
            out = model(
                xb.to(device),
                xc.to(device),
                xe.to(device),
                xs.to(device),
            )
            median = out["quantiles"][..., 3]
            medians.append(median.detach().cpu().numpy())
    pred_scaled = np.concatenate(medians, axis=0)
    selected = [dataset.samples[int(idx)] for idx in indices]
    target_scaled = np.asarray([sample.y_vol_scaled_cum_return for sample in selected], dtype=np.float64)
    recent_vol = np.asarray([sample.recent_realized_volatility for sample in selected], dtype=np.float64)[:, None]
    current_price = np.asarray([sample.current_price for sample in selected], dtype=np.float64)[:, None]
    pred_log_path = np.clip(pred_scaled.astype(np.float64) * recent_vol, -1.5, 1.5)
    true_log_path = np.clip(target_scaled * recent_vol, -1.5, 1.5)
    pred_price = current_price * np.exp(pred_log_path)
    actual_price = current_price * np.exp(true_log_path)
    return _point_metrics(
        pred_price=pred_price,
        actual_price=actual_price,
        true_log_path=true_log_path,
        pred_log_path=pred_log_path,
    )


def train_deep_model(
    model_name: str,
    dataset: DeepDataset,
    *,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    device: str = "cpu",
    seed: int = 42,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_every_batches: int = 10,
) -> TrainingResult:
    if not dataset.samples:
        raise ValueError("Dataset has no samples")
    started = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device)
    model = _make_model(model_name, dataset).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _loader(dataset, dataset.train_indices, batch_size, shuffle=True)
    val_indices = dataset.validation_indices if len(dataset.validation_indices) else dataset.test_indices
    val_loader = _loader(dataset, val_indices, batch_size, shuffle=False)
    total_epochs = max(1, int(epochs))
    total_batches = len(train_loader)
    progress_step = max(1, int(progress_every_batches))
    if progress_callback:
        progress_callback(
            {
                "phase": "train_start",
                "model_name": model_name,
                "device": str(resolved_device),
                "epochs": total_epochs,
                "batch_size": batch_size,
                "train_batches": total_batches,
                "n_train": int(len(dataset.train_indices)),
                "n_val": int(len(val_indices)),
                "elapsed_seconds": 0.0,
            }
        )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    last_train = float("nan")
    stale = 0
    epochs_ran = 0
    for epoch in range(total_epochs):
        epochs_ran = epoch + 1
        model.train()
        batch_losses: list[float] = []
        if progress_callback:
            progress_callback(
                {
                    "phase": "epoch_start",
                    "model_name": model_name,
                    "epoch": epochs_ran,
                    "epochs": total_epochs,
                    "train_batches": total_batches,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        for batch_idx, (xb, xc, xe, xs, y, yd, yv) in enumerate(train_loader, start=1):
            xb, xc, xe, xs, y, yd, yv = xb.to(resolved_device), xc.to(resolved_device), xe.to(resolved_device), xs.to(resolved_device), y.to(resolved_device), yd.to(resolved_device), yv.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, xc, xe, xs)
            loss = deep_forecast_loss(out, y_vol_scaled_cum_return=y, y_direction=yd, y_future_volatility=yv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            if progress_callback and (batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_step == 0):
                progress_callback(
                    {
                        "phase": "batch_done",
                        "model_name": model_name,
                        "epoch": epochs_ran,
                        "epochs": total_epochs,
                        "batch": batch_idx,
                        "train_batches": total_batches,
                        "batch_loss": float(loss.item()),
                        "running_train_loss": float(np.mean(batch_losses)),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
        last_train = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = _eval_loss(model, val_loader, resolved_device)
        comparable = val_loss if np.isfinite(val_loss) else last_train
        improved = comparable < best_val
        if comparable < best_val:
            best_val = comparable
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if progress_callback:
            progress_callback(
                {
                    "phase": "epoch_done",
                    "model_name": model_name,
                    "epoch": epochs_ran,
                    "epochs": total_epochs,
                    "train_loss": last_train,
                    "validation_loss": val_loss,
                    "best_validation_loss": best_val,
                    "improved": improved,
                    "stale_epochs": stale,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        if not improved:
            if stale >= patience:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "early_stop",
                            "model_name": model_name,
                            "epoch": epochs_ran,
                            "patience": patience,
                            "best_validation_loss": best_val,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                break
    model.load_state_dict(best_state)
    validation_metrics = _eval_point_metrics(
        model,
        dataset,
        val_indices,
        batch_size=batch_size,
        device=resolved_device,
    )
    test_metrics = _eval_point_metrics(
        model,
        dataset,
        dataset.test_indices,
        batch_size=batch_size,
        device=resolved_device,
    )
    if progress_callback:
        progress_callback(
            {
                "phase": "train_done",
                "model_name": model_name,
                "epochs_ran": epochs_ran,
                "train_loss": last_train,
                "validation_loss": best_val,
                "validation_rmse": validation_metrics.get("rmse"),
                "validation_mae": validation_metrics.get("mae"),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    return TrainingResult(
        model=model.cpu(),
        train_loss=last_train,
        validation_loss=best_val,
        epochs_ran=epochs_ran,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
