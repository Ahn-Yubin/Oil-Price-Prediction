from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from market_ai.data.deep_dataset import DeepDataset
from market_ai.modeling.deep.losses import deep_forecast_loss
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.llm_seq_moe import LLMContextSeqMoE


@dataclass(frozen=True)
class TrainingResult:
    model: torch.nn.Module
    train_loss: float
    validation_loss: float | None
    epochs_ran: int


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
) -> TrainingResult:
    if not dataset.samples:
        raise ValueError("Dataset has no samples")
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device)
    model = _make_model(model_name, dataset).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _loader(dataset, dataset.train_indices, batch_size, shuffle=True)
    val_indices = dataset.validation_indices if len(dataset.validation_indices) else dataset.test_indices
    val_loader = _loader(dataset, val_indices, batch_size, shuffle=False)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    last_train = float("nan")
    stale = 0
    epochs_ran = 0
    for epoch in range(max(1, int(epochs))):
        epochs_ran = epoch + 1
        model.train()
        batch_losses: list[float] = []
        for xb, xc, xe, xs, y, yd, yv in train_loader:
            xb, xc, xe, xs, y, yd, yv = xb.to(resolved_device), xc.to(resolved_device), xe.to(resolved_device), xs.to(resolved_device), y.to(resolved_device), yd.to(resolved_device), yv.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, xc, xe, xs)
            loss = deep_forecast_loss(out, y_vol_scaled_cum_return=y, y_direction=yd, y_future_volatility=yv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        last_train = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = _eval_loss(model, val_loader, resolved_device)
        comparable = val_loss if np.isfinite(val_loss) else last_train
        if comparable < best_val:
            best_val = comparable
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(model=model.cpu(), train_loss=last_train, validation_loss=best_val, epochs_ran=epochs_ran)
