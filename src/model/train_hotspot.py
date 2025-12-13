from __future__ import annotations

import json
from pathlib import Path


def _weighted_mse(pred, target, *, pos_weight: float):
    import torch

    w = torch.where(target > 0.0, torch.tensor(pos_weight, device=target.device), torch.tensor(1.0, device=target.device))
    return ((pred - target) ** 2 * w).mean()


def train_hotspot_model(
    *,
    data_dir: Path,
    out_path: Path,
    context_steps: int,
    arch: str = "mamba",
    d_model: int = 128,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    pos_weight_coll: float = 50.0,
    pos_weight_inc: float = 10.0,
    train_frac: float = 0.8,
    device: str | None = None,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
) -> None:
    import numpy as np
    import torch

    ds = np.load(data_dir / "dataset.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    X = ds["X"].astype("float32")
    X_mean = ds["X_mean"].astype("float32")
    X_std = ds["X_std"].astype("float32")
    Xn = (X - X_mean) / X_std

    T, D = Xn.shape
    n_cells = int(meta["n_cells"])
    if T <= context_steps:
        raise ValueError(f"Need T > context_steps (T={T}, context_steps={context_steps})")

    coll_ptr = ds["coll_ptr"].astype("int64")
    coll_idx = ds["coll_idx"].astype("int64")
    inc_ptr = ds["inc_ptr"].astype("int64")
    inc_idx = ds["inc_idx"].astype("int64")

    from .model_hotspot import build_model

    device_t = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_features=D, n_cells=n_cells, d_model=d_model, arch=arch).to(device_t)
    resolved_arch = getattr(model, "arch", arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def dense_labels(ptr: np.ndarray, idx: np.ndarray, t: int) -> torch.Tensor:
        y = torch.zeros(n_cells, device=device_t)
        start = int(ptr[t])
        end = int(ptr[t + 1])
        if end > start:
            y[torch.from_numpy(idx[start:end]).to(device_t)] = 1.0
        return y

    indices = list(range(context_steps, T))
    split = int(len(indices) * float(train_frac))
    split = min(max(split, 0), len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else []

    Xn_t = torch.from_numpy(Xn).to(device_t)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for bi in range(0, len(train_idx), batch_size):
            if max_train_batches > 0 and n_batches >= max_train_batches:
                break
            batch_ts = train_idx[bi : bi + batch_size]
            xb = torch.stack([Xn_t[t - context_steps : t, :] for t in batch_ts], dim=0)
            y_coll = torch.stack([dense_labels(coll_ptr, coll_idx, t) for t in batch_ts], dim=0)
            y_inc = torch.stack([dense_labels(inc_ptr, inc_idx, t) for t in batch_ts], dim=0)

            opt.zero_grad(set_to_none=True)
            logits_coll, logits_inc = model(xb)
            pred_coll = torch.sigmoid(logits_coll)
            pred_inc = torch.sigmoid(logits_inc)
            loss = _weighted_mse(pred_coll, y_coll, pos_weight=pos_weight_coll) + _weighted_mse(
                pred_inc, y_inc, pos_weight=pos_weight_inc
            )
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = total / max(n_batches, 1)

        model.eval()
        v_total = 0.0
        v_batches = 0
        with torch.no_grad():
            for bi in range(0, len(val_idx), batch_size):
                if max_val_batches > 0 and v_batches >= max_val_batches:
                    break
                batch_ts = val_idx[bi : bi + batch_size]
                xb = torch.stack([Xn_t[t - context_steps : t, :] for t in batch_ts], dim=0)
                y_coll = torch.stack([dense_labels(coll_ptr, coll_idx, t) for t in batch_ts], dim=0)
                y_inc = torch.stack([dense_labels(inc_ptr, inc_idx, t) for t in batch_ts], dim=0)

                logits_coll, logits_inc = model(xb)
                pred_coll = torch.sigmoid(logits_coll)
                pred_inc = torch.sigmoid(logits_inc)
                loss = _weighted_mse(pred_coll, y_coll, pos_weight=pos_weight_coll) + _weighted_mse(
                    pred_inc, y_inc, pos_weight=pos_weight_inc
                )
                v_total += float(loss.detach().cpu().item())
                v_batches += 1

        val_loss = v_total / max(v_batches, 1) if val_idx else float("nan")
        print(f"[train] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": {
            "arch": resolved_arch,
            "d_model": d_model,
            "n_features": D,
            "n_cells": n_cells,
            "context_steps": context_steps,
        },
        "tokenizer_meta": meta,
        "x_mean": X_mean,
        "x_std": X_std,
        "data_dir": str(data_dir),
    }
    torch.save(ckpt, out_path)
