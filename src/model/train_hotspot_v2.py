from __future__ import annotations

import json
from pathlib import Path


def train_hotspot_v2_model(
    *,
    data_dir: Path,
    out_path: Path,
    hidden_dim: int = 128,
    dropout: float = 0.0,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 3e-4,
    train_frac: float = 0.8,
    device: str | None = None,
    pos_weight_coll: float = 1.0,
    pos_weight_inc: float = 1.0,
    use_amp: bool = True,
    grad_clip_norm: float = 0.5,
    logit_clamp: float = 20.0,
    log_every_steps: int = 50,
) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if int(epochs) <= 0:
        raise ValueError(f"epochs must be > 0 (got {epochs})")

    ds = np.load(data_dir / "dataset.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    X = ds["X"].astype("float32")
    y_coll = ds["y_coll"].astype("float32", copy=False)
    y_inc = ds["y_inc"].astype("float32", copy=False)
    bucket_idx = ds["bucket_idx"].astype("int32")

    n_features = int(X.shape[1])
    n_buckets = int(meta.get("n_buckets") or (int(bucket_idx.max()) + 1 if bucket_idx.size else 0))
    if n_buckets <= 0:
        raise ValueError("invalid dataset: n_buckets")

    n_cells = int(meta.get("n_cells") or 0)
    if n_cells <= 0:
        raise ValueError("invalid dataset: n_cells")

    n_samples = int(y_coll.shape[0])
    pos_cell_hours_coll = int(float(y_coll.sum()))
    pos_cell_hours_inc = int(float(y_inc.sum()))
    sample_rate_coll = float(pos_cell_hours_coll) / float(max(n_samples, 1))
    sample_rate_inc = float(pos_cell_hours_inc) / float(max(n_samples, 1))
    true_rate_coll = float(pos_cell_hours_coll) / float(n_cells * n_buckets)
    true_rate_inc = float(pos_cell_hours_inc) / float(n_cells * n_buckets)

    split_t = int(float(train_frac) * n_buckets)
    split_t = min(max(split_t, 0), n_buckets)
    train_mask = bucket_idx < split_t
    val_mask = ~train_mask

    device_t = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .model_hotspot_v2 import build_hotspot_v2_model

    model = build_hotspot_v2_model(n_features=n_features, hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    # Train-only normalization (TrafficMamba-style hygiene).
    if not bool(train_mask.any()):
        raise ValueError("empty train split (train_frac too small?)")
    X_train_np = X[train_mask]
    x_mean = X_train_np.mean(axis=0).astype(np.float32)
    x_std = X_train_np.std(axis=0).astype(np.float32)
    x_std = np.where(x_std == 0.0, 1.0, x_std).astype(np.float32)

    Xn = (X - x_mean) / x_std

    x_train = torch.from_numpy(Xn[train_mask]).float()
    y_coll_train = torch.from_numpy(y_coll[train_mask]).float()
    y_inc_train = torch.from_numpy(y_inc[train_mask]).float()

    x_val = torch.from_numpy(Xn[val_mask]).float() if bool(val_mask.any()) else None
    y_coll_val = torch.from_numpy(y_coll[val_mask]).float() if bool(val_mask.any()) else None
    y_inc_val = torch.from_numpy(y_inc[val_mask]).float() if bool(val_mask.any()) else None

    dl_train = DataLoader(
        TensorDataset(x_train, y_coll_train, y_inc_train),
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        drop_last=False,
    )
    dl_val = (
        DataLoader(
            TensorDataset(x_val, y_coll_val, y_inc_val),  # type: ignore[arg-type]
            batch_size=max(int(batch_size), 1),
            shuffle=False,
            drop_last=False,
        )
        if x_val is not None and y_coll_val is not None and y_inc_val is not None and int(x_val.shape[0]) > 0
        else None
    )

    use_cuda_amp = bool(use_amp) and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_cuda_amp)

    pos_w_coll = torch.tensor(float(pos_weight_coll), device=device_t)
    pos_w_inc = torch.tensor(float(pos_weight_inc), device=device_t)

    def bce_loss(logits: torch.Tensor, target: torch.Tensor, *, pos_weight: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def save_ckpt(
        *,
        path: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_auc_coll: float,
        val_auc_inc: float,
        val_ap_coll: float,
        val_ap_inc: float,
        best_score: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "model_config": {
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "n_features": int(n_features),
            },
            "dataset_meta": meta,
            "feature_names": list(meta.get("feature_names") or []),
            "x_mean": x_mean,
            "x_std": x_std,
            "data_dir": str(data_dir),
            "calibration": {
                "n_cells": int(n_cells),
                "n_buckets": int(n_buckets),
                "n_samples": int(n_samples),
                "pos_cell_hours_coll": int(pos_cell_hours_coll),
                "pos_cell_hours_inc": int(pos_cell_hours_inc),
                "sample_rate_coll": float(sample_rate_coll),
                "sample_rate_inc": float(sample_rate_inc),
                "true_rate_coll": float(true_rate_coll),
                "true_rate_inc": float(true_rate_inc),
            },
            "train": {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "split_t": int(split_t),
                "n_buckets": int(n_buckets),
                "pos_weight_coll": float(pos_weight_coll),
                "pos_weight_inc": float(pos_weight_inc),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "use_amp": bool(use_amp),
                "grad_clip_norm": float(grad_clip_norm),
                "logit_clamp": float(logit_clamp),
                "best_score": float(best_score),
                "val_auc_coll": float(val_auc_coll),
                "val_auc_inc": float(val_auc_inc),
                "val_ap_coll": float(val_ap_coll),
                "val_ap_inc": float(val_ap_inc),
            },
        }
        torch.save(ckpt, path)
        # Keep checkpoints loadable on CPU even if trained on CUDA.

    best_score = float("-inf")
    best_epoch = 0
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    def score_from_aucs(a_coll: float, a_inc: float) -> float:
        vals = [v for v in [a_coll, a_inc] if v == v]
        if not vals:
            return float("-inf")
        return float(sum(vals) / len(vals))

    def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
        except Exception:
            return float("nan"), float("nan")
        if y_true.min() == y_true.max():
            return float("nan"), float("nan")
        return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))

    def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> list[tuple[float, float, float]]:
        try:
            from sklearn.metrics import precision_score, recall_score  # type: ignore
        except Exception:
            return []
        out: list[tuple[float, float, float]] = []
        for th in thresholds:
            pred = (y_prob >= th).astype(np.int32)
            out.append(
                (
                    float(th),
                    float(recall_score(y_true, pred, zero_division=0)),
                    float(precision_score(y_true, pred, zero_division=0)),
                )
            )
        return out

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        n_steps = 0
        for xb, yb_coll, yb_inc in dl_train:
            xb = xb.to(device_t, non_blocking=True)
            yb_coll = yb_coll.to(device_t, non_blocking=True)
            yb_inc = yb_inc.to(device_t, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=str(device_t.type),
                dtype=(torch.float16 if device_t.type == "cuda" else None),
                enabled=scaler.is_enabled(),
            ):
                logits_coll, logits_inc = model(xb)
                if float(logit_clamp) > 0:
                    c = float(logit_clamp)
                    logits_coll = torch.clamp(logits_coll, -c, c)
                    logits_inc = torch.clamp(logits_inc, -c, c)
                loss = bce_loss(logits_coll, yb_coll, pos_weight=pos_w_coll) + bce_loss(
                    logits_inc, yb_inc, pos_weight=pos_w_inc
                )

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if float(grad_clip_norm) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()

            loss_f = float(loss.detach().cpu().item())
            total += loss_f
            n_steps += 1
            if int(log_every_steps) > 0 and (n_steps % int(log_every_steps) == 0):
                avg = total / max(n_steps, 1)
                print(f"[train_v2] epoch={epoch} step={n_steps} loss={loss_f:.6f} avg={avg:.6f}", flush=True)

        train_loss = total / max(n_steps, 1)

        model.eval()
        val_loss = float("nan")
        val_auc_coll = float("nan")
        val_auc_inc = float("nan")
        val_ap_coll = float("nan")
        val_ap_inc = float("nan")

        if dl_val is not None:
            v_total = 0.0
            v_steps = 0
            probs_coll: list[np.ndarray] = []
            probs_inc: list[np.ndarray] = []
            labels_coll: list[np.ndarray] = []
            labels_inc: list[np.ndarray] = []

            with torch.no_grad():
                for xb, yb_coll, yb_inc in dl_val:
                    xb = xb.to(device_t, non_blocking=True)
                    yb_coll = yb_coll.to(device_t, non_blocking=True)
                    yb_inc = yb_inc.to(device_t, non_blocking=True)

                    logits_coll, logits_inc = model(xb)
                    if float(logit_clamp) > 0:
                        c = float(logit_clamp)
                        logits_coll = torch.clamp(logits_coll, -c, c)
                        logits_inc = torch.clamp(logits_inc, -c, c)
                    loss = bce_loss(logits_coll, yb_coll, pos_weight=pos_w_coll) + bce_loss(
                        logits_inc, yb_inc, pos_weight=pos_w_inc
                    )
                    v_total += float(loss.detach().cpu().item())
                    v_steps += 1

                    probs_coll.append(torch.sigmoid(logits_coll).detach().cpu().numpy())
                    probs_inc.append(torch.sigmoid(logits_inc).detach().cpu().numpy())
                    labels_coll.append(yb_coll.detach().cpu().numpy())
                    labels_inc.append(yb_inc.detach().cpu().numpy())

            val_loss = v_total / max(v_steps, 1)

            y_coll_true = np.concatenate(labels_coll, axis=0).astype(np.float32)
            y_inc_true = np.concatenate(labels_inc, axis=0).astype(np.float32)
            p_coll = np.concatenate(probs_coll, axis=0).astype(np.float32)
            p_inc = np.concatenate(probs_inc, axis=0).astype(np.float32)

            val_auc_coll, val_ap_coll = eval_metrics(y_coll_true, p_coll)
            val_auc_inc, val_ap_inc = eval_metrics(y_inc_true, p_inc)

            sweeps_coll = sweep_thresholds(y_coll_true, p_coll)
            sweeps_inc = sweep_thresholds(y_inc_true, p_inc)
            if sweeps_coll:
                print("[val_v2] collisions threshold sweep:", flush=True)
                for th, rec, prec in sweeps_coll:
                    print(f"[val_v2]   th={th:.1f} recall={rec:.4f} precision={prec:.4f}", flush=True)
            if sweeps_inc:
                print("[val_v2] incidents threshold sweep:", flush=True)
                for th, rec, prec in sweeps_inc:
                    print(f"[val_v2]   th={th:.1f} recall={rec:.4f} precision={prec:.4f}", flush=True)

        score = score_from_aucs(val_auc_coll, val_auc_inc) if dl_val is not None else -train_loss
        improved = (epoch == 1) or (score > best_score)
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            save_ckpt(
                path=out_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auc_coll=val_auc_coll,
                val_auc_inc=val_auc_inc,
                val_ap_coll=val_ap_coll,
                val_ap_inc=val_ap_inc,
                best_score=best_score,
            )

        last_path = out_path.with_name(f"{out_path.stem}_last{out_path.suffix}")
        save_ckpt(
            path=last_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auc_coll=val_auc_coll,
            val_auc_inc=val_auc_inc,
            val_ap_coll=val_ap_coll,
            val_ap_inc=val_ap_inc,
            best_score=best_score,
        )

        print(
            f"[train_v2] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"auc_coll={val_auc_coll:.4f} auc_inc={val_auc_inc:.4f} "
            f"ap_coll={val_ap_coll:.4f} ap_inc={val_ap_inc:.4f} "
            f"{'(best)' if improved else ''}",
            flush=True,
        )

    print(f"[train_v2] done best_epoch={best_epoch} best_score={best_score:.6f} -> {out_path}", flush=True)
