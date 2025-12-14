# main.py
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from traffic_grid_dataset import TrafficGridDataset, SplitConfig
from mamba_model import MambaIncidentModel


# ======================================================
# Utils
# ======================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# Data
# ======================================================
def build_loaders(args, device):
    split_cfg = SplitConfig(train=0.70, val=0.15, test=0.15)

    def make(split, shuffle):
        ds = TrafficGridDataset(
            csv_path=args.csv,
            split=split,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
            split_cfg=split_cfg,
            scale=True,
            time_encoding=args.time_encoding,
            target="any_incident",   # 🔒 single target
            interpolate_weather=args.interpolate_weather,
            add_holiday=args.add_holiday,
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )
        return ds, dl

    ds_train, dl_train = make("train", True)
    ds_val, dl_val = make("val", False)
    ds_test, dl_test = make("test", False)

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test


# ======================================================
# Main
# ======================================================
def main():
    ap = argparse.ArgumentParser("Traffic Mamba Training (Single Target)")

    # data
    ap.add_argument("--csv", type=str, default="data/traffic_data.csv")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--pred_horizon", type=int, default=0)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=256)

    # features
    ap.add_argument("--time_encoding", type=str, default="cyclic",
                    choices=["none", "raw", "cyclic"])
    ap.add_argument("--add_holiday", action="store_true", default=True)
    ap.add_argument("--interpolate_weather", action="store_true", default=True)

    # system
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Args:", vars(args))

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test = build_loaders(args, device)

    print("\nDataset sizes")
    print("train:", len(ds_train))
    print("val  :", len(ds_val))
    print("test :", len(ds_test))

    # --------------------------------------------------
    # Infer input dimension
    # --------------------------------------------------
    batch = next(iter(dl_train))
    if len(batch) == 2:
        x, y = batch
        in_dim = x.shape[-1]
    else:
        x, t, y = batch
        in_dim = x.shape[-1] + t.shape[-1]

    print("Input dim:", in_dim)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = MambaIncidentModel(
        input_dim=in_dim,
        d_model=128,
        n_layers=4,
    ).to(device)

    # --------------------------------------------------
    # Loss (imbalanced binary classification)
    # --------------------------------------------------
    y_train = np.asarray(ds_train.y).astype(np.float32)
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos

    # 🔒 smoothed pos_weight (stability)
    pos_weight = torch.tensor(
        [(num_neg / (num_pos + 1e-6)) ** 0.5],
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --------------------------------------------------
    # Optimizer + Scheduler
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=0.3,
    )

    scaler = torch.cuda.amp.GradScaler()

    # --------------------------------------------------
    # Checkpointing
    # --------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_auc = -1.0
    ckpt_path = "checkpoints/best_model.pt"

    # ==================================================
    # Training Loop
    # ==================================================
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        # ---------------- Train ----------------
        model.train()
        train_losses = []

        for batch in dl_train:
            if len(batch) == 2:
                x, y = batch
                x = x.to(device)
            else:
                x, t, y = batch
                x = torch.cat([x, t], dim=-1).to(device)

            y = y.float().to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = model(x)

                # 🔒 critical numerical guard
                logits = torch.clamp(logits, -20, 20)

                loss = criterion(logits, y)

            # skip bad batches
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # 🔒 Mamba needs strong clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())

        print(f"Train loss: {np.mean(train_losses):.4f}")

        # ---------------- Validation ----------------
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in dl_val:
                if len(batch) == 2:
                    x, y = batch
                    x = x.to(device)
                else:
                    x, t, y = batch
                    x = torch.cat([x, t], dim=-1).to(device)

                y = y.float().to(device)

                logits = model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        if np.isnan(all_probs).any():
            print("⚠️ NaNs in validation — skipping epoch")
            continue

        val_auc = roc_auc_score(all_labels, all_probs)
        # ---- threshold sweep (CRITICAL) ----
        for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = (all_probs >= th).astype(int)
            recall = recall_score(all_labels, preds, zero_division=0)
            precision = precision_score(all_labels, preds, zero_division=0)
            print(f"th={th:.1f} | recall={recall:.4f} | precision={precision:.4f}")

        # ---------------- Save best ----------------
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_auc,
                },
                ckpt_path,
            )
            print(f"✓ Saved new best model (AUC={val_auc:.4f})")

    print("\nTraining finished")
    print("Best Val AUC:", best_auc)
    print("Checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
