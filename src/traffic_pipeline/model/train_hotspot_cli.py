from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from traffic_pipeline.config import load_config
from traffic_pipeline.model.train_hotspot import train_hotspot_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train hotspot predictor (Mamba/GRU) on tokenized H3 dataset")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--data-dir", default=None, help="Tokenizer output dir (defaults to config tokenizer.output_dir)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--arch", default="mamba", help="mamba or gru (mamba will fall back to gru if unavailable)")
    ap.add_argument("--pos-weight-coll", type=float, default=25.0)
    ap.add_argument("--pos-weight-inc", type=float, default=10.0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--device", default=None, help="cpu|cuda (auto if omitted)")
    ap.add_argument("--max-train-batches", type=int, default=0, help="0=run all")
    ap.add_argument("--max-val-batches", type=int, default=0, help="0=run all")
    ap.add_argument("--out", default="artifacts/h3_hotspot_model.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else cfg.tokenizer.output_dir
    ds_path = data_dir / "dataset.npz"
    meta_path = data_dir / "meta.json"
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    context_steps = int(cfg.train.context_steps)

    out_path = Path(args.out)
    train_hotspot_model(
        data_dir=data_dir,
        out_path=out_path,
        context_steps=context_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        arch=args.arch,
        train_frac=args.train_frac,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        pos_weight_coll=args.pos_weight_coll,
        pos_weight_inc=args.pos_weight_inc,
    )


if __name__ == "__main__":
    main()
