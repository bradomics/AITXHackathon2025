from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from model.train_hotspot_v2 import train_hotspot_v2_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train v2 hotspot predictor (MLP) on enriched H3 dataset")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--data-dir", default=None, help="Dataset dir (defaults to config hotspot_v2.output_dir)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--device", default=None, help="cpu|cuda (auto if omitted)")
    ap.add_argument("--pos-weight-coll", type=float, default=1.0)
    ap.add_argument("--pos-weight-inc", type=float, default=1.0)
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--logit-clamp", type=float, default=20.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="artifacts/h3_hotspot_v2_model.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else cfg.hotspot_v2.output_dir
    ds_path = data_dir / "dataset.npz"
    meta_path = data_dir / "meta.json"
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    out_path = Path(args.out)
    train_hotspot_v2_model(
        data_dir=data_dir,
        out_path=out_path,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_frac=args.train_frac,
        device=args.device,
        pos_weight_coll=args.pos_weight_coll,
        pos_weight_inc=args.pos_weight_inc,
        use_amp=not bool(args.no_amp),
        grad_clip_norm=float(args.grad_clip),
        logit_clamp=float(args.logit_clamp),
        log_every_steps=args.log_every,
    )


if __name__ == "__main__":
    main()
