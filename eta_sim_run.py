from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sim.common import load_controls


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA SIM: train the Torch demand forecaster (synthetic).")
    ap.add_argument("--controls", default="sim/controls_example.json")
    ap.add_argument("--out", default="sim/artifacts/demand_gru.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_controls(repo_root / args.controls)
    n_controls = len(cfg.controls)
    if n_controls <= 0:
        raise SystemExit(f"No controls found in {args.controls}")

    cmd = [
        sys.executable,
        str(repo_root / "sim" / "train_demand_gru.py"),
        "--out",
        str(repo_root / args.out),
        "--n-controls",
        str(n_controls),
        "--context-steps",
        str(int(cfg.context_steps)),
        "--control-interval-s",
        str(float(cfg.control_interval_s)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--epochs",
        str(int(args.epochs)),
        "--batches-per-epoch",
        str(int(args.batches_per_epoch)),
        "--batch-size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
    ]

    print("[eta_sim_run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()

