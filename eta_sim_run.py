from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sim.common import load_controls


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA SIM: train the Torch demand forecaster (synthetic or counts).")
    ap.add_argument("--mode", choices=["synthetic", "counts"], default="counts")
    ap.add_argument("--controls", default="sim/controls_austin_radar_auto.json")
    ap.add_argument("--out", default="sim/artifacts/demand_gru_counts_austin.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batches-per-epoch", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cpu", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=13)
    # Counts-mode knobs (kept minimal; see sim/train_demand_gru_counts.py for details).
    ap.add_argument("--counts-root", default="data/bronze/austin_traffic_counts/counts")
    ap.add_argument("--use", choices=["radar", "camera", "both"], default="radar")
    ap.add_argument("--start", default="2019-01-01T00:00:00")
    ap.add_argument("--end", default="2019-02-01T00:00:00")
    ap.add_argument("--bin-duration-s", type=int, default=900)
    ap.add_argument("--log-mult-min", type=float, default=-2.0)
    ap.add_argument("--log-mult-max", type=float, default=2.0)
    ap.add_argument("--write-controls-out", default="sim/controls_austin_radar_auto_filled.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    controls_path = repo_root / args.controls

    if args.mode == "synthetic":
        cfg = load_controls(controls_path)
        n_controls = len(cfg.controls)
        if n_controls <= 0:
            raise SystemExit(f"No controls found in {args.controls}")

        cmd = [
            python,
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
            "--log-every",
            str(int(args.log_every)),
            "--lr",
            str(float(args.lr)),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
        ]
    else:
        cmd = [
            python,
            str(repo_root / "sim" / "train_demand_gru_counts.py"),
            "--controls",
            str(controls_path),
            "--out",
            str(repo_root / args.out),
            "--counts-root",
            str(repo_root / args.counts_root),
            "--use",
            str(args.use),
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--bin-duration-s",
            str(int(args.bin_duration_s)),
            "--hidden-dim",
            str(int(args.hidden_dim)),
            "--epochs",
            str(int(args.epochs)),
            "--batches-per-epoch",
            str(int(args.batches_per_epoch)),
            "--batch-size",
            str(int(args.batch_size)),
            "--log-every",
            str(int(args.log_every)),
            "--lr",
            str(float(args.lr)),
            "--log-mult-min",
            str(float(args.log_mult_min)),
            "--log-mult-max",
            str(float(args.log_mult_max)),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
        ]
        if str(args.write_controls_out).strip():
            cmd += ["--write-controls-out", str(repo_root / str(args.write_controls_out))]

    print("[eta_sim_run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
