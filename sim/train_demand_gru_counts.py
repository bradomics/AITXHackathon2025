from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim.common import how_sin_cos, load_controls
from sim.model_gru import DemandGRUConfig, build_model, save_checkpoint


def _pick_device(s: str | None) -> torch.device:
    if not s or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _parse_dt(row: dict[str, str], *, interval_s: int) -> datetime | None:
    try:
        y = int(float(row["year"]))
        m = int(float(row["month"]))
        d = int(float(row["day"]))
        hh = int(float(row["hour"]))
        mm = int(float(row["minute"]))
    except Exception:
        return None

    dt = datetime(y, m, d, hh, mm)
    if interval_s >= 60:
        # Floor to the interval boundary (radar data sometimes reports e.g. :29/:59).
        minutes = int(interval_s // 60)
        if minutes > 1:
            dt = dt.replace(minute=(dt.minute // minutes) * minutes, second=0, microsecond=0)
        else:
            dt = dt.replace(second=0, microsecond=0)
    return dt


def _iter_gz_month_files(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.glob("*.csv.gz") if p.is_file()])


def _load_counts(
    *,
    controls_path: Path,
    counts_root: Path,
    use: str,
    start_dt: datetime,
    end_dt: datetime,
    bin_duration_s: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    cfg = load_controls(controls_path)
    interval_s = int(round(float(cfg.control_interval_s)))
    if interval_s <= 0:
        raise ValueError("controls config control_interval_s must be > 0")
    if interval_s != int(bin_duration_s):
        raise ValueError(
            f"counts training expects control_interval_s == bin_duration_s; got {interval_s} vs {bin_duration_s}"
        )

    n_controls = len(cfg.controls)
    if n_controls <= 0:
        raise ValueError("no controls in controls config")

    # Map count source IDs -> control indices.
    radar_map: dict[str, list[int]] = {}
    cam_map: dict[str, list[int]] = {}
    for i, c in enumerate(cfg.controls):
        if c.radar_detids:
            for detid in c.radar_detids:
                radar_map.setdefault(str(detid), []).append(i)
        if c.camera_device_ids:
            for dev in c.camera_device_ids:
                cam_map.setdefault(str(dev), []).append(i)

    if use in ("radar", "both") and not radar_map:
        raise ValueError("use=radar requires at least one control with radar_detids")
    if use in ("camera", "both") and not cam_map:
        raise ValueError("use=camera requires at least one control with camera_device_ids")

    total_s = (end_dt - start_dt).total_seconds()
    if total_s <= 0:
        raise ValueError("--end must be after --start")
    t_steps = int(total_s // interval_s)
    if t_steps <= 0:
        raise ValueError("time range too small for interval")

    vph = torch.zeros((t_steps, n_controls), dtype=torch.float32)
    obs = torch.zeros((t_steps, n_controls), dtype=torch.bool)

    def ingest_dir(*, dir_path: Path, id_field: str, id_map: dict[str, list[int]]) -> None:
        files = _iter_gz_month_files(dir_path)
        if not files:
            raise ValueError(f"no .csv.gz files found under {dir_path}")
        for p in files:
            with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    key = (row.get(id_field) or "").strip()
                    if not key:
                        continue
                    idxs = id_map.get(key)
                    if not idxs:
                        continue

                    dt = _parse_dt(row, interval_s=interval_s)
                    if dt is None:
                        continue
                    if dt < start_dt or dt >= end_dt:
                        continue

                    try:
                        vol = float(row.get("volume") or 0.0)
                    except Exception:
                        continue

                    # volume per bin -> veh/hour
                    veh_per_hour = vol * (3600.0 / float(bin_duration_s))
                    t = int(((dt - start_dt).total_seconds()) // interval_s)
                    if t < 0 or t >= t_steps:
                        continue
                    for ci in idxs:
                        vph[t, ci] += float(veh_per_hour)
                        obs[t, ci] = True

    if use in ("radar", "both"):
        ingest_dir(
            dir_path=counts_root / "i626-g7ub",
            id_field="detid",
            id_map=radar_map,
        )
    if use in ("camera", "both"):
        ingest_dir(
            dir_path=counts_root / "sh59-i6y9",
            id_field="atd_device_id",
            id_map=cam_map,
        )

    # Baselines: controls baseline_veh_per_hour if provided; otherwise mean over observed bins.
    base: list[float] = []
    for i, c in enumerate(cfg.controls):
        if float(c.baseline_veh_per_hour) > 0:
            base.append(float(c.baseline_veh_per_hour))
            continue
        mask = obs[:, i]
        n = int(mask.sum().item())
        if n <= 0:
            base.append(1.0)
            continue
        base.append(float(vph[:, i][mask].mean().item()))

    base_t = torch.tensor(base, dtype=torch.float32).clamp(min=1e-6)
    ratio = torch.ones_like(vph)
    ratio[obs] = (vph / base_t)[obs]
    ratio = torch.clamp(ratio, min=0.0)

    # Exogenous features (hour-of-week sin/cos).
    exog = torch.zeros((t_steps, 2), dtype=torch.float32)
    for t in range(t_steps):
        dt = start_dt + timedelta(seconds=interval_s * t)
        s, c = how_sin_cos(dt)
        exog[t, 0] = float(s)
        exog[t, 1] = float(c)

    return ratio, exog, obs, base


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tiny GRU forecaster on Austin traffic counts (radar/camera).")
    ap.add_argument("--controls", required=True, help="Controls JSON (must include radar_detids and/or camera_device_ids)")
    ap.add_argument("--out", default="sim/artifacts/demand_gru_counts.pt")
    ap.add_argument("--counts-root", default="data/bronze/austin_traffic_counts/counts")
    ap.add_argument("--use", choices=["radar", "camera", "both"], default="radar")
    ap.add_argument("--start", default="2019-01-01T00:00:00")
    ap.add_argument("--end", default="2019-02-01T00:00:00")
    ap.add_argument("--bin-duration-s", type=int, default=900)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50, help="Print a step log every N batches (0=off)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-mult-min", type=float, default=-2.0)
    ap.add_argument("--log-mult-max", type=float, default=2.0)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--write-controls-out", default="", help="Optional path to write baselines-filled controls JSON")
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    controls_path = Path(args.controls)
    ratio, exog, obs, base = _load_counts(
        controls_path=controls_path,
        counts_root=Path(args.counts_root),
        use=str(args.use),
        start_dt=datetime.fromisoformat(str(args.start)),
        end_dt=datetime.fromisoformat(str(args.end)),
        bin_duration_s=int(args.bin_duration_s),
    )

    cfg = load_controls(controls_path)
    n_controls = len(cfg.controls)
    context_steps = int(cfg.context_steps)
    interval_s = float(cfg.control_interval_s)

    cover = (obs.sum(dim=0).tolist() if n_controls > 0 else [])
    print(f"[counts] controls={n_controls} interval_s={interval_s} steps={ratio.shape[0]} context_steps={context_steps}")
    for i, c in enumerate(cfg.controls):
        print(
            f"[counts] control[{i}] {c.control_id} obs_bins={int(cover[i])} baseline_vph={base[i]:.2f}",
            flush=True,
        )

    if args.write_controls_out:
        raw = json.loads(controls_path.read_text(encoding="utf-8"))
        for i, c in enumerate(raw.get("controls", [])):
            if float(c.get("baseline_veh_per_hour", 0.0)) <= 0.0:
                c["baseline_veh_per_hour"] = float(base[i])
        outp = Path(args.write_controls_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"[counts] wrote {outp}", flush=True)

    # Build model
    device = _pick_device(args.device)
    model_cfg = DemandGRUConfig(
        n_controls=int(n_controls),
        context_steps=int(context_steps),
        hidden_dim=int(args.hidden_dim),
        log_multiplier_min=float(args.log_mult_min),
        log_multiplier_max=float(args.log_mult_max),
    )
    model = build_model(model_cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = torch.nn.MSELoss()

    eps = 1e-4
    t_steps = int(ratio.shape[0])
    valid = [t for t in range(context_steps, t_steps) if bool(obs[t].any().item())]
    if not valid:
        raise SystemExit("no observed bins found in the requested time range (check controls + dates)")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        for step in range(1, int(args.batches_per_epoch) + 1):
            ts = [random.choice(valid) for _ in range(int(args.batch_size))]
            x = torch.empty((len(ts), context_steps, n_controls + 2), dtype=torch.float32, device=device)
            y = torch.empty((len(ts), n_controls), dtype=torch.float32, device=device)
            for b, t in enumerate(ts):
                xr = ratio[t - context_steps : t, :]
                xe = exog[t - context_steps : t, :]
                x[b] = torch.cat([xr, xe], dim=-1).to(device)
                y[b] = torch.log(torch.clamp(ratio[t, :], min=eps)).to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())
            if int(args.log_every) > 0 and step % int(args.log_every) == 0:
                print(
                    f"[train] epoch={epoch} step={step}/{int(args.batches_per_epoch)} loss={float(loss.detach().cpu().item()):.6f}",
                    flush=True,
                )

        avg = total / max(int(args.batches_per_epoch), 1)
        print(f"[train] epoch={epoch} mse={avg:.6f}", flush=True)
        save_checkpoint(Path(args.out), model=model)
        print(f"[train] wrote {Path(args.out)}", flush=True)

    # Final checkpoint is already written at the end of the last epoch.


if __name__ == "__main__":
    main()
