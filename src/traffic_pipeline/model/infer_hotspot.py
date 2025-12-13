from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from traffic_pipeline.config import load_config
from traffic_pipeline.model.model_hotspot import build_model
from traffic_pipeline.util import parse_dt


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _sin_cos(pos: float, period: float) -> tuple[float, float]:
    import math

    ang = 2.0 * math.pi * (pos / period)
    return math.sin(ang), math.cos(ang)


def _season_id(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _feature_row(bucket: datetime, *, feature_names: list[str], weather: dict[str, float]) -> list[float]:
    hour = bucket.hour
    dow = bucket.weekday()
    how = dow * 24 + hour
    month = bucket.month
    season = _season_id(month)

    hour_sin, hour_cos = _sin_cos(hour, 24.0)
    dow_sin, dow_cos = _sin_cos(dow, 7.0)
    how_sin, how_cos = _sin_cos(how, 168.0)
    month_sin, month_cos = _sin_cos(month - 1, 12.0)

    time_map = {
        "hour_of_day": float(hour),
        "day_of_week": float(dow),
        "hour_of_week": float(how),
        "month": float(month),
        "season": float(season),
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "dow_sin": float(dow_sin),
        "dow_cos": float(dow_cos),
        "how_sin": float(how_sin),
        "how_cos": float(how_cos),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
    }

    out: list[float] = []
    for name in feature_names:
        if name in time_map:
            out.append(time_map[name])
        else:
            out.append(float(weather.get(name, float("nan"))))
    return out


def _load_cells(cells_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    lats: list[float] = []
    lons: list[float] = []
    with cells_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["center_lat"]))
            lons.append(float(row["center_lon"]))
    return np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32)


def _load_weather_history_by_bucket(
    weather_hourly_csv: Path, *, dt_format: str, weather_cols: list[str]
) -> dict[datetime, dict[str, float]]:
    out: dict[datetime, dict[str, float]] = {}
    with weather_hourly_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bucket_s = (row.get("bucket_start") or "").strip()
            if not bucket_s:
                continue
            try:
                bucket = parse_dt(bucket_s, datetime_format=dt_format)
            except ValueError:
                continue

            out[bucket] = {c: _to_float(row.get(c) or "") for c in weather_cols}
    return out


def _load_forecast_rows(forecast_csv: Path, *, dt_format: str, weather_cols: list[str]) -> list[tuple[datetime, dict[str, float]]]:
    rows: list[tuple[datetime, dict[str, float]]] = []
    with forecast_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fd = (row.get("formatted_date") or "").strip()
            if not fd:
                continue
            try:
                bucket = parse_dt(fd, datetime_format=dt_format).replace(minute=0, second=0, microsecond=0)
            except ValueError:
                continue
            rows.append(
                (
                    bucket,
                    {c: _to_float(row.get(c) or "") for c in weather_cols},
                )
            )
    rows.sort(key=lambda x: x[0])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run hotspot inference and emit HeatPoint arrays")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--model", default="artifacts/h3_hotspot_model.pt")
    ap.add_argument("--forecast-csv", default="data/bronze/austin_forecast_live.csv")
    ap.add_argument("--out", default="output/phase1_output.json")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--horizon-index", type=int, default=0, help="0=first forecast hour, 1=second, etc.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model_cfg = payload["model_config"]
    n_features = int(model_cfg["n_features"])
    n_cells = int(model_cfg["n_cells"])
    d_model = int(model_cfg["d_model"])
    arch = str(model_cfg["arch"])
    context_steps = int(model_cfg["context_steps"])

    model = build_model(n_features=n_features, n_cells=n_cells, d_model=d_model, arch=arch)
    model.load_state_dict(payload["model_state"])
    model.eval()

    data_dir = Path(payload.get("data_dir") or cfg.tokenizer.output_dir)
    cells_csv = data_dir / "cells.csv"
    if not cells_csv.exists():
        raise FileNotFoundError(cells_csv)
    cell_lats, cell_lons = _load_cells(cells_csv)

    x_mean = payload["x_mean"].astype(np.float32)
    x_std = payload["x_std"].astype(np.float32)
    tokenizer_meta = payload.get("tokenizer_meta") or {}
    feature_names = list(tokenizer_meta.get("feature_names") or [])
    if not feature_names:
        raise ValueError("Checkpoint missing tokenizer_meta.feature_names")
    if x_mean.shape[0] != len(feature_names) or x_std.shape[0] != len(feature_names):
        raise ValueError(
            f"Checkpoint x_mean/x_std shape mismatch: mean={x_mean.shape} std={x_std.shape} features={len(feature_names)}"
        )

    weather_cols = [n for n in feature_names if n not in {"hour_of_day", "day_of_week", "hour_of_week", "month", "season", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "how_sin", "how_cos", "month_sin", "month_cos"}]
    hist_by_bucket = _load_weather_history_by_bucket(
        cfg.paths.silver_dir / cfg.silverize.weather_output_name,
        dt_format=cfg.silverize.datetime_format,
        weather_cols=weather_cols,
    )

    fc = _load_forecast_rows(
        Path(args.forecast_csv),
        dt_format=cfg.silverize.datetime_format,
        weather_cols=weather_cols,
    )
    if not fc:
        raise ValueError(f"No forecast rows found in {args.forecast_csv}")

    horizon_i = int(args.horizon_index)
    if horizon_i < 0 or horizon_i >= len(fc):
        raise ValueError(f"horizon-index out of range: {horizon_i} (rows={len(fc)})")

    target_bucket, target_weather = fc[horizon_i]

    # Context ending at target_bucket (history for prior hours, forecast for the target hour).
    buckets = [target_bucket - timedelta(hours=k) for k in range(context_steps - 1, 0, -1)] + [target_bucket]
    x_rows = []
    for b in buckets:
        w = target_weather if b == target_bucket else hist_by_bucket.get(b, {})
        x_rows.append(_feature_row(b, feature_names=feature_names, weather=w))

    x = np.asarray(x_rows, dtype=np.float32)
    x = np.where(np.isfinite(x), x, x_mean).astype(np.float32)
    x = (x - x_mean) / x_std

    with torch.no_grad():
        logits_coll, logits_inc = model(torch.from_numpy(x).unsqueeze(0))
        p_coll = torch.sigmoid(logits_coll)
        p_inc = torch.sigmoid(logits_inc)
    p_coll = p_coll.squeeze(0).numpy()
    p_inc = p_inc.squeeze(0).numpy()

    def top_points(p: np.ndarray) -> list[tuple[float, float, float]]:
        idx = np.argsort(-p)[: max(0, int(args.top_k))]
        pts: list[tuple[float, float, float]] = []
        for i in idx:
            w = float(p[i])
            if w < float(args.threshold):
                continue
            pts.append((float(cell_lats[i]), float(cell_lons[i]), w))
        return pts

    coll_pts = top_points(p_coll)
    inc_pts = top_points(p_inc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")

    def emit(name: str, pts: list[tuple[float, float, float]]) -> str:
        lines = [f"const {name}: HeatPoint[] = ["]
        for lat, lon, w in pts:
            lines.append(f"    {{ position: [{lon:.6f}, {lat:.6f}], weight: {w:.6f} }},")
        lines.append("];")
        return "\n".join(lines)

    text = (
        f"// Generated {ts}\n"
        f"// target_bucket={target_bucket.isoformat(timespec='seconds')}\n"
        + emit("COLLISION_POINTS", coll_pts)
        + "\n\n"
        + emit("INCIDENT_POINTS", inc_pts)
        + "\n"
    )
    out_path.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "target_bucket": target_bucket.isoformat(timespec="seconds"),
                "collisions_points": len(coll_pts),
                "incidents_points": len(inc_pts),
            }
        )
    )


if __name__ == "__main__":
    main()
