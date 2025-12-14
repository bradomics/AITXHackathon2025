from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import load_config
from model.model_hotspot_v2 import build_hotspot_v2_model
from model.ar_buffer_v2 import (
    compute_ar_features_v2,
    load_ar_state_v2,
    save_ar_state_v2,
    seed_ar_state_v2_from_silver,
    update_ar_state_v2,
)
from util import floor_dt, parse_dt


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _sin_cos(pos: float, period: float) -> tuple[float, float]:
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


def _time_features(bucket: datetime) -> tuple[dict[str, float], int]:
    how = bucket.weekday() * 24 + bucket.hour
    angle = 2.0 * math.pi * (how / 168.0)
    try:
        import holidays  # type: ignore
    except Exception:  # pragma: no cover
        holidays = None  # type: ignore[assignment]
    us_holidays = holidays.US() if holidays is not None else None
    is_holiday = 1.0 if us_holidays is not None and bucket.date() in us_holidays else 0.0
    return {
        "hour_of_week": float(how),
        "how_sin": float(math.sin(angle)),
        "how_cos": float(math.cos(angle)),
        "month": float(bucket.month),
        "season": float(_season_id(bucket.month)),
        "is_holiday": float(is_holiday),
    }, int(how)


def _logit(p: float) -> float:
    p = float(p)
    if not (p == p):
        return float("nan")
    if p <= 0.0:
        p = 1e-9
    if p >= 1.0:
        p = 1.0 - 1e-9
    return float(math.log(p / (1.0 - p)))


def _calibration_shift(*, true_rate: float, sample_rate: float, pos_weight: float) -> float:
    """
    Adjust logits trained on a sampled/weighted distribution to approximate full-grid base rates.

    If pos_weight > 1, training is approximately equivalent to duplicating positives by pos_weight,
    so we first convert sample_rate -> effective_rate under that weighting.
    """
    w = float(pos_weight) if pos_weight and pos_weight > 0 else 1.0
    p = float(sample_rate)
    if not (p == p) or p <= 0.0 or p >= 1.0:
        return 0.0
    eff = (w * p) / ((w * p) + (1.0 - p))
    lt = _logit(true_rate)
    ls = _logit(eff)
    if not (lt == lt) or not (ls == ls):
        return 0.0
    return float(lt - ls)


def _load_cells(cells_csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    h3_cells: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    with cells_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h3_cells.append(str(row.get("h3_cell") or "").strip())
            lats.append(float(row["center_lat"]))
            lons.append(float(row["center_lon"]))
    return h3_cells, np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32)


def _load_forecast_rows(
    forecast_csv: Path, *, dt_format: str, weather_cols: list[str]
) -> list[tuple[datetime, dict[str, float]]]:
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
            rows.append((bucket, {c: _to_float(row.get(c) or "") for c in weather_cols}))
    rows.sort(key=lambda x: x[0])
    return rows


def _bucket_weather(
    fc: list[tuple[datetime, dict[str, float]]],
    *,
    bucket_start: datetime,
    bucket_minutes: int,
    weather_cols: list[str],
) -> dict[str, float]:
    end = bucket_start + timedelta(minutes=int(bucket_minutes))
    sums = {c: 0.0 for c in weather_cols}
    counts = {c: 0 for c in weather_cols}
    for dt, w in fc:
        if dt < bucket_start or dt >= end:
            continue
        for c in weather_cols:
            v = float(w.get(c, float("nan")))
            if v == v:
                sums[c] += v
                counts[c] += 1
    out: dict[str, float] = {}
    for c in weather_cols:
        n = counts[c]
        out[c] = float("nan") if n <= 0 else float(sums[c] / float(n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v2 hotspot inference and emit HeatPoint arrays")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--model", default="artifacts/h3_hotspot_v2_model.pt")
    ap.add_argument("--forecast-csv", default="data/bronze/austin_forecast_live.csv")
    ap.add_argument("--out", default="output/phase1_output.json")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--horizon-index", type=int, default=0, help="0=first forecast hour, 1=second, etc.")
    ap.add_argument("--runtime-dir", default=None, help="Defaults to config hotspot_v2.runtime_dir")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model_cfg = payload["model_config"]
    n_features = int(model_cfg["n_features"])
    hidden_dim = int(model_cfg["hidden_dim"])
    dropout = float(model_cfg.get("dropout") or 0.0)

    feature_names = list(payload.get("feature_names") or [])
    if not feature_names:
        raise ValueError("Checkpoint missing feature_names")
    if len(feature_names) != n_features:
        raise ValueError(f"Checkpoint n_features mismatch: {n_features} vs feature_names={len(feature_names)}")

    x_mean = payload["x_mean"].astype(np.float32)
    x_std = payload["x_std"].astype(np.float32)
    if x_mean.shape[0] != n_features or x_std.shape[0] != n_features:
        raise ValueError("Checkpoint x_mean/x_std shape mismatch")

    cal = payload.get("calibration") or {}
    train_cfg = payload.get("train") or {}
    if not isinstance(cal, dict):
        cal = {}
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    pos_weight_coll = float(train_cfg.get("pos_weight_coll") or 1.0)
    pos_weight_inc = float(train_cfg.get("pos_weight_inc") or 1.0)

    shift_coll = 0.0
    shift_inc = 0.0
    if "true_rate_coll" in cal and "sample_rate_coll" in cal:
        shift_coll = _calibration_shift(
            true_rate=float(cal.get("true_rate_coll") or 0.0),
            sample_rate=float(cal.get("sample_rate_coll") or 0.0),
            pos_weight=pos_weight_coll,
        )
    if "true_rate_inc" in cal and "sample_rate_inc" in cal:
        shift_inc = _calibration_shift(
            true_rate=float(cal.get("true_rate_inc") or 0.0),
            sample_rate=float(cal.get("sample_rate_inc") or 0.0),
            pos_weight=pos_weight_inc,
        )

    data_dir = Path(payload.get("data_dir") or cfg.hotspot_v2.output_dir)
    cells_csv = data_dir / "cells.csv"
    static_npz = data_dir / "cell_static.npz"
    if not cells_csv.exists():
        raise FileNotFoundError(cells_csv)
    if not static_npz.exists():
        raise FileNotFoundError(static_npz)

    if not shift_coll and not shift_inc:
        ds_path = data_dir / "dataset.npz"
        meta_path = data_dir / "meta.json"
        if ds_path.exists() and meta_path.exists():
            try:
                ds = np.load(ds_path)
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                n_cells_meta = int(meta.get("n_cells") or 0)
                n_buckets_meta = int(meta.get("n_buckets") or 0)
                if n_cells_meta > 0 and n_buckets_meta > 0:
                    y_coll = ds["y_coll"].astype(np.float32)
                    y_inc = ds["y_inc"].astype(np.float32)
                    pos_coll = float(y_coll.sum())
                    pos_inc = float(y_inc.sum())
                    n_samples = max(int(y_coll.shape[0]), 1)
                    sample_rate_coll = pos_coll / float(n_samples)
                    sample_rate_inc = pos_inc / float(n_samples)
                    true_rate_coll = pos_coll / float(n_cells_meta * n_buckets_meta)
                    true_rate_inc = pos_inc / float(n_cells_meta * n_buckets_meta)
                    shift_coll = _calibration_shift(
                        true_rate=true_rate_coll,
                        sample_rate=sample_rate_coll,
                        pos_weight=pos_weight_coll,
                    )
                    shift_inc = _calibration_shift(
                        true_rate=true_rate_inc,
                        sample_rate=sample_rate_inc,
                        pos_weight=pos_weight_inc,
                    )
            except Exception:
                pass

    h3_cells, cell_lats, cell_lons = _load_cells(cells_csv)
    n_cells = int(cell_lats.shape[0])
    cell_to_idx = {c: i for i, c in enumerate(h3_cells) if c}

    static = np.load(static_npz)
    aadt_log1p = static["aadt_log1p"].astype(np.float32)
    aadt_distance_km = static["aadt_distance_km"].astype(np.float32)
    has_aadt = static["has_aadt"].astype(np.float32)
    radar_distance_km = static["radar_distance_km"].astype(np.float32)
    has_radar = static["has_radar"].astype(np.float32)
    radar_vol_baseline = static["radar_vol_baseline"].astype(np.float32)
    radar_speed_baseline = static["radar_speed_baseline"].astype(np.float32)
    radar_occ_baseline = static["radar_occ_baseline"].astype(np.float32)

    if aadt_log1p.shape[0] != n_cells or radar_vol_baseline.shape[0] != n_cells:
        raise ValueError("cell_static.npz shape mismatch")

    time_cols = {"hour_of_week", "how_sin", "how_cos", "month", "season", "is_holiday"}
    weather_cols = [n for n in feature_names if n not in time_cols]
    # Remove non-weather feature names from the list.
    non_weather = {
        "aadt_log1p",
        "aadt_distance_km",
        "has_aadt",
        "radar_vol_baseline",
        "radar_speed_baseline",
        "radar_occ_baseline",
        "radar_distance_km",
        "has_radar",
        "ema_collisions",
        "ema_traffic_incidents",
        "hrs_since_last_collision",
        "hrs_since_last_traffic_incident",
    }
    weather_cols = [c for c in weather_cols if c not in non_weather and not c.startswith("n_collisions_lb_") and not c.startswith("n_traffic_incidents_lb_")]

    fc = _load_forecast_rows(Path(args.forecast_csv), dt_format=cfg.silverize.datetime_format, weather_cols=weather_cols)
    if not fc:
        raise ValueError(f"No forecast rows found in {args.forecast_csv}")
    horizon_i = int(args.horizon_index)
    if horizon_i < 0 or horizon_i >= len(fc):
        raise ValueError(f"horizon-index out of range: {horizon_i} (rows={len(fc)})")

    forecast_dt, _ = fc[horizon_i]
    target_bucket = floor_dt(forecast_dt, bucket_minutes=int(cfg.features.bucket_minutes))
    target_weather = _bucket_weather(
        fc,
        bucket_start=target_bucket,
        bucket_minutes=int(cfg.features.bucket_minutes),
        weather_cols=weather_cols,
    )
    time_map, how = _time_features(target_bucket)

    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else cfg.hotspot_v2.runtime_dir
    state_path = runtime_dir / "ar_state.json"
    incidents_ndjson = runtime_dir / "incidents.ndjson"

    state = load_ar_state_v2(state_path=state_path, n_cells=n_cells, bucket_minutes=int(cfg.features.bucket_minutes))
    if (
        state.last_ema_bucket_id < 0
        and not state.events_by_bucket
        and int(state.processed_bytes) <= 0
        and (not incidents_ndjson.exists() or incidents_ndjson.stat().st_size <= 0)
    ):
        seed_ar_state_v2_from_silver(
            state=state,
            incidents_csv=cfg.paths.silver_dir / cfg.silverize.incidents_output_name,
            cell_to_idx=cell_to_idx,
            h3_resolution=int(cfg.tokenizer.h3_resolution),
            datetime_format=cfg.silverize.datetime_format,
            target_bucket=target_bucket,
            ema_half_life_hours=float(cfg.features.ema_half_life_hours),
            lookback_hours=cfg.features.lookback_hours,
            seed_hours=int(cfg.hotspot_v2.seed_hours),
            bucket_minutes=int(cfg.features.bucket_minutes),
        )
    update_ar_state_v2(
        state=state,
        incidents_ndjson=incidents_ndjson,
        cell_to_idx=cell_to_idx,
        h3_resolution=int(cfg.tokenizer.h3_resolution),
        datetime_format=cfg.silverize.datetime_format,
        ema_half_life_hours=float(cfg.features.ema_half_life_hours),
        lookback_hours=cfg.features.lookback_hours,
        bucket_minutes=int(cfg.features.bucket_minutes),
    )
    ar = compute_ar_features_v2(
        state=state,
        target_bucket=target_bucket,
        n_cells=n_cells,
        lookback_hours=cfg.features.lookback_hours,
        ema_half_life_hours=float(cfg.features.ema_half_life_hours),
        bucket_minutes=int(cfg.features.bucket_minutes),
    )
    save_ar_state_v2(state_path=state_path, state=state, n_cells=n_cells)

    X = np.zeros((n_cells, n_features), dtype=np.float32)
    col = {n: i for i, n in enumerate(feature_names)}

    for k, v in time_map.items():
        if k in col:
            X[:, col[k]] = float(v)

    for c in weather_cols:
        i = col.get(c)
        if i is None:
            continue
        X[:, i] = float(target_weather.get(c, float("nan")))

    def fill(name: str, arr: np.ndarray) -> None:
        i = col.get(name)
        if i is None:
            return
        X[:, i] = arr.astype(np.float32, copy=False)

    fill("aadt_log1p", aadt_log1p)
    fill("aadt_distance_km", aadt_distance_km)
    fill("has_aadt", has_aadt)
    fill("radar_distance_km", radar_distance_km)
    fill("has_radar", has_radar)

    if "radar_vol_baseline" in col:
        X[:, col["radar_vol_baseline"]] = radar_vol_baseline[:, how]
    if "radar_speed_baseline" in col:
        X[:, col["radar_speed_baseline"]] = radar_speed_baseline[:, how]
    if "radar_occ_baseline" in col:
        X[:, col["radar_occ_baseline"]] = radar_occ_baseline[:, how]

    for name, arr in ar.items():
        fill(name, arr)

    X = np.where(np.isfinite(X), X, x_mean).astype(np.float32)
    Xn = (X - x_mean) / x_std

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_hotspot_v2_model(n_features=n_features, hidden_dim=hidden_dim, dropout=dropout).to(device_t)
    model.load_state_dict(payload["model_state"])
    model.eval()

    p_coll = np.zeros(n_cells, dtype=np.float32)
    p_inc = np.zeros(n_cells, dtype=np.float32)
    bs = max(int(args.batch_size), 1)

    with torch.no_grad():
        for start in range(0, n_cells, bs):
            end = min(start + bs, n_cells)
            xb = torch.from_numpy(Xn[start:end]).to(device_t)
            logits_coll, logits_inc = model(xb)
            if shift_coll:
                logits_coll = logits_coll + float(shift_coll)
            if shift_inc:
                logits_inc = logits_inc + float(shift_inc)
            p_coll[start:end] = torch.sigmoid(logits_coll).detach().cpu().numpy()
            p_inc[start:end] = torch.sigmoid(logits_inc).detach().cpu().numpy()

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
        f"// v2_calibration_shift_coll={shift_coll:.6f} v2_calibration_shift_inc={shift_inc:.6f}\n"
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
