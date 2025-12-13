from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traffic_pipeline.config import load_config
from traffic_pipeline.feature_factory import build_hotspot_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Silver -> Gold: build hotspot features")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    silver_incidents_csv = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_hourly_csv = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    aadt_stations_csv = cfg.paths.silver_dir / cfg.silverize.aadt_output_name
    out_csv = cfg.paths.gold_dir / "features" / "hotspot_features.csv"

    stats = build_hotspot_features(
        silver_incidents_csv=silver_incidents_csv,
        weather_hourly_csv=weather_hourly_csv,
        aadt_stations_csv=aadt_stations_csv,
        out_csv=out_csv,
        bucket_minutes=cfg.features.bucket_minutes,
        cell_round_decimals=cfg.features.cell_round_decimals,
        lookback_hours=cfg.features.lookback_hours,
        label_horizon_hours=cfg.features.label_horizon_hours,
        aadt_max_distance_km=cfg.features.aadt_max_distance_km,
        ema_half_life_hours=cfg.features.ema_half_life_hours,
    )

    print(
        "[feature_factory] "
        f"silver_rows={stats.silver_rows_read} collisions_used={stats.collisions_used} "
        f"traffic_incidents_used={stats.traffic_incidents_used} "
        f"cells={stats.unique_cells} cell_buckets={stats.unique_cell_buckets} "
        f"out_rows={stats.output_rows} -> {out_csv}"
    )


if __name__ == "__main__":
    main()
