from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traffic_pipeline.config import load_config
from traffic_pipeline.silverize import silverize_incidents
from traffic_pipeline.weather import silverize_weather_hourly
from traffic_pipeline.aadt import silverize_aadt_stations
from traffic_pipeline.feature_factory import build_hotspot_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Run bronze -> silver -> gold pipeline")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    incidents_path = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_path = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    aadt_path = cfg.paths.silver_dir / cfg.silverize.aadt_output_name
    features_path = cfg.paths.gold_dir / "features" / "hotspot_features.csv"

    stats_inc = silverize_incidents(
        bronze_dir=cfg.paths.bronze_dir,
        out_path=incidents_path,
        datetime_format=cfg.silverize.datetime_format,
    )
    print(
        f"[run_pipeline] incidents files={stats_inc.input_files} rows={stats_inc.output_rows} -> {incidents_path}"
    )

    stats_w = silverize_weather_hourly(
        bronze_path=cfg.paths.weather_bronze_path,
        out_path=weather_path,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=cfg.features.bucket_minutes,
    )
    print(f"[run_pipeline] weather rows={stats_w.output_rows} -> {weather_path}")

    stats_a = silverize_aadt_stations(
        bronze_path=cfg.paths.aadt_bronze_path,
        out_path=aadt_path,
    )
    print(f"[run_pipeline] aadt stations={stats_a.unique_stations} -> {aadt_path}")

    stats_f = build_hotspot_features(
        silver_incidents_csv=incidents_path,
        weather_hourly_csv=weather_path,
        aadt_stations_csv=aadt_path,
        out_csv=features_path,
        bucket_minutes=cfg.features.bucket_minutes,
        cell_round_decimals=cfg.features.cell_round_decimals,
        lookback_hours=cfg.features.lookback_hours,
        label_horizon_hours=cfg.features.label_horizon_hours,
        aadt_max_distance_km=cfg.features.aadt_max_distance_km,
        ema_half_life_hours=cfg.features.ema_half_life_hours,
    )
    print(
        "[run_pipeline] features "
        f"rows={stats_f.output_rows} cells={stats_f.unique_cells} -> {features_path}"
    )


if __name__ == "__main__":
    main()
