from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from traffic_pipeline.config import load_config
from traffic_pipeline.model.tokenizer_h3 import tokenize_h3_time_series


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize incidents into an H3 hex grid time series dataset")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    incidents_csv = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_csv = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    out_dir = cfg.tokenizer.output_dir

    stats = tokenize_h3_time_series(
        incidents_csv=incidents_csv,
        weather_hourly_csv=weather_csv,
        out_dir=out_dir,
        h3_resolution=cfg.tokenizer.h3_resolution,
        austin_center_lat=cfg.tokenizer.austin_center_lat,
        austin_center_lon=cfg.tokenizer.austin_center_lon,
        austin_radius_km=cfg.tokenizer.austin_radius_km,
        context_steps=cfg.train.context_steps,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=cfg.features.bucket_minutes,
    )

    print(
        "[tokenize_h3] "
        f"t={stats.n_time_steps} cells={stats.n_cells} "
        f"coll_events={stats.collisions_events} inc_events={stats.traffic_incidents_events} "
        f"coll_pairs={stats.collisions_pairs} inc_pairs={stats.traffic_incidents_pairs} "
        f"skipped_outside_radius={stats.skipped_outside_radius} "
        f"skipped_missing_bucket={stats.skipped_missing_time_bucket} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
