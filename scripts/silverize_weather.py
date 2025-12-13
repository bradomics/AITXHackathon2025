from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traffic_pipeline.config import load_config
from traffic_pipeline.weather import silverize_weather_hourly


def main() -> None:
    ap = argparse.ArgumentParser(description="Bronze -> Silver: normalize hourly weather CSV")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--bronze-path", default=None, help="Override input CSV path")
    ap.add_argument("--out-name", default=None, help="Override output filename under data/silver")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bronze_path = Path(args.bronze_path) if args.bronze_path else cfg.paths.weather_bronze_path
    out_name = args.out_name if args.out_name else cfg.silverize.weather_output_name
    out_path = cfg.paths.silver_dir / out_name

    stats = silverize_weather_hourly(
        bronze_path=bronze_path,
        out_path=out_path,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=cfg.features.bucket_minutes,
    )
    print(f"[silverize_weather] input_rows={stats.input_rows} output_rows={stats.output_rows} -> {out_path}")


if __name__ == "__main__":
    main()
