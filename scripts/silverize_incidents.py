import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traffic_pipeline.config import load_config
from traffic_pipeline.silverize import silverize_incidents


def main() -> None:
    ap = argparse.ArgumentParser(description="Bronze -> Silver: normalize incident CSVs")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bronze_dir = cfg.paths.bronze_dir
    silver_dir = cfg.paths.silver_dir

    out_path = silver_dir / cfg.silverize.incidents_output_name
    stats = silverize_incidents(
        bronze_dir=bronze_dir,
        out_path=out_path,
        datetime_format=cfg.silverize.datetime_format,
    )
    print(
        f"[silverize] files={stats.input_files} input_rows={stats.input_rows} "
        f"output_rows={stats.output_rows} deduped_rows={stats.deduped_rows} -> {out_path}"
    )


if __name__ == "__main__":
    main()
