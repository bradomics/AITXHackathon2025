from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traffic_pipeline.config import load_config
from traffic_pipeline.aadt import silverize_aadt_stations


def main() -> None:
    ap = argparse.ArgumentParser(description="Bronze -> Silver: normalize TxDOT AADT station data")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = cfg.paths.silver_dir / cfg.silverize.aadt_output_name

    stats = silverize_aadt_stations(
        bronze_path=cfg.paths.aadt_bronze_path,
        out_path=out_path,
    )
    print(
        f"[silverize_aadt] input_rows={stats.input_rows} unique_stations={stats.unique_stations} "
        f"output_rows={stats.output_rows} -> {out_path}"
    )


if __name__ == "__main__":
    main()

