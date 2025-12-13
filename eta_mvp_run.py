from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_config
from data_pipeline.aadt import silverize_aadt_stations
from data_pipeline.feature_factory import build_hotspot_features
from data_pipeline.silverize import silverize_incidents
from data_pipeline.weather import silverize_weather_hourly


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _venv_python() -> Path | None:
    root = Path(__file__).resolve().parent
    if sys.platform.startswith("win"):
        cand = root / ".venv" / "Scripts" / "python.exe"
    else:
        cand = root / ".venv" / "bin" / "python"
    return cand if cand.exists() else None


def _maybe_reexec_into_venv(*, required_modules: list[str]) -> None:
    if os.environ.get("ETA_MVP_REEXEC") == "1":
        return

    missing = sorted({m for m in required_modules if not _has_module(m)})
    if not missing:
        return

    venv_py = _venv_python()
    if venv_py is None:
        return

    try:
        current_py = Path(sys.executable).resolve()
    except FileNotFoundError:
        current_py = Path(sys.executable)

    if current_py == venv_py.resolve():
        return

    env = os.environ.copy()
    env["ETA_MVP_REEXEC"] = "1"
    print(f"[env] missing {', '.join(missing)}; re-exec -> {venv_py}", flush=True)
    os.execve(
        str(venv_py),
        [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def _exit_missing_deps(*, stage: str, missing: str) -> "NoReturn":
    raise SystemExit(
        "\n".join(
            [
                f"[{stage}] missing optional dependency: {missing}",
                "Tokenizer/training/inference require third-party deps (h3/torch/numpy).",
                "Run end-to-end with the venv python:",
                "  .venv/bin/python -m pip install -r requirements.txt",
                "  .venv/bin/python eta_mvp_run.py --config configs/pipeline.toml",
                "Or run ETL only (no h3/torch):",
                "  python eta_mvp_run.py --only-etl",
            ]
        )
    )


def _run_silver(*, config_path: str) -> None:
    cfg = load_config(config_path)

    incidents_path = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_path = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    aadt_path = cfg.paths.silver_dir / cfg.silverize.aadt_output_name

    stats_inc = silverize_incidents(
        bronze_dir=cfg.paths.bronze_dir,
        out_path=incidents_path,
        datetime_format=cfg.silverize.datetime_format,
    )
    print(f"[silver] incidents files={stats_inc.input_files} rows={stats_inc.output_rows} -> {incidents_path}")

    stats_w = silverize_weather_hourly(
        bronze_path=cfg.paths.weather_bronze_path,
        out_path=weather_path,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=cfg.features.bucket_minutes,
    )
    print(f"[silver] weather rows={stats_w.output_rows} -> {weather_path}")

    stats_a = silverize_aadt_stations(
        bronze_path=cfg.paths.aadt_bronze_path,
        out_path=aadt_path,
    )
    print(f"[silver] aadt stations={stats_a.unique_stations} -> {aadt_path}")


def _run_gold(*, config_path: str) -> None:
    cfg = load_config(config_path)

    incidents_path = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_path = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    aadt_path = cfg.paths.silver_dir / cfg.silverize.aadt_output_name
    traffic_counts_dir = cfg.paths.bronze_dir / "austin_traffic_counts"
    out_csv = cfg.paths.gold_dir / "features" / "hotspot_features.csv"

    if not incidents_path.exists():
        raise FileNotFoundError(incidents_path)

    stats = build_hotspot_features(
        silver_incidents_csv=incidents_path,
        weather_hourly_csv=weather_path,
        aadt_stations_csv=aadt_path,
        traffic_counts_dir=traffic_counts_dir if traffic_counts_dir.exists() else None,
        out_csv=out_csv,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=cfg.features.bucket_minutes,
        cell_round_decimals=cfg.features.cell_round_decimals,
        lookback_hours=cfg.features.lookback_hours,
        label_horizon_hours=cfg.features.label_horizon_hours,
        aadt_max_distance_km=cfg.features.aadt_max_distance_km,
        ema_half_life_hours=cfg.features.ema_half_life_hours,
    )

    print(
        "[gold] "
        f"out_rows={stats.output_rows} cells={stats.unique_cells} cell_buckets={stats.unique_cell_buckets} -> {out_csv}"
    )


def _run_tok(*, config_path: str) -> None:
    cfg = load_config(config_path)

    incidents_csv = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_csv = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    out_dir = cfg.tokenizer.output_dir

    if not incidents_csv.exists():
        raise FileNotFoundError(incidents_csv)
    if not weather_csv.exists():
        raise FileNotFoundError(weather_csv)

    if not _has_module("h3"):
        _exit_missing_deps(stage="tok", missing="h3")
    if not _has_module("numpy"):
        _exit_missing_deps(stage="tok", missing="numpy")

    from model.tokenizer_h3 import tokenize_h3_time_series

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
        "[tok] "
        f"t={stats.n_time_steps} cells={stats.n_cells} "
        f"coll_events={stats.collisions_events} inc_events={stats.traffic_incidents_events} -> {out_dir}"
    )


def _run_train(*, config_path: str) -> None:
    cfg = load_config(config_path)

    data_dir = cfg.tokenizer.output_dir
    ds_path = data_dir / "dataset.npz"
    meta_path = data_dir / "meta.json"
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    if not _has_module("torch"):
        _exit_missing_deps(stage="train", missing="torch")
    if not _has_module("numpy"):
        _exit_missing_deps(stage="train", missing="numpy")

    from model.train_hotspot import train_hotspot_model

    out_path = Path("artifacts/h3_hotspot_model.pt")
    train_hotspot_model(
        data_dir=data_dir,
        out_path=out_path,
        context_steps=cfg.train.context_steps,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA MVP: bronze -> silver -> gold -> tokenize -> train")
    ap.add_argument("--config", default="configs/pipeline.toml")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--only-silver", action="store_true", help="Run bronze -> silver only")
    g.add_argument("--only-gold", action="store_true", help="Run silver -> gold only")
    g.add_argument("--only-etl", action="store_true", help="Run bronze -> silver -> gold (no tokenize/train)")
    g.add_argument("--only-tok", action="store_true", help="Run tokenizer only")
    g.add_argument("--only-train", action="store_true", help="Run training only")

    args = ap.parse_args()

    required_modules: list[str] = []
    if args.only_tok:
        required_modules = ["h3", "numpy"]
    elif args.only_train:
        required_modules = ["torch", "numpy"]
    elif not (args.only_silver or args.only_gold or args.only_etl):
        required_modules = ["h3", "numpy", "torch"]

    _maybe_reexec_into_venv(required_modules=required_modules)

    if args.only_silver:
        _run_silver(config_path=args.config)
        return
    if args.only_gold:
        _run_gold(config_path=args.config)
        return
    if args.only_etl:
        _run_silver(config_path=args.config)
        _run_gold(config_path=args.config)
        return
    if args.only_tok:
        _run_tok(config_path=args.config)
        return
    if args.only_train:
        _run_train(config_path=args.config)
        return

    _run_silver(config_path=args.config)
    _run_gold(config_path=args.config)
    _run_tok(config_path=args.config)
    _run_train(config_path=args.config)


if __name__ == "__main__":
    main()
