from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _venv_python() -> Path | None:
    root = Path(__file__).resolve().parent
    if sys.platform.startswith("win"):
        cand = root / ".venv" / "Scripts" / "python.exe"
    else:
        cand = root / ".venv" / "bin" / "python"
    return cand if cand.exists() else None


def _reexec_into_venv() -> None:
    if os.environ.get("ETA_MVP_REEXEC") == "1":
        return

    root = Path(__file__).resolve().parent
    venv_dir = root / ".venv"

    venv_py = _venv_python()
    if venv_py is None:
        raise SystemExit(
            "\n".join(
                [
                    "[env] missing .venv; create it and install deps:",
                    "  python3 -m venv .venv",
                    "  .venv/bin/python -m pip install -r requirements.txt",
                ]
            )
        )

    # Avoid comparing resolved executables: venv python is often a symlink to the base interpreter.
    # `sys.prefix` reliably points at the active environment's root directory.
    if Path(sys.prefix).resolve() == venv_dir.resolve():
        return

    env = os.environ.copy()
    env["ETA_MVP_REEXEC"] = "1"
    print(f"[env] re-exec -> {venv_py}", flush=True)
    os.execve(
        str(venv_py),
        [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        os.environ.setdefault(key, value)


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA MVP: fetch forecast + run hotspot inference (venv)")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--model", default="artifacts/h3_hotspot_model.pt")
    ap.add_argument("--forecast-csv", default="data/bronze/austin_forecast_live.csv")
    ap.add_argument("--out", default="output/phase1_output.json")
    ap.add_argument("--horizon-index", type=int, default=0, help="0=first forecast hour, 1=second, etc.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--skip-forecast", action="store_true", help="Use existing forecast CSV (no API call)")
    ap.add_argument("--once", action="store_true", help="Run a single inference cycle and exit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env")

    _reexec_into_venv()

    model_path = (root / args.model).resolve()
    if not model_path.exists():
        raise SystemExit(
            "\n".join(
                [
                    f"[infer] missing model checkpoint: {model_path}",
                    "Train it with:",
                    "  .venv/bin/python eta_mvp_run.py --config configs/pipeline.toml",
                ]
            )
        )

    forecast_path = (root / args.forecast_csv).resolve()

    try:
        loop_min = float(os.environ.get("mvp_infer_loop_min", "5"))
    except ValueError:
        raise SystemExit("[env] mvp_infer_loop_min must be a number (minutes)")
    if loop_min <= 0:
        raise SystemExit("[env] mvp_infer_loop_min must be > 0")

    def run_once() -> None:
        if not args.skip_forecast:
            _run([sys.executable, "scripts/fetch_forecast.py"], cwd=root)

        if not forecast_path.exists():
            raise SystemExit(
                "\n".join(
                    [
                        f"[infer] missing forecast CSV: {forecast_path}",
                        "Create it with:",
                        "  .venv/bin/python scripts/fetch_forecast.py",
                    ]
                )
            )

        _run(
            [
                sys.executable,
                "src/model/infer_hotspot.py",
                "--config",
                args.config,
                "--model",
                str(model_path),
                "--forecast-csv",
                str(forecast_path),
                "--out",
                args.out,
                "--horizon-index",
                str(args.horizon_index),
                "--top-k",
                str(args.top_k),
                "--threshold",
                str(args.threshold),
            ],
            cwd=root,
        )

    if args.once:
        run_once()
        return

    print(f"[loop] interval_min={loop_min:g} (set via .env: mvp_infer_loop_min)", flush=True)
    while True:
        run_once()
        time.sleep(loop_min * 60.0)


if __name__ == "__main__":
    main()
