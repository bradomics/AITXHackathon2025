from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_config


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


_TARGET_BUCKET_RE = re.compile(r"^//\s*target_bucket=(.+)\s*$")


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _preview_phase1_output(path: Path, *, n: int) -> str:
    if n <= 0:
        return ""
    if not path.exists():
        return f"[preview] missing {path}"

    target_bucket = ""
    coll_lines: list[str] = []
    inc_lines: list[str] = []
    section: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not target_bucket:
            m = _TARGET_BUCKET_RE.match(line)
            if m:
                target_bucket = m.group(1).strip()

        if line.startswith("const COLLISION_POINTS"):
            section = "collision"
            continue
        if line.startswith("const INCIDENT_POINTS"):
            section = "incident"
            continue
        if line.startswith("];"):
            section = None
            continue

        if section == "collision" and len(coll_lines) < n and line.startswith("{ position:"):
            coll_lines.append(line)
        elif section == "incident" and len(inc_lines) < n and line.startswith("{ position:"):
            inc_lines.append(line)

        if len(coll_lines) >= n and len(inc_lines) >= n:
            break

    out: list[str] = []
    bucket_s = f" target_bucket={target_bucket}" if target_bucket else ""
    out.append(f"[preview]{bucket_s} {path}")
    if coll_lines:
        out.append(f"[preview] COLLISION_POINTS (first {len(coll_lines)}):")
        out.extend(["  " + s for s in coll_lines])
    if inc_lines:
        out.append(f"[preview] INCIDENT_POINTS (first {len(inc_lines)}):")
        out.extend(["  " + s for s in inc_lines])
    return "\n".join(out)


def _preview_safety_output(path: Path, *, n: int) -> str:
    if n <= 0:
        return ""
    if not path.exists():
        return f"[preview] missing {path}"

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"[preview] invalid json {path}"

    target_bucket = str(obj.get("target_bucket") or "")
    assets = obj.get("assets") or {}
    type1 = assets.get("type1_collisions") or []
    type2 = assets.get("type2_incidents") or []

    def fmt_asset(a: dict[str, object]) -> str:
        return f"{a.get('asset_id')} lat={a.get('lat')} lon={a.get('lon')} expected_hit={a.get('expected_hit')}"

    out: list[str] = []
    bucket_s = f" target_bucket={target_bucket}" if target_bucket else ""
    out.append(f"[preview]{bucket_s} {path}")
    out.append(f"[preview] type1_collisions assets={len(type1)} (first {min(n, len(type1))}):")
    for a in type1[:n]:
        if isinstance(a, dict):
            out.append("  " + fmt_asset(a))
    out.append(f"[preview] type2_incidents assets={len(type2)} (first {min(n, len(type2))}):")
    for a in type2[:n]:
        if isinstance(a, dict):
            out.append("  " + fmt_asset(a))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA MVP: fetch forecast + run hotspot inference (v2 default; venv)")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--variant", choices=["v1", "v2"], default="v2")
    ap.add_argument("--model", default=None, help="Defaults to the checkpoint for --variant")
    ap.add_argument("--forecast-csv", default="data/bronze/austin_forecast_live.csv")
    ap.add_argument("--out", default="output/phase1_output.json")
    ap.add_argument("--horizon-index", type=int, default=0, help="0=first forecast hour, 1=second, etc.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--skip-forecast", action="store_true", help="Use existing forecast CSV (no API call)")
    ap.add_argument("--skip-live-incidents", action="store_true", help="(v2) Skip Socrata live-incident fetch")
    ap.add_argument("--preview-lines", type=int, default=8)
    ap.add_argument("--once", action="store_true", help="Run a single inference cycle and exit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env.local")
    _load_dotenv(root / ".env")

    _reexec_into_venv()

    cfg = load_config(args.config)

    default_model = "artifacts/h3_hotspot_model.pt" if args.variant == "v1" else "artifacts/h3_hotspot_v2_model.pt"
    model_arg = str(args.model) if args.model else default_model

    model_path = (root / model_arg).resolve()
    if not model_path.exists():
        train_cmd = (
            ".venv/bin/python eta_mvp_run.py --config configs/pipeline.toml --hotspot v1"
            if args.variant == "v1"
            else ".venv/bin/python eta_mvp_run.py --config configs/pipeline.toml"
        )
        raise SystemExit(
            "\n".join(
                [
                    f"[infer] missing model checkpoint: {model_path}",
                    "Train it with:",
                    f"  {train_cmd}",
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

        if args.variant == "v2" and not args.skip_live_incidents:
            runtime_dir = cfg.hotspot_v2.runtime_dir
            out_ndjson = runtime_dir / "incidents.ndjson"
            out_state = runtime_dir / "incidents_fetch_state.json"
            _run(
                [
                    sys.executable,
                    "scripts/fetch_live_incidents.py",
                    "--out",
                    str(out_ndjson),
                    "--state",
                    str(out_state),
                    "--seed-hours",
                    str(int(cfg.hotspot_v2.seed_hours)),
                    "--datetime-format",
                    str(cfg.silverize.datetime_format),
                ],
                cwd=root,
            )

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

        infer_script = "src/model/infer_hotspot.py" if args.variant == "v1" else "src/model/infer_hotspot_v2.py"
        infer_cmd = [
            sys.executable,
            infer_script,
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
        ]
        if args.variant == "v2":
            infer_cmd.extend(["--runtime-dir", str(cfg.hotspot_v2.runtime_dir)])

        _run(
            infer_cmd,
            cwd=root,
        )

        out_path = _resolve_path(root, str(args.out))
        preview = _preview_phase1_output(out_path, n=int(args.preview_lines))
        if preview:
            print(preview, flush=True)

        safety_out = out_path.with_name("phase1_safety_output.json")
        _run(
            [
                sys.executable,
                "scripts/plan_safety_assets.py",
                "--in",
                str(out_path),
                "--out",
                str(safety_out),
            ],
            cwd=root,
        )
        safety_preview = _preview_safety_output(safety_out, n=int(args.preview_lines))
        if safety_preview:
            print(safety_preview, flush=True)

    if args.once:
        run_once()
        return

    print(f"[loop] interval_min={loop_min:g} (set via .env: mvp_infer_loop_min)", flush=True)
    while True:
        run_once()
        time.sleep(loop_min * 60.0)


if __name__ == "__main__":
    main()
