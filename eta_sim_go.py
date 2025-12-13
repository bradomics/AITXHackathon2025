from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA SIM: run the digital twin server (Torch inference drives the sim).")
    ap.add_argument("--engine", choices=["mock", "sumo"], default="mock")
    ap.add_argument("--controls", default="sim/controls_counts_example_filled.json")
    ap.add_argument("--model", default="sim/artifacts/demand_gru_counts.pt")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--ws-host", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--max-intervals", type=int, default=0, help="0=run forever")
    ap.add_argument("--sumo-cfg", default=None, help="Required for --engine sumo")
    ap.add_argument("--sumo-binary", default="sumo")
    ap.add_argument("--sumo-step-s", type=float, default=1.0)
    ap.add_argument("--close-edge", action="append", default=[])
    ap.add_argument("--close-speed-mps", type=float, default=0.1)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [
        python,
        str(repo_root / "sim" / "digital_twin_server.py"),
        "--engine",
        args.engine,
        "--controls",
        str(repo_root / args.controls),
        "--model",
        str(repo_root / args.model),
        "--device",
        str(args.device),
        "--ws-host",
        str(args.ws_host),
        "--ws-port",
        str(int(args.ws_port)),
        "--close-speed-mps",
        str(float(args.close_speed_mps)),
    ]
    if args.realtime:
        cmd.append("--realtime")
    if int(args.max_intervals) > 0:
        cmd.extend(["--max-intervals", str(int(args.max_intervals))])
    for e in args.close_edge:
        cmd.extend(["--close-edge", str(e)])

    if args.engine == "sumo":
        if not args.sumo_cfg:
            raise SystemExit("--sumo-cfg is required when --engine sumo")
        cmd.extend(
            [
                "--sumo-cfg",
                str(args.sumo_cfg),
                "--sumo-binary",
                str(args.sumo_binary),
                "--sumo-step-s",
                str(float(args.sumo_step_s)),
            ]
        )

    print("[eta_sim_go] starting (Ctrl+C to stop)", flush=True)
    print("[eta_sim_go] " + " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, cwd=str(repo_root))
    try:
        raise SystemExit(proc.wait())
    except KeyboardInterrupt:
        print("\n[eta_sim_go] stopping...", flush=True)
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
