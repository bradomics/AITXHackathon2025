from __future__ import annotations

import argparse
import errno
import getpass
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path


def _pick_free_port(host: str, preferred: int) -> int:
    if preferred <= 0:
        raise ValueError("preferred port must be > 0")
    for port in range(preferred, preferred + 50):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                continue
            raise
        finally:
            try:
                s.close()
            except Exception:
                pass
        return port
    raise SystemExit(f"No free port found starting at {preferred} for host={host!r}.")


def main() -> None:
    ap = argparse.ArgumentParser(description="ETA SIM: run the digital twin stack.")
    ap.add_argument(
        "--mode",
        choices=["ws-sumo", "inference"],
        default="ws-sumo",
        help="ws-sumo = SUMO->TraCI->WebSocket publisher; inference = Torch-driven digital_twin_server",
    )

    # --- WS publisher args (both modes use ws-host/ws-port) ---
    ap.add_argument("--ws-host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=8765)

    # --- ws-sumo mode args ---
    ap.add_argument("--net-path", default="sumo/austin/austin.net.xml")
    ap.add_argument("--traci-host", default="localhost")
    ap.add_argument("--traci-port", type=int, default=8813)
    ap.add_argument("--hz", type=float, default=10.0)

    # --- inference mode args ---
    ap.add_argument("--engine", choices=["mock", "sumo"], default="sumo")
    ap.add_argument("--controls", default="sim/controls_austin_radar_auto_filled.json")
    ap.add_argument("--model", default="sim/artifacts/demand_gru_counts_austin.pt")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--max-intervals", type=int, default=0, help="0=run forever")
    ap.add_argument("--sumo-cfg", default="sumo/austin/twin.sumocfg", help="Required for --engine sumo")
    ap.add_argument("--sumo-binary", default="sumo")
    ap.add_argument("--sumo-step-s", type=float, default=0.1)
    ap.add_argument("--close-edge", action="append", default=[])
    ap.add_argument("--close-speed-mps", type=float, default=0.1)

    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent

    # Pick Python
    venv_python = repo_root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    # Avoid a common footgun: stale publisher still holding the default port.
    ws_host = str(args.ws_host)
    requested_ws_port = int(args.ws_port)
    ws_port = requested_ws_port
    if requested_ws_port == 8765 and "--ws-port" not in sys.argv:
        ws_port = _pick_free_port(ws_host, requested_ws_port)
        args.ws_port = ws_port

    ws_url = f"ws://{ws_host}:{ws_port}"
    if ws_port != requested_ws_port:
        print(f"[eta_sim_go] ws-port {requested_ws_port} in use; using {ws_port}", flush=True)

    # Helpful tunnel hint when binding local-only
    if ws_host in {"127.0.0.1", "localhost"}:
        user = getpass.getuser()
        host = socket.gethostname()
        ssh_host = f"{host}.local" if "." not in host else host
        ssh_cmd = f"ssh -N -L {ws_port}:localhost:{ws_port} {user}@{ssh_host}"
        print(f"[eta_sim_go] ws publisher: {ws_url} (local-only)", flush=True)
        print(f"[eta_sim_go] remote access: {ssh_cmd}", flush=True)
        print(f"[eta_sim_go] then connect: ws://localhost:{ws_port}", flush=True)
    else:
        print(f"[eta_sim_go] ws publisher: {ws_url}", flush=True)

    # Build command per mode
    if args.mode == "ws-sumo":
        net_path = (repo_root / str(args.net_path)).resolve()
        if not net_path.exists():
            raise SystemExit(f"net-path not found: {net_path}")

        cmd = [
            python,
            str(repo_root / "sim" / "ws_sumo_server.py"),
            "--net-path",
            str(net_path),
            "--traci-host",
            str(args.traci_host),
            "--traci-port",
            str(int(args.traci_port)),
            "--ws-host",
            str(args.ws_host),
            "--ws-port",
            str(int(args.ws_port)),
            "--hz",
            str(float(args.hz)),
        ]

        print("[eta_sim_go] SUMO must already be running with TraCI enabled, e.g.:", flush=True)
        print("  sumo -c sumo/austin/twin.sumocfg --remote-port 8813 --start", flush=True)

    else:
        # Only do SUMO binary resolution/checks for inference mode (when we might launch/need it)
        if args.engine == "sumo":
            requested = str(args.sumo_binary)
            if requested in {"sumo", "sumo-gui"}:
                sumo_home = os.environ.get("SUMO_HOME")
                if sumo_home:
                    cand = Path(sumo_home) / "bin" / requested
                    if cand.exists():
                        args.sumo_binary = str(cand)
                if str(args.sumo_binary) == requested:
                    if sys.platform.startswith("win"):
                        cand = repo_root / ".venv" / "Scripts" / f"{requested}.exe"
                    else:
                        cand = repo_root / ".venv" / "bin" / requested
                    if cand.exists():
                        args.sumo_binary = str(cand)
                if str(args.sumo_binary) == requested:
                    cand = repo_root / ".cache" / "sumo" / "EclipseSUMO-1.25.0" / "bin" / requested
                    if cand.exists():
                        args.sumo_binary = str(cand)

            if shutil.which(str(args.sumo_binary)) is None:
                raise SystemExit(
                    "SUMO binary not found (install SUMO, set SUMO_HOME, or pass --sumo-binary /path/to/sumo)."
                )

        controls_path = repo_root / str(args.controls)
        if not controls_path.exists():
            raise SystemExit(f"Controls file not found: {controls_path}")

        model_path = repo_root / str(args.model)
        if not model_path.exists():
            raise SystemExit(
                "\n".join(
                    [
                        f"Model checkpoint not found: {model_path}",
                        "Train one with `eta_sim_run.py` (see `sim/README.md`) or pass --model /path/to/checkpoint.pt.",
                    ]
                )
            )

        cmd = [
            python,
            str(repo_root / "sim" / "digital_twin_server.py"),
            "--engine",
            args.engine,
            "--controls",
            str(controls_path),
            "--model",
            str(model_path),
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
