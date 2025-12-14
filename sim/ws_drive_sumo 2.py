from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim.common import load_controls


def _ensure_routes(conn, controls) -> None:
    for c in controls:
        if not c.route_edges:
            continue
        rid = f"route_{c.control_id}"
        try:
            conn.route.add(rid, c.route_edges)
        except Exception:
            # If it already exists, that's fine.
            pass


def _parse_multipliers(msg: dict, controls) -> dict[str, float]:
    mult_by_id: dict[str, float] = {}

    if isinstance(msg.get("multipliers"), dict):
        for k, v in msg["multipliers"].items():
            try:
                mult_by_id[str(k)] = float(v)
            except Exception:
                continue

    if not mult_by_id and isinstance(msg.get("controls"), list):
        for c in msg["controls"]:
            if not isinstance(c, dict):
                continue
            cid = c.get("control_id")
            if cid is None:
                continue
            try:
                mult_by_id[str(cid)] = float(c.get("multiplier", 1.0))
            except Exception:
                continue

    # Default missing controls to 1.0
    for c in controls:
        mult_by_id.setdefault(c.control_id, 1.0)
    return mult_by_id


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bridge: subscribe to the digital_twin_server WebSocket and drive a running SUMO via TraCI."
    )
    ap.add_argument("--ws-url", default="ws://localhost:8765")
    ap.add_argument("--controls", default="sim/controls_austin_radar_auto_filled.json")
    ap.add_argument("--sumo-host", default="localhost")
    ap.add_argument("--sumo-port", type=int, default=8813)
    ap.add_argument("--no-step", action="store_true", help="Only inject vehicles; do not advance the simulation.")
    ap.add_argument("--tick-s", type=float, default=1.0, help="SUMO step size for the twin (only when not using --no-step).")
    ap.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sleep to approximate real-time (only when not using --no-step).",
    )
    ap.add_argument("--max-msgs", type=int, default=0, help="0=run forever")
    ap.add_argument("--ws-retries", type=int, default=60, help="How many times to retry the WS connection.")
    args = ap.parse_args()

    try:
        import websockets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: `websockets` (install sim/requirements.txt).") from e

    try:
        import traci
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: `traci` (install sim/requirements.txt).") from e

    repo_root = Path(__file__).resolve().parents[1]
    controls_path = Path(args.controls)
    if not controls_path.is_absolute() and not controls_path.exists():
        controls_path = repo_root / controls_path
    if not controls_path.exists():
        raise SystemExit(f"Controls file not found: {controls_path}")

    cfg = load_controls(controls_path)
    controls = cfg.controls
    interval_s = float(cfg.control_interval_s)

    sumo_host = str(args.sumo_host)
    sumo_port = int(args.sumo_port)
    print(f"[ws_drive_sumo] connecting TraCI: {sumo_host}:{sumo_port}", flush=True)
    try:
        connect_kwargs = {"host": sumo_host, "port": sumo_port}
        if "numRetries" in inspect.signature(traci.connect).parameters:
            connect_kwargs["numRetries"] = 1
        conn = traci.connect(**connect_kwargs)
    except Exception as e:
        raise SystemExit(
            "\n".join(
                [
                    f"Could not connect to TraCI at {sumo_host}:{sumo_port}.",
                    "Start SUMO GUI with TraCI enabled, for example:",
                    f"  sumo-gui -c sumo/austin/sim.sumocfg --remote-port {sumo_port} --start",
                    f"Then re-run: python3 sim/ws_drive_sumo.py --ws-url {args.ws_url} --sumo-port {sumo_port}",
                    f"Error: {e}",
                ]
            )
        ) from e
    _ensure_routes(conn, controls)

    print(f"[ws_drive_sumo] ws={args.ws_url} traci={args.sumo_host}:{args.sumo_port} controls={len(controls)}", flush=True)
    print("[ws_drive_sumo] connecting to WS...", flush=True)

    n_msgs = 0
    ws = None
    max_tries = max(1, int(args.ws_retries))
    for attempt in range(1, max_tries + 1):
        try:
            ws = await websockets.connect(str(args.ws_url))
            break
        except Exception as e:
            if attempt == max_tries:
                raise RuntimeError(f"Could not connect to WebSocket at {args.ws_url}: {e}") from e
            if attempt == 1 or attempt % 5 == 0:
                print(f"[ws_drive_sumo] WS not ready (attempt {attempt}/{max_tries}); retrying...", flush=True)
            await asyncio.sleep(1)
            continue
    if ws is None:
        raise RuntimeError(f"Could not connect to WebSocket at {args.ws_url}")
    print("[ws_drive_sumo] connected; starting twin (uses multiplier=1.0 until the first message arrives)...", flush=True)

    carry: dict[str, float] = {c.control_id: 0.0 for c in controls}
    veh_seq = 0
    mult_by_id: dict[str, float] = {c.control_id: 1.0 for c in controls}
    stop = asyncio.Event()

    async def recv_loop() -> None:
        nonlocal mult_by_id, n_msgs, veh_seq
        async with ws:
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                mult_by_id = _parse_multipliers(msg, controls)

                n_msgs += 1
                if n_msgs == 1 or n_msgs % 10 == 0:
                    ts = msg.get("ts")
                    ts_s = f" ts={ts}" if isinstance(ts, str) else ""
                    print(f"[ws_drive_sumo] msg={n_msgs}{ts_s}", flush=True)

                if bool(args.no_step):
                    for c in controls:
                        if not c.route_edges:
                            continue

                        base = max(0.0, float(c.baseline_veh_per_hour))
                        m = float(mult_by_id.get(c.control_id, 1.0))
                        expected = base * m * (interval_s / 3600.0) + carry[c.control_id]
                        n_add = int(expected)
                        carry[c.control_id] = expected - n_add

                        rid = f"route_{c.control_id}"
                        for _ in range(n_add):
                            vid = f"twin_{veh_seq}"
                            veh_seq += 1
                            try:
                                conn.vehicle.add(vid, rid, depart="now")
                            except Exception:
                                continue

                if int(args.max_msgs) > 0 and n_msgs >= int(args.max_msgs):
                    stop.set()
                    return

    recv_task = asyncio.create_task(recv_loop())

    if bool(args.no_step):
        print("[ws_drive_sumo] --no-step: injecting on each message; SUMO runs/steps independently.", flush=True)
        try:
            await recv_task
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return

    tick_s = float(args.tick_s)
    if tick_s <= 0:
        raise SystemExit("--tick-s must be > 0")

    print(f"[ws_drive_sumo] stepping SUMO in {tick_s:.3f}s ticks (--no-realtime to run as fast as possible)", flush=True)
    try:
        while not stop.is_set():
            if recv_task.done():
                exc = recv_task.exception()
                if exc is not None:
                    raise exc
                break

            # Inject vehicles for this tick using baseline_veh_per_hour * multiplier.
            for c in controls:
                if not c.route_edges:
                    continue

                base = max(0.0, float(c.baseline_veh_per_hour))
                m = float(mult_by_id.get(c.control_id, 1.0))
                expected = base * m * (tick_s / 3600.0) + carry[c.control_id]
                n_add = int(expected)
                carry[c.control_id] = expected - n_add

                rid = f"route_{c.control_id}"
                for _ in range(n_add):
                    vid = f"twin_{veh_seq}"
                    veh_seq += 1
                    try:
                        conn.vehicle.add(vid, rid, depart="now")
                    except Exception:
                        continue

            now = float(conn.simulation.getTime())
            conn.simulationStep(now + tick_s)

            if bool(args.realtime):
                await asyncio.sleep(tick_s)
            else:
                await asyncio.sleep(0)
    finally:
        if not recv_task.done():
            recv_task.cancel()
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
