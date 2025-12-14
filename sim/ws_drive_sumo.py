from __future__ import annotations

import argparse
import asyncio
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

    cfg = load_controls(Path(args.controls))
    controls = cfg.controls
    interval_s = float(cfg.control_interval_s)

    conn = traci.connect(host=str(args.sumo_host), port=int(args.sumo_port))
    _ensure_routes(conn, controls)

    carry: dict[str, float] = {c.control_id: 0.0 for c in controls}
    veh_seq = 0

    print(f"[ws_drive_sumo] ws={args.ws_url} traci={args.sumo_host}:{args.sumo_port} controls={len(controls)}", flush=True)
    print("[ws_drive_sumo] waiting for inference messages...", flush=True)

    n_msgs = 0
    ws = None
    for _ in range(max(1, int(args.ws_retries))):
        try:
            ws = await websockets.connect(str(args.ws_url))
            break
        except Exception:
            await asyncio.sleep(1)
            continue
    if ws is None:
        raise RuntimeError(f"Could not connect to WebSocket at {args.ws_url}")

    async with ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            mult_by_id = _parse_multipliers(msg, controls)

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

            if not bool(args.no_step):
                now = float(conn.simulation.getTime())
                conn.simulationStep(now + interval_s)

            n_msgs += 1
            n_msgs = int(n_msgs)
            if int(args.max_msgs) > 0 and n_msgs >= int(args.max_msgs):
                break


if __name__ == "__main__":
    asyncio.run(main())
