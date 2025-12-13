from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim.common import how_sin_cos, load_controls
from sim.engine_mock import MockSimEngine
from sim.model_gru import load_checkpoint


def _pick_device(s: str | None) -> torch.device:
    if not s or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Digital twin server: Torch inference drives SUMO (or mock), broadcasts WS JSON.")
    ap.add_argument("--engine", choices=["mock", "sumo"], default="mock")
    ap.add_argument("--controls", default="sim/controls_example.json")
    ap.add_argument("--model", required=True, help="Path to Torch checkpoint (from sim/train_demand_gru.py)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--ws-host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--realtime", action="store_true", help="Sleep to approximate real-time control intervals")
    ap.add_argument("--max-intervals", type=int, default=0, help="0=run forever (useful for local smoke tests)")
    ap.add_argument("--close-edge", action="append", default=[], help="Edge ID(s) to close immediately (SUMO engine only)")
    ap.add_argument("--close-speed-mps", type=float, default=0.1)
    ap.add_argument("--sumo-cfg", default=None, help="Required for --engine sumo")
    ap.add_argument("--sumo-binary", default="sumo")
    ap.add_argument("--sumo-step-s", type=float, default=1.0)
    args = ap.parse_args()

    controls_cfg = load_controls(Path(args.controls))
    controls = controls_cfg.controls
    control_interval_s = float(controls_cfg.control_interval_s)
    context_steps = int(controls_cfg.context_steps)

    device = _pick_device(args.device)
    model = load_checkpoint(Path(args.model), map_location=str(device)).to(device)
    if model.cfg.n_controls != len(controls):
        raise ValueError(f"Model n_controls={model.cfg.n_controls} but controls={len(controls)} (check your controls JSON).")
    if model.cfg.context_steps != context_steps:
        raise ValueError(
            f"Model context_steps={model.cfg.context_steps} but controls config context_steps={context_steps}."
        )

    if args.engine == "mock":
        engine: Any = MockSimEngine(controls=controls, control_interval_s=control_interval_s)
    else:
        if not args.sumo_cfg:
            raise ValueError("--sumo-cfg is required when --engine sumo")
        from sim.engine_sumo import SumoSimEngine

        engine = SumoSimEngine(
            sumo_cfg=Path(args.sumo_cfg),
            controls=controls,
            control_interval_s=control_interval_s,
            sumo_binary=str(args.sumo_binary),
            step_length_s=float(args.sumo_step_s),
            close_speed_mps=float(args.close_speed_mps),
        )
        if args.close_edge:
            engine.close_edges(list(args.close_edge))

    # Rolling context (flows ratios) for model input.
    buf = [[1.0 for _ in controls] for _ in range(context_steps)]
    sim_dt = datetime.now().replace(minute=0, second=0, microsecond=0)

    # --- WebSocket broadcast server ---
    try:
        import websockets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: `websockets` (install sim/requirements.txt).") from e

    clients: set[Any] = set()
    last_payload: dict[str, Any] | None = None

    async def ws_handler(ws):
        clients.add(ws)
        if last_payload is not None:
            try:
                await ws.send(json.dumps(last_payload))
            except Exception:
                pass
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    async def broadcast(payload: dict[str, Any]) -> None:
        nonlocal last_payload
        last_payload = payload
        if not clients:
            return
        msg = json.dumps(payload)
        dead: list[Any] = []
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    try:
        async with websockets.serve(ws_handler, args.ws_host, int(args.ws_port), ping_interval=20, ping_timeout=20):
            print(f"[sim] ws://{args.ws_host}:{args.ws_port} engine={args.engine} device={device}", flush=True)

            intervals = 0
            while True:
                if int(args.max_intervals) > 0 and intervals >= int(args.max_intervals):
                    break

                # Build model input: (1, context_steps, n_controls + 2)
                exog = []
                for t in range(context_steps):
                    s, c = how_sin_cos(sim_dt - timedelta(seconds=control_interval_s * (context_steps - 1 - t)))
                    exog.append([s, c])
                x_rows = [buf[t] + exog[t] for t in range(context_steps)]
                x = torch.tensor([x_rows], dtype=torch.float32, device=device)

                with torch.no_grad():
                    mult = model.predict_multipliers(x).squeeze(0).detach().cpu().tolist()

                engine.apply_multipliers(mult)
                engine.step_interval()
                obs = engine.observe_flows_ratio()

                # Update rolling buffer
                buf = buf[1:] + [obs]

                edges = [
                    {"edge_id": s.edge_id, "veh_count": s.veh_count, "mean_speed_mps": s.mean_speed_mps}
                    for s in engine.edge_states()
                ]
                payload = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "sim_time_s": float(getattr(engine, "sim_time_s", 0.0)),
                    "controls": [
                        {
                            "control_id": c.control_id,
                            "edge_id": c.edge_id,
                            "baseline_veh_per_hour": float(c.baseline_veh_per_hour),
                            "multiplier": float(m),
                        }
                        for c, m in zip(controls, mult, strict=True)
                    ],
                    "edges": edges,
                }
                await broadcast(payload)

                intervals += 1
                sim_dt = sim_dt + timedelta(seconds=control_interval_s)

                if args.realtime:
                    await asyncio.sleep(control_interval_s)
                else:
                    await asyncio.sleep(0)
    finally:
        if hasattr(engine, "close"):
            engine.close()


if __name__ == "__main__":
    asyncio.run(main())
