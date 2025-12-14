# sim/ws_sumo_server.py
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Set

import websockets
import traci
import sumolib


@dataclass
class Vehicle:
    vehicle_id: str
    lat: float
    lon: float
    heading: float
    vtype: str


def serialize_vehicles(vehicles: list[Vehicle]) -> str:
    return json.dumps(
        {
            "t": time.time(),
            "vehicles": [
                {
                    "vehicle-id": v.vehicle_id,
                    "lat": v.lat,
                    "lon": v.lon,
                    "heading": v.heading,
                    "type": v.vtype,
                }
                for v in vehicles
            ],
        }
    )


def make_linear_xy_to_lonlat(net: sumolib.net.Net) -> Callable[[float, float], tuple[float, float]]:
    """
    Map SUMO XY -> lon/lat using convBoundary + origBoundary (good enough for visualization).
    """
    L = net._location
    cx0, cy0, cx1, cy1 = map(float, L["convBoundary"].split(","))
    olon0, olat0, olon1, olat1 = map(float, L["origBoundary"].split(","))

    dx = (cx1 - cx0) if (cx1 - cx0) != 0 else 1.0
    dy = (cy1 - cy0) if (cy1 - cy0) != 0 else 1.0

    def xy_to_lonlat(x: float, y: float) -> tuple[float, float]:
        tx = (x - cx0) / dx
        ty = (y - cy0) / dy
        lon = olon0 + tx * (olon1 - olon0)
        lat = olat0 + ty * (olat1 - olat0)
        return lon, lat

    return xy_to_lonlat


def snapshot_vehicles(xy_to_lonlat: Callable[[float, float], tuple[float, float]]) -> list[Vehicle]:
    vids = traci.vehicle.getIDList()
    out: list[Vehicle] = []

    if out and len(out) < 5:
        print("sample type:", vtype)

    for vid in vids:
        x, y = traci.vehicle.getPosition(vid)
        lon, lat = xy_to_lonlat(x, y)

        # SUMO angle: 0=N, 90=E, 180=S, 270=W
        heading = float(traci.vehicle.getAngle(vid) % 360.0)
        vtype = traci.vehicle.getTypeID(vid)

        out.append(Vehicle(str(vid), float(lat), float(lon), heading, vtype))

    return out


async def broadcaster(
    clients: Set[websockets.WebSocketServerProtocol],
    hz: float,
    xy_to_lonlat: Callable[[float, float], tuple[float, float]],
) -> None:
    interval = 1.0 / hz

    while True:
        traci.simulationStep()
        msg = serialize_vehicles(snapshot_vehicles(xy_to_lonlat))

        if clients:
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        await asyncio.sleep(interval)


async def handler(ws: websockets.WebSocketServerProtocol, clients: Set[websockets.WebSocketServerProtocol]):
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)


async def amain() -> None:
    ap = argparse.ArgumentParser(description="SUMO -> WebSocket vehicle publisher (Deck.gl compatible).")
    ap.add_argument("--net-path", required=True, help="Path to *.net.xml for boundary mapping")
    ap.add_argument("--traci-host", default="localhost")
    ap.add_argument("--traci-port", type=int, default=8813)
    ap.add_argument("--ws-host", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    net_path = Path(args.net_path).expanduser().resolve()
    if not net_path.exists():
        raise SystemExit(f"net-path not found: {net_path}")

    net = sumolib.net.readNet(str(net_path))
    xy_to_lonlat = make_linear_xy_to_lonlat(net)

    # Connect to SUMO started with: --remote-port <traci-port>
    traci.init(port=int(args.traci_port), host=str(args.traci_host))

    clients: Set[websockets.WebSocketServerProtocol] = set()

    async with websockets.serve(
        lambda ws: handler(ws, clients),
        str(args.ws_host),
        int(args.ws_port),
        ping_interval=20,
        ping_timeout=20,
    ):
        print(f"[ws_sumo_server] ws://{args.ws_host}:{args.ws_port} (clients connect here)", flush=True)
        print(f"[ws_sumo_server] traci: {args.traci_host}:{args.traci_port}", flush=True)
        await broadcaster(clients, float(args.hz), xy_to_lonlat)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
