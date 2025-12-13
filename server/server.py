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
    Your net has projParameter="-" so sumolib can't do real projection math.
    But it DOES include:
      - convBoundary (network XY extents)
      - origBoundary (lon/lat extents)
    So we map XY -> lon/lat linearly. Good enough for visualization.
    """
    L = net._location  # contains strings for convBoundary/origBoundary/projParameter

    cx0, cy0, cx1, cy1 = map(float, L["convBoundary"].split(","))
    olon0, olat0, olon1, olat1 = map(float, L["origBoundary"].split(","))

    # Precompute denominators to avoid divide-by-zero checks every call
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
        # Advance SUMO one step
        traci.simulationStep()

        # Snapshot + serialize once
        msg = serialize_vehicles(snapshot_vehicles(xy_to_lonlat))

        # Broadcast
        if clients:
            dead = []
            for ws in clients:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        await asyncio.sleep(interval)


async def handler(websocket: websockets.WebSocketServerProtocol, clients: Set[websockets.WebSocketServerProtocol]):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def main():
    # ---- CONFIG ----
    TRACI_HOST = "localhost"
    TRACI_PORT = 8813
    WS_HOST = "0.0.0.0"
    WS_PORT = 8765
    HZ = 10.0

    NET_PATH = Path("/Users/brad/sumo_osm/austin/austin.net.xml").resolve()

    # Load net (needed only for the linear mapping boundaries)
    net = sumolib.net.readNet(NET_PATH.as_uri())
    xy_to_lonlat = make_linear_xy_to_lonlat(net)

    # Connect to already-running SUMO that was started with: --remote-port 8813
    # Use init() because we call global traci.* methods.
    traci.init(port=TRACI_PORT, host=TRACI_HOST)

    clients: Set[websockets.WebSocketServerProtocol] = set()

    async with websockets.serve(
        lambda ws: handler(ws, clients),
        WS_HOST,
        WS_PORT,
        ping_interval=20,
        ping_timeout=20,
    ):
        print(f"WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
        await broadcaster(clients, HZ, xy_to_lonlat)


if __name__ == "__main__":
    asyncio.run(main())
