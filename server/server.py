import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, asdict

import websockets


@dataclass
class Vehicle:
    # JSON wants "vehicle-id" (kebab case), so we’ll map at serialization time.
    vehicle_id: int
    lat: float
    lon: float
    heading: float  # degrees 0..359


def serialize_vehicles(vehicles: list[Vehicle]) -> str:
    payload = {
        "t": time.time(),
        "vehicles": [
            {
                "vehicle-id": v.vehicle_id,
                "lat": v.lat,
                "lon": v.lon,
                "heading": v.heading,
            }
            for v in vehicles
        ],
    }
    return json.dumps(payload)


def advance_vehicle(v: Vehicle, dt: float) -> None:
    """
    Update position + heading with a tiny “wandering” motion.
    dt: seconds since last update
    """
    # Randomly wiggle heading a bit
    v.heading = (v.heading + random.uniform(-10, 10)) % 360

    # Move forward based on heading
    speed_mps = random.uniform(2.0, 8.0)  # ~7–28 km/h
    distance_m = speed_mps * dt

    # Convert meters to degrees (rough approximation; good enough for demos)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(v.lat))

    dlat = (distance_m * math.cos(math.radians(v.heading))) / meters_per_deg_lat
    dlon = (distance_m * math.sin(math.radians(v.heading))) / max(meters_per_deg_lon, 1e-6)

    v.lat += dlat
    v.lon += dlon


async def stream_loop(websocket, vehicles: list[Vehicle], hz: float = 5.0) -> None:
    """
    Continuously push vehicle states to the client.
    """
    interval = 1.0 / hz
    last = time.perf_counter()

    while True:
        now = time.perf_counter()
        dt = now - last
        last = now

        for v in vehicles:
            advance_vehicle(v, dt)

        await websocket.send(serialize_vehicles(vehicles))
        await asyncio.sleep(interval)


async def handler(websocket):
    # Initial demo vehicles near Austin
    vehicles = [
        Vehicle(vehicle_id=1, lat=30.2050, lon=-97.7664, heading=180),
        Vehicle(vehicle_id=2, lat=30.2100, lon=-97.7700, heading=90),
        Vehicle(vehicle_id=3, lat=30.2150, lon=-97.7750, heading=270),
    ]

    try:
        await stream_loop(websocket, vehicles, hz=10.0)  # 10 updates/sec
    except websockets.ConnectionClosed:
        pass


async def main():
    host = "0.0.0.0"
    port = 8765
    async with websockets.serve(handler, host, port, ping_interval=20, ping_timeout=20):
        print(f"WebSocket server running on ws://{host}:{port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
