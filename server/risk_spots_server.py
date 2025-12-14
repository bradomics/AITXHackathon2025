#!/usr/bin/env python3
"""
API server for serving collision and incident risk spots.
Reads from phase1_output.json and serves via HTTP endpoints.
"""
import asyncio
import json
import os
import re
from pathlib import Path
from aiohttp import web
import aiohttp_cors

# Default paths to look for the JSON files
OUTPUT_JSON_PATH = Path("output/phase1_output.json")
SAFETY_JSON_PATH = Path("output/phase1_safety_output.json")


def find_json_file(path: Path) -> Path | None:
    """Find a JSON file at the given path."""
    if path.exists():
        return path.resolve()
    # Try relative to server directory
    server_path = Path(__file__).parent / path
    if server_path.exists():
        return server_path.resolve()
    # Try from project root
    project_path = Path(__file__).parent.parent / path
    if project_path.exists():
        return project_path.resolve()
    return None


def load_output_json(file_path: Path) -> dict:
    """
    Load JSON from TypeScript-like file (handles comments and const declarations).
    Extracts COLLISION_POINTS and INCIDENT_POINTS arrays from phase1_output.json.
    """
    content = file_path.read_text()
    
    collision_points = []
    incident_points = []
    
    # Extract COLLISION_POINTS array
    collision_match = re.search(
        r'const\s+COLLISION_POINTS[^=]*=\s*\[(.*?)\];',
        content,
        re.DOTALL
    )
    
    if collision_match:
        # Parse the array content
        array_content = collision_match.group(1)
        # Extract objects with position and weight
        for match in re.finditer(
            r'\{[^}]*position:\s*\[([-\d.]+),\s*([-\d.]+)\][^}]*weight:\s*([\d.]+)[^}]*\}',
            array_content
        ):
            lon, lat, weight = match.groups()
            collision_points.append({
                "position": [float(lon), float(lat)],
                "weight": float(weight)
            })
    
    # Extract INCIDENT_POINTS array
    incident_match = re.search(
        r'const\s+INCIDENT_POINTS[^=]*=\s*\[(.*?)\];',
        content,
        re.DOTALL
    )
    
    if incident_match:
        # Parse the array content
        array_content = incident_match.group(1)
        # Extract objects with position and weight
        for match in re.finditer(
            r'\{[^}]*position:\s*\[([-\d.]+),\s*([-\d.]+)\][^}]*weight:\s*([\d.]+)[^}]*\}',
            array_content
        ):
            lon, lat, weight = match.groups()
            incident_points.append({
                "position": [float(lon), float(lat)],
                "weight": float(weight)
            })
    
    return {
        "collision_points": collision_points,
        "incident_points": incident_points
    }


def load_safety_json(file_path: Path) -> dict:
    """
    Load JSON from phase1_safety_output.json.
    Extracts collision and incident points from assets.covers arrays.
    """
    content = file_path.read_text()
    data = json.loads(content)
    
    collision_points = []
    incident_points = []
    
    # Extract collision points from type1_collisions assets
    if "assets" in data and "type1_collisions" in data["assets"]:
        for asset in data["assets"]["type1_collisions"]:
            if "covers" in asset:
                for cover in asset["covers"]:
                    if "lon" in cover and "lat" in cover and "weight" in cover:
                        collision_points.append({
                            "position": [float(cover["lon"]), float(cover["lat"])],
                            "weight": float(cover["weight"])
                        })
    
    # Extract incident points from type2_incidents assets
    if "assets" in data and "type2_incidents" in data["assets"]:
        for asset in data["assets"]["type2_incidents"]:
            if "covers" in asset:
                for cover in asset["covers"]:
                    if "lon" in cover and "lat" in cover and "weight" in cover:
                        incident_points.append({
                            "position": [float(cover["lon"]), float(cover["lat"])],
                            "weight": float(cover["weight"])
                        })
    
    return {
        "collision_points": collision_points,
        "incident_points": incident_points
    }


def load_collision_spots() -> list:
    """
    Load collision risk spots from phase1_output.json (COLLISION_POINTS).
    """
    collision_points = []
    
    # Load from phase1_output.json
    output_file = find_json_file(OUTPUT_JSON_PATH)
    if output_file:
        try:
            output_data = load_output_json(output_file)
            collision_points.extend(output_data["collision_points"])
        except Exception as e:
            print(f"Warning: Error loading collision spots from {output_file}: {e}")
    else:
        print(f"Warning: {OUTPUT_JSON_PATH} not found")
    
    return collision_points


def load_incident_spots() -> list:
    """
    Load incident risk spots from phase1_safety_output.json (from assets.type2_incidents).
    """
    incident_points = []
    
    # Load from phase1_safety_output.json
    safety_file = find_json_file(SAFETY_JSON_PATH)
    if safety_file:
        try:
            safety_data = load_safety_json(safety_file)
            incident_points.extend(safety_data["incident_points"])
        except Exception as e:
            print(f"Warning: Error loading incident spots from {safety_file}: {e}")
    else:
        print(f"Warning: {SAFETY_JSON_PATH} not found")
    
    return incident_points


async def get_risk_spots(request):
    """GET /api/risk-spots - Returns both collision and incident risk spots."""
    try:
        collision_points = load_collision_spots()
        incident_points = load_incident_spots()
        return web.json_response({
            "collision_points": collision_points,
            "incident_points": incident_points
        })
    except Exception as e:
        return web.json_response(
            {"error": f"Error loading risk spots: {str(e)}"},
            status=500
        )


async def get_collision_spots(request):
    """GET /api/risk-spots/collisions - Returns only collision risk spots."""
    try:
        collision_points = load_collision_spots()
        return web.json_response({"points": collision_points})
    except Exception as e:
        return web.json_response(
            {"error": f"Error loading collision spots: {str(e)}"},
            status=500
        )


async def get_incident_spots(request):
    """GET /api/risk-spots/incidents - Returns only incident risk spots."""
    try:
        incident_points = load_incident_spots()
        return web.json_response({"points": incident_points})
    except Exception as e:
        return web.json_response(
            {"error": f"Error loading incident spots: {str(e)}"},
            status=500
        )


async def health_check(request):
    """GET /health - Health check endpoint."""
    return web.Response(text="OK")


async def main():
    """Start the API server."""
    app = web.Application()
    
    # Add routes
    app.router.add_get("/api/risk-spots", get_risk_spots)
    app.router.add_get("/api/risk-spots/collisions", get_collision_spots)
    app.router.add_get("/api/risk-spots/incidents", get_incident_spots)
    app.router.add_get("/health", health_check)
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "PUT", "DELETE"]
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Get port from environment or use default
    port = int(os.environ.get("RISK_SPOTS_PORT", "8001"))
    host = os.environ.get("RISK_SPOTS_HOST", "0.0.0.0")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    print(f"Risk spots API server running on http://{host}:{port}")
    print(f"Endpoints:")
    print(f"  GET /api/risk-spots - All risk spots")
    print(f"  GET /api/risk-spots/collisions - Collision spots only")
    print(f"  GET /api/risk-spots/incidents - Incident spots only")
    print(f"  GET /health - Health check")
    
    await site.start()
    await asyncio.Event().wait()  # Keep server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())

