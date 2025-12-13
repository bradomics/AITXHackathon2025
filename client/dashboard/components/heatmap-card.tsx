"use client";

import * as React from "react";
import { useState, useEffect, useMemo } from "react";
import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import { ScenegraphLayer } from "@deck.gl/mesh-layers";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Card, CardDescription, CardHeader, CardContent, CardTitle } from "@/components/ui/card";

import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

(mapboxgl as any).accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

type HeatPoint = { position: [number, number]; weight?: number };

type VehicleType = "ambulance" | "police" | "red-car" | "sports-car" | "cybertruck";

/**
 * Keep this permissive: backend may omit or send unexpected type strings.
 * We'll normalize on ingest.
 */
type Vehicle = {
    "vehicle-id": string | number;
    lat: number;
    lon: number;
    heading: number; // degrees
    type?: string;
};

type VehicleMessage = {
    t?: number;
    vehicles: Vehicle[];
};

const AUSTIN_CENTER = { latitude: 30.2672, longitude: -97.7431 };

function makeAustinBaselineDense(weight = 0.005, step = 0.006): HeatPoint[] {
    const west = -97.9;
    const east = -97.55;
    const south = 30.15;
    const north = 30.45;

    const pts: HeatPoint[] = [];
    for (let lon = west; lon <= east; lon += step) {
        for (let lat = south; lat <= north; lat += step) {
            pts.push({ position: [lon, lat], weight });
        }
    }
    return pts;
}

// function normalizeVehicleType(raw: unknown, vehicleId?: number): VehicleType {
//     const t = typeof raw === "string" ? raw.trim().toLowerCase() : "";

//     // Exact matches
//     if (t === "ambulance") return "ambulance";
//     if (t === "police") return "police";
//     if (t === "red-car") return "red-car";
//     if (t === "sports-car") return "sports-car";

//     // Common variants
//     if (t === "police-car" || t === "policecar" || t === "cop") return "police";
//     if (t === "red_car" || t === "redcar" || t === "car" || t === "sedan") return "red-car";
//     if (t === "ambulance-car" || t === "ems") return "ambulance";
//     if (t === "sports-car" || t === "ems") return "sports-car";


//     // Deterministic fallback (so you see a mix even if backend forgets type)
//     if (typeof vehicleId === "number") {
//         const mod = vehicleId % 3;
//         return mod === 0 ? "ambulance" : mod === 1 ? "police" : mod === 2 ? "red-car" : "sports-car";
//     }

//     return "red-car";
// }

function normalizeVehicleType(raw: unknown, vehicleId?: any): VehicleType {
  const t = typeof raw === "string" ? raw.trim().toLowerCase() : "";

  const idNum =
  typeof vehicleId === "number" ? vehicleId :
  typeof vehicleId === "string" ? parseInt(vehicleId.replace(/\D+/g, ""), 10) :
  undefined;

  // Exact matches from backend
  if (t === "ambulance") return "ambulance";
  if (t === "police") return "police";
  if (t === "red-car" || t === "redcar" || t === "red_car") return "red-car";
  if (t === "sports-car" || t === "sportscar" || t === "sports_car") return "sports-car";
  if (t === "cybertruck") return "cybertruck";

  // Common SUMO-ish / generic variants
  if (t === "police-car" || t === "policecar" || t === "cop") return "police";
  if (t === "ambulance-car" || t === "ems") return "ambulance";

  // If SUMO sends "passenger", "truck", etc, map them:
  if (t === "passenger" || t === "car" || t === "sedan") return "red-car";
  if (t === "truck" || t === "delivery") return "cybertruck";

  // Deterministic fallback
  if (typeof idNum === "number") {
    const mod = idNum % 10;
    if (mod === 0) return "ambulance";
    if (mod === 1) return "police";
    if (mod <= 6) return "red-car";       // ~60%
    if (mod <= 8) return "sports-car";    // ~20%
    return "cybertruck";                  // ~20%
  }

  return "red-car";
}

export function AustinHeatmapCard() {
    const [mapView, setMapView] = useState<"heatmap" | "digital-twin" | "composite-view">("digital-twin");
    const [vehicles, setVehicles] = useState<(Vehicle & { type: VehicleType })[]>([]);
    const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");

    // --- Heatmap layers ---
    const BASELINE_POINTS: HeatPoint[] = useMemo(() => makeAustinBaselineDense(0.005, 0.006), []);
    const HOTSPOTS: HeatPoint[] = useMemo(
        () => [
            { position: [-97.7431, 30.2672], weight: 1.2 },
            { position: [-97.768, 30.285], weight: 0.9 },
            { position: [-97.7219, 30.2849], weight: 0.7 },
            { position: [-97.7, 30.26], weight: 1.0 },
            { position: [-97.75, 30.24], weight: 1.4 },
        ],
        []
    );

    const heatmapLayers = useMemo(
        () => [
            new HeatmapLayer<HeatPoint>({
                id: "austin-baseline",
                data: BASELINE_POINTS,
                getPosition: (d) => d.position,
                getWeight: (d) => d.weight ?? 1,
                radiusPixels: 600,
                intensity: 0.005,
                threshold: 0.09,
                aggregation: "SUM",
            }),
            new HeatmapLayer<HeatPoint>({
                id: "austin-hotspots",
                data: HOTSPOTS,
                getPosition: (d) => d.position,
                getWeight: (d) => d.weight ?? 1,
                radiusPixels: 45,
                intensity: 1.3,
                threshold: 0.03,
                aggregation: "SUM",
            }),
        ],
        [BASELINE_POINTS, HOTSPOTS]
    );

    function sumoAngleToDeckYaw(angle: number) {
        return (90 - angle + 360) % 360;
    }

    // --- Digital twin layers (multiple types) ---
    const digitalTwinLayers = useMemo(() => {
        // Per-model offsets/scales. Tweak yawOffset/pitch/roll to match your GLB "forward".
        // const MODEL: Record<VehicleType, { url: string; sizeScale: number; yawOffset: number; flip: number; pitch: number; roll: number }> =
        // {
        //     ambulance: { url: "/models/ambulance.glb", sizeScale: 20, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
        //     police: { url: "/models/police-car.glb", sizeScale: 100, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
        //     "red-car": { url: "/models/red-car.glb", sizeScale: 5, yawOffset: 270, flip: 0, pitch: 0, roll: 90 },
        //     "sports-car": { url: "/models/sports-car.glb", sizeScale: 2000, yawOffset: 0, flip: 0, pitch: 0, roll: 0 },

        // };

        const MODEL: Record<VehicleType, { url: string; sizeScale: number; yawOffset: number; flip: number; pitch: number; roll: number }> = {
        ambulance: { url: "/models/ambulance.glb", sizeScale: 1, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
        police: { url: "/models/police-car.glb", sizeScale: 2, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
        "red-car": { url: "/models/red-car.glb", sizeScale: 10, yawOffset: 270, flip: 0, pitch: 0, roll: 90 },
        "sports-car": { url: "/models/sports-car.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
        cybertruck: { url: "/models/cybertruck.glb", sizeScale: 10, yawOffset: 0, flip: 0, pitch: 0, roll: 0 },
        };

        const mkLayer = (type: VehicleType) => {
            const cfg = MODEL[type];
            const data = vehicles.filter((v) => v.type === type);

            return new ScenegraphLayer<(Vehicle & { type: VehicleType })>({
                id: `vehicles-${type}`,
                data,
                scenegraph: cfg.url,
                sizeScale: cfg.sizeScale,
                getPosition: (d) => [d.lon, d.lat, 0],
                getOrientation: (d) => {
                const baseYaw = sumoAngleToDeckYaw(d.heading);
                const yaw = (baseYaw + cfg.yawOffset + cfg.flip) % 360;
                return [cfg.pitch, yaw, cfg.roll];
                },
                pickable: true,
                _lighting: "pbr",
            });
        };

        return [mkLayer("ambulance"), mkLayer("police"), mkLayer("red-car"), mkLayer("sports-car"), mkLayer("cybertruck")];
    }, [vehicles]);

    // Only connect WS when digital-twin is selected
    useEffect(() => {
        if (mapView !== "digital-twin") {
            setWsStatus("disconnected");
            setVehicles([]); // optional: clear when leaving view
            return;
        }

        // IMPORTANT: Browsers cannot connect to 0.0.0.0. Use localhost or an actual host/IP.
        const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8765";

        let ws: WebSocket | null = null;
        let isClosed = false;

        setWsStatus("connecting");
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            if (isClosed) return;
            setWsStatus("connected");
        };

        ws.onerror = () => {
            if (isClosed) return;
            setWsStatus("error");
        };

        ws.onclose = () => {
            if (isClosed) return;
            setWsStatus("disconnected");
        };

        ws.onmessage = (evt) => {
            try {
                const msg: VehicleMessage = JSON.parse(evt.data);

                const normalized = (msg?.vehicles ?? []).map((v) => {
                    const type = normalizeVehicleType((v as any).type, v["vehicle-id"]);
                    return { ...v, type };
                });

                setVehicles(normalized);
            } catch {
                // ignore bad packets
            }
        };

        return () => {
            isClosed = true;
            try {
                ws?.close();
            } catch { }
            ws = null;
        };
    }, [mapView]);

    const handleMapViewChange = (next: typeof mapView) => setMapView(next);

    return (
        <Card className="overflow-hidden p-0">
            <CardHeader className="p-6">
                <CardTitle>Austin Heatmap</CardTitle>
                <CardDescription className="pb-0 gap-0">
                    {mapView === "digital-twin" ? `Digital Twin (${wsStatus})` : "Traffic Hotspots"}
                </CardDescription>

                <div className="flex flex-col items-start pb-0">
                    <ButtonGroup>
                        <Button onClick={() => handleMapViewChange("heatmap")} variant={mapView === "heatmap" ? "default" : "outline"}>
                            Heatmap
                        </Button>
                        <Button
                            onClick={() => handleMapViewChange("digital-twin")}
                            variant={mapView === "digital-twin" ? "default" : "outline"}
                        >
                            Digital Twin
                        </Button>
                        <Button
                            onClick={() => handleMapViewChange("composite-view")}
                            variant={mapView === "composite-view" ? "default" : "outline"}
                        >
                            Composite View
                        </Button>
                    </ButtonGroup>
                </div>
            </CardHeader>

            <CardContent className="p-0">
                {mapView === "heatmap" && (
                    <div className="relative h-[460px] w-full">
                        <DeckGL
                            initialViewState={{ latitude: AUSTIN_CENTER.latitude, longitude: AUSTIN_CENTER.longitude, zoom: 11.5 }}
                            controller
                            layers={heatmapLayers}
                            style={{ position: "absolute", inset: 0 }}
                        >
                            <Map
                                mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
                                mapStyle="mapbox://styles/mapbox/dark-v11"
                                style={{ position: "absolute", inset: 0 }}
                            />
                        </DeckGL>
                    </div>
                )}

                {mapView === "digital-twin" && (
                    <div className="relative h-[460px] w-full">
                        <DeckGL
                            initialViewState={{
                                latitude: AUSTIN_CENTER.latitude,
                                longitude: AUSTIN_CENTER.longitude,
                                zoom: 12,
                                pitch: 45,
                                bearing: 0,
                            }}
                            controller
                            layers={digitalTwinLayers}
                            style={{ position: "absolute", inset: 0 }}
                        >
                            <Map
                                mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
                                mapStyle="mapbox://styles/mapbox/dark-v11"
                                style={{ position: "absolute", inset: 0 }}
                            />
                        </DeckGL>
                    </div>
                )}

                {mapView === "composite-view" && (
                    <div className="relative h-[460px] w-full">
                        <DeckGL
                            initialViewState={{
                                latitude: AUSTIN_CENTER.latitude,
                                longitude: AUSTIN_CENTER.longitude,
                                zoom: 12,
                                pitch: 45,
                                bearing: 0,
                            }}
                            controller
                            layers={[...heatmapLayers, ...digitalTwinLayers]}
                            style={{ position: "absolute", inset: 0 }}
                        >
                            <Map
                                mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
                                mapStyle="mapbox://styles/mapbox/dark-v11"
                                style={{ position: "absolute", inset: 0 }}
                            />
                        </DeckGL>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
