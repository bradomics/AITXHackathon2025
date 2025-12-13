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

type Vehicle = {
  "vehicle-id": number;
  lat: number;
  lon: number;
  heading: number; // degrees
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

export function AustinHeatmapCard() {
  const [mapView, setMapView] = useState<"heatmap" | "digital-twin" | "composite-view">("digital-twin");
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");

  // --- Heatmap layers (your existing stuff) ---
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

  // --- Digital-twin scenegraph layer ---
  const digitalTwinLayers = useMemo(() => {
    // Deck.gl Scenegraph orientation = [pitch, yaw, roll] in degrees.
    // Your "heading" is yaw around Z axis; depending on your model forward axis,
    // you may need +90/-90 offset. Start with this and tweak if needed.
    const headingToYaw = (headingDeg: number) => headingDeg;

    return [
      new ScenegraphLayer<Vehicle>({
        id: "vehicles-scenegraph",
        data: vehicles,
        scenegraph: "/models/sports-car.glb", // put glb in /public/models/vehicle.glb
        sizeScale: 10, // tweak for your model
        getPosition: (d) => [d.lon, d.lat, 0],
        getOrientation: (d) => [0, headingToYaw(d.heading), 0],
        _lighting: "pbr",
        pickable: true,
        updateTriggers: {
          getPosition: vehicles,
          getOrientation: vehicles,
        },
      }),
    ];
  }, [vehicles]);

  // Only connect WS when digital-twin is selected
  useEffect(() => {
    if (mapView !== "digital-twin") {
      setWsStatus("disconnected");
      setVehicles([]); // optional: clear when leaving view
      return;
    }

    const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://0.0.0.0:8765";
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
        if (msg?.vehicles?.length) {
          setVehicles(msg.vehicles);
        } else {
          setVehicles([]);
        }
      } catch {
        // ignore bad packets
      }
    };

    return () => {
      isClosed = true;
      try {
        ws?.close();
      } catch {}
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
              // Example: composite could be heatmap + vehicles
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
