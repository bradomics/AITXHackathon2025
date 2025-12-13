"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import Map from "react-map-gl/mapbox";
import { NavigationControl } from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";

import { PlusIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ButtonGroup } from "@/components/ui/button-group"

import { Card, CardDescription, CardHeader, CardContent, CardTitle } from "@/components/ui/card";

// Mapbox CSS (put this in app/layout.tsx or globals.css instead if you prefer)
// import "mapbox-gl/dist/mapbox-gl.css";

declare module "*?worker" {
    const workerConstructor: new () => Worker;
    export default workerConstructor;
}

// import mapboxgl from "mapbox-gl";
// ✅ modern worker import (works with Next/Webpack 5)
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

(mapboxgl as any).accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;


type HeatPoint = { position: [number, number]; weight?: number };

const AUSTIN_CENTER = { latitude: 30.2672, longitude: -97.7431 };



function makeAustinBaselineGrid(weight = 0.005): HeatPoint[] {
    // Rough Austin bounding box (tweak if you want tighter)
    const west = -97.90;
    const east = -97.55;
    const south = 30.15;
    const north = 30.45;

    // Grid step in degrees (~0.01 deg lon/lat is ~0.6–0.7 miles here)
    const step = 0.012;

    const pts: HeatPoint[] = [];
    for (let lon = west; lon <= east; lon += step) {
        for (let lat = south; lat <= north; lat += step) {
            pts.push({ position: [lon, lat], weight });
        }
    }
    return pts;
}

const HOTSPOTS: HeatPoint[] = [
    { position: [-97.7431, 30.2672], weight: 1.0 },   // Downtown
    { position: [-97.7680, 30.2850], weight: 0.8 },   // West Campus-ish
    { position: [-97.7219, 30.2849], weight: 0.6 },   // East Austin
    { position: [-97.7000, 30.2600], weight: 0.9 },   // Riverside-ish
    { position: [-97.7500, 30.2400], weight: 1.2 },   // South Lamar-ish (make it pop)
];

// Exact grid baseline (more uniform haze)
function makeAustinBaselineDense(
    weight = 0.005,
    step = 0.006
): HeatPoint[] {
    const west = -97.90;
    const east = -97.55;
    const south = 30.15;
    const north = 30.45;

    const pts: HeatPoint[] = [];

    for (let lon = west; lon <= east; lon += step) {
        for (let lat = south; lat <= north; lat += step) {
            pts.push({
                position: [lon, lat],
                weight,
            });
        }
    }

    return pts;
}

const DEMO_POINTS: HeatPoint[] = [
    ...makeAustinBaselineDense(0.005),
    ...HOTSPOTS,
];

export function AustinHeatmapCard() {
    const [viewState, setViewState] = React.useState({
        longitude: AUSTIN_CENTER.longitude,
        latitude: AUSTIN_CENTER.latitude,
        zoom: 9,
        pitch: 0,
        bearing: 0,
    });

    const BASELINE_POINTS: HeatPoint[] = makeAustinBaselineDense(0.005, 0.006);

    const HOTSPOTS: HeatPoint[] = [
        { position: [-97.7431, 30.2672], weight: 1.2 },
        { position: [-97.7680, 30.2850], weight: 0.9 },
        { position: [-97.7219, 30.2849], weight: 0.7 },
        { position: [-97.7000, 30.2600], weight: 1.0 },
        { position: [-97.7500, 30.2400], weight: 1.4 },
    ];

    const layers = [
        new HeatmapLayer<HeatPoint>({
            id: "austin-baseline",
            data: BASELINE_POINTS,
            getPosition: d => d.position,
            getWeight: d => d.weight ?? 1,
            radiusPixels: 600,     // smoother haze
            intensity: 0.005,      // keep it subtle
            threshold: 0.09,
            aggregation: "SUM",
        }),

        new HeatmapLayer<HeatPoint>({
            id: "austin-hotspots",
            data: HOTSPOTS,
            getPosition: d => d.position,
            getWeight: d => d.weight ?? 1,
            radiusPixels: 45,     // tighter punch
            intensity: 1.3,       // make them pop
            threshold: 0.03,
            aggregation: "SUM",
        }),
    ];

    const [mapView, setMapView] = useState(viewState);

    const handleMapViewChange = (viewState: any) => {
        alert(`here is the view state: ${viewState}`);
        setMapView(viewState);
    }

    return (
        <Card className="overflow-hidden p-0">
            <CardHeader className="p-6">
                <CardTitle>Austin Heatmap</CardTitle>
                <CardDescription className="pb-0 gap-0">Traffic Hotspots</CardDescription>
                <div className="flex flex-col items-start pb-0">
                    <ButtonGroup>
                        <Button onClick={() => { handleMapViewChange('heatmap'); }} variant="outline">Heatmap</Button>
                        <Button onClick={() => { handleMapViewChange('digital-twin'); }} variant="outline">Digital Twin</Button>
                        <Button onClick={() => { handleMapViewChange('composite-view'); }} variant="outline">Composite View</Button>
                    </ButtonGroup>
                </div>
            </CardHeader>
            <CardContent className="p-0">
                <div className="relative h-[460px] w-full">
                    <DeckGL
                        initialViewState={{ latitude: 30.2672, longitude: -97.7431, zoom: 11.5 }}
                        controller
                        layers={layers}
                        style={{ position: "absolute", inset: 0 }}
                    >
                        <Map
                            mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
                            mapStyle="mapbox://styles/mapbox/dark-v11"
                            style={{ position: "absolute", inset: 0 }}
                        />
                    </DeckGL>
                </div>
            </CardContent>
        </Card>

    )

}
