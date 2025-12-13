"use client";

import * as React from "react";
import Map from "react-map-gl/mapbox";
import { NavigationControl } from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";

import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

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

// Demo points (lon, lat). Replace with your real incident/visitor/crash data.
const DEMO_POINTS: HeatPoint[] = [
    { position: [-97.7431, 30.2672], weight: 1 }, // Downtown
    { position: [-97.7680, 30.2850], weight: 0.8 }, // West Campus-ish
    { position: [-97.7219, 30.2849], weight: 0.6 }, // East Austin
    { position: [-97.7000, 30.2600], weight: 0.9 }, // Riverside-ish
    { position: [-97.7500, 30.2400], weight: 0.7 }, // South Lamar-ish
];

export function AustinHeatmapCard() {
    const [viewState, setViewState] = React.useState({
        longitude: AUSTIN_CENTER.longitude,
        latitude: AUSTIN_CENTER.latitude,
        zoom: 11.5,
        pitch: 0,
        bearing: 0,
    });

    const layers = React.useMemo(
        () => [
            new HeatmapLayer<HeatPoint>({
                id: "austin-heatmap",
                data: DEMO_POINTS,
                getPosition: (d) => d.position,
                getWeight: (d) => d.weight ?? 1,

                // Tune these
                radiusPixels: 60,
                intensity: 1.2,
                threshold: 0.05,
                aggregation: "SUM",
            }),
        ],
        []
    );

    return (
        <Card className="@container/card mb-4">
            <CardHeader>
                <CardTitle>Austin Heatmap</CardTitle>
                <CardDescription>
                    Example heatmap overlay (swap in your real Austin incident/visitor points)
                </CardDescription>
            </CardHeader>

            <div className="h-[420px] w-full overflow-hidden rounded-b-xl">
                <div className="relative h-[360px] w-full overflow-hidden rounded-b-xl">
                    <DeckGL
                        initialViewState={{ latitude: 30.2672, longitude: -97.7431, zoom: 11.5 }}
                        controller
                        layers={layers}
                        style={{ position: "absolute", inset: 0 }}   // 👈 force fill
                    >
                        <Map
                            mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
                            mapStyle="mapbox://styles/mapbox/dark-v11"
                            style={{ position: "absolute", inset: 0 }} // 👈 force fill
                        />
                    </DeckGL>
                </div>
            </div>
        </Card>
    );
}
