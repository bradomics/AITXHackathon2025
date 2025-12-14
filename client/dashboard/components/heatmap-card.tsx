"use client";

import * as React from "react";
import { useState, useEffect, useMemo } from "react";
import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import { ScenegraphLayer } from "@deck.gl/mesh-layers";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Card, CardDescription, CardHeader, CardContent, CardTitle } from "@/components/ui/card";

import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

import assetPlacementSafetyOutput from "../public/outputs/phase1_safety_output.json";


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

type Trip = {
    path: [number, number][];
    timestamps: number[]; // seconds
};

type Incident = {
    traffic_report_id: string;
    published_date?: string;
    issue_reported?: string;
    address?: string;
    agency?: string;
    latitude?: string;
    longitude?: string;
    location?: { type: "Point"; coordinates: [number, number] };
};

type Asset = {
    asset_id: string;
    asset_type: string;
    lat: number;
    lon: number;
};

type DispatchTrip = {
    id: string;                 // unique trip id
    kind: "collision" | "incident";
    targetId: string;           // traffic_report_id
    assetId: string;

    path: [number, number][];
    timestamps: number[];        // absolute timestamps (seconds on global clock)
    endTime: number;             // for cleanup
};


const ASSET_PLACEMENT = assetPlacementSafetyOutput as any;


function midpoint(a: [number, number], b: [number, number]): [number, number] {
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

// Haversine distance in meters
function haversineMeters(a: [number, number], b: [number, number]) {
    const toRad = (d: number) => (d * Math.PI) / 180;
    const R = 6371000;
    const dLat = toRad(b[1] - a[1]);
    const dLon = toRad(b[0] - a[0]);
    const lat1 = toRad(a[1]);
    const lat2 = toRad(b[1]);

    const s =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;

    return 2 * R * Math.asin(Math.sqrt(s));
}

function buildTimestamps(path: [number, number][], metersPerSecond = 30) {
    // ~15 m/s ≈ 33 mph (tweak as you like)
    const ts: number[] = [0];
    let acc = 0;

    for (let i = 1; i < path.length; i++) {
        acc += haversineMeters(path[i - 1], path[i]) / metersPerSecond;
        ts.push(acc);
    }
    return ts;
}

function buildScaledTimestamps(path: [number, number][], targetDurationS = 20) {
    if (path.length < 2) return [0];

    const segMeters: number[] = [];
    let total = 0;

    for (let i = 1; i < path.length; i++) {
        const m = haversineMeters(path[i - 1], path[i]);
        segMeters.push(m);
        total += m;
    }

    // cumulative distance ratio -> timestamps
    const ts: number[] = [0];
    let acc = 0;

    for (let i = 0; i < segMeters.length; i++) {
        acc += segMeters[i];
        ts.push((acc / total) * targetDurationS);
    }

    return ts; // seconds, last = targetDurationS
}


function incidentLonLat(i: Incident): [number, number] | null {
    const c = i.location?.coordinates;
    if (c && c.length === 2) return [c[0], c[1]];

    const lon = i.longitude ? Number(i.longitude) : NaN;
    const lat = i.latitude ? Number(i.latitude) : NaN;
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;

    return [lon, lat];
}

function isCrashIncident(i: Incident) {
    const t = (i.issue_reported ?? "").toLowerCase();
    return t.includes("crash") || t.includes("collision");
}





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
    const [mapView, setMapView] = useState<"heatmap" | "digital-twin" | "composite-view">("heatmap");
    const [vehicles, setVehicles] = useState<(Vehicle & { type: VehicleType })[]>([]);
    const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");

    const seenIncidentIdsRef = React.useRef<Set<string>>(new Set());
    const [incidentMarkers, setIncidentMarkers] = useState<Incident[]>([]);


    // Trips
    const [simTime, setSimTime] = useState(0);              // global clock (seconds)
    const [trips, setTrips] = useState<DispatchTrip[]>([]); // active trips (max 10)
    const dispatchedIdsRef = React.useRef<Set<string>>(new Set()); // traffic_report_id we've dispatched
    function closestAsset(
        assets: Asset[],
        target: [number, number] // [lon, lat]
    ): Asset | null {
        let best: Asset | null = null;
        let bestM = Infinity;

        for (const a of assets) {
            const d = haversineMeters([a.lon, a.lat], target);
            if (d < bestM) {
                bestM = d;
                best = a;
            }
        }
        return best;
    }

    async function routeTrip(
        token: string,
        start: [number, number], // [lon, lat]
        end: [number, number]    // [lon, lat]
    ): Promise<[number, number][]> {
        const url =
            `https://api.mapbox.com/directions/v5/mapbox/driving/` +
            `${start[0]},${start[1]};${end[0]},${end[1]}` +
            `?geometries=geojson&overview=full&access_token=${token}`;

        const res = await fetch(url);
        const json = await res.json();
        return json?.routes?.[0]?.geometry?.coordinates ?? [];
    }


    // useEffect(() => {
    //     let isMounted = true;

    //     const fetchIncidents = async () => {
    //         try {
    //             const res = await fetch(
    //                 "https://data.austintexas.gov/resource/dx9v-zd7x.json?$order=published_date DESC&$limit=10"
    //             );
    //             if (!res.ok) return;

    //             const incidents = await res.json();

    //             if (!isMounted || !Array.isArray(incidents)) return;

    //             const newOnes = [];

    //             for (const incident of incidents) {
    //                 const id = incident.traffic_report_id;
    //                 if (!id) continue;

    //                 if (!seenIncidentIdsRef.current.has(id)) {
    //                     seenIncidentIdsRef.current.add(id);
    //                     newOnes.push(incident);
    //                 }
    //             }

    //             if (newOnes.length > 0) {
    //                 // 🔔 ALERT / HANDLE NEW EVENTS HERE
    //                 console.log("🚨 New traffic incidents:", newOnes);

    //                 // Example: browser alert (replace later)
    //                 newOnes.forEach((i) => {
    //                     console.log(
    //                         `[NEW] ${i.issue_reported} @ ${i.address} (${i.agency?.trim()})`
    //                     );
    //                 });

    //                 // Optional: setState(newOnes) if you want to render them
    //             }
    //         } catch (err) {
    //             // swallow network errors
    //         }
    //     };

    //     // initial fetch
    //     fetchIncidents();

    //     // poll every minute
    //     const intervalId = setInterval(fetchIncidents, 60_000);

    //     return () => {
    //         isMounted = false;
    //         clearInterval(intervalId);
    //     };
    // }, []);

    const { collisionAssets, incidentAssets } = useMemo(() => {
        const rawCollisions: Asset[] = (ASSET_PLACEMENT?.assets?.type1_collisions ?? []).map((a: any) => ({
            asset_id: a.asset_id,
            asset_type: a.asset_type,
            lat: Number(a.lat),
            lon: Number(a.lon),
        }));

        const rawIncidents: Asset[] = (ASSET_PLACEMENT?.assets?.type2_incidents ?? []).map((a: any) => ({
            asset_id: a.asset_id,
            asset_type: a.asset_type,
            lat: Number(a.lat),
            lon: Number(a.lon),
        }));

        const dedupe = (xs: Asset[]) => {
            const seen = new Set<string>();
            return xs.filter((x) => {
                if (!x.asset_id || !Number.isFinite(x.lat) || !Number.isFinite(x.lon)) return false;
                if (seen.has(x.asset_id)) return false;
                seen.add(x.asset_id);
                return true;
            });
        };

        return {
            collisionAssets: dedupe(rawCollisions),
            incidentAssets: dedupe(rawIncidents),
        };
    }, []);



    useEffect(() => {
        let isMounted = true;

        const fetchIncidents = async () => {
            try {
                const res = await fetch(
                    "https://data.austintexas.gov/resource/dx9v-zd7x.json?$order=published_date DESC&$limit=10"
                );
                if (!res.ok) return;

                const incidents: Incident[] = await res.json();
                if (!isMounted || !Array.isArray(incidents)) return;

                const newOnes: Incident[] = [];

                for (const incident of incidents) {
                    const id = incident.traffic_report_id;
                    if (!id) continue;

                    if (!seenIncidentIdsRef.current.has(id)) {
                        seenIncidentIdsRef.current.add(id);
                        newOnes.push(incident);
                    }
                }

                if (newOnes.length > 0) {
                    // markers
                    setIncidentMarkers((prev) => {
                        // keep list bounded so the browser stays fast
                        const next = [...newOnes, ...prev];
                        return next.slice(0, 100); // keep newest 100
                    });

                    // (optional) alert/log
                    newOnes.forEach((i) => {
                        console.log(`[NEW] ${i.issue_reported} @ ${i.address} (${i.agency?.trim()})`);
                    });
                }
            } catch {
                // ignore
            }
        };

        fetchIncidents();
        const intervalId = setInterval(fetchIncidents, 60_000);

        return () => {
            isMounted = false;
            clearInterval(intervalId);
        };
    }, []);


    const latestCrash = useMemo(() => {
        const crashes = incidentMarkers.filter(isCrashIncident);

        // because you're fetching with $order=published_date DESC, incidentMarkers[0] is often newest,
        // but we’ll be safe and sort by published_date.
        crashes.sort((a, b) => {
            const ta = Date.parse(a.published_date ?? "") || 0;
            const tb = Date.parse(b.published_date ?? "") || 0;
            return tb - ta;
        });

        return crashes[0] ?? null;
    }, [incidentMarkers]);

    const latestCrashPos = useMemo<[number, number] | null>(() => {
        if (!latestCrash) return null;
        return incidentLonLat(latestCrash);
    }, [latestCrash]);


    // -------- Heatmap stuff ---------
    const BASELINE_POINTS: HeatPoint[] = useMemo(() => makeAustinBaselineDense(0.005, 0.006), []);;

    // Generated 2025-12-13T12:39:37
    // target_bucket=2025-12-13T13:00:00
    const COLLISION_POINTS: HeatPoint[] = [
        { position: [-97.665321, 30.449991], weight: 0.582160 },
        { position: [-97.536064, 30.346228], weight: 0.451347 },
        { position: [-97.660706, 30.457869], weight: 0.437221 },
        { position: [-97.666298, 30.465401], weight: 0.435874 },
        { position: [-97.670921, 30.457523], weight: 0.424278 },
        { position: [-97.556465, 30.345551], weight: 0.416964 },
        { position: [-97.669937, 30.442114], weight: 0.411175 },
        { position: [-97.693146, 30.163204], weight: 0.404382 },
        { position: [-97.605309, 30.228052], weight: 0.313027 },
        { position: [-97.674561, 30.434235], weight: 0.305027 },
        { position: [-97.525864, 30.346565], weight: 0.299816 },
        { position: [-97.671791, 30.148493], weight: 0.294711 },
        { position: [-97.834869, 30.142717], weight: 0.280970 },
        { position: [-97.546265, 30.345890], weight: 0.267719 },
        { position: [-97.607246, 30.258886], weight: 0.249486 },
        { position: [-97.601662, 30.251352], weight: 0.246917 },
        { position: [-97.820450, 30.236010], weight: 0.227800 },
        { position: [-97.726616, 30.447903], weight: 0.223450 },
        { position: [-97.592651, 30.352060], weight: 0.201729 },
        { position: [-97.566666, 30.345209], weight: 0.190156 },
        { position: [-97.692863, 30.078184], weight: 0.187783 },
        { position: [-97.723724, 30.162134], weight: 0.180282 },
        { position: [-97.689926, 30.031897], weight: 0.180232 },
        { position: [-97.576866, 30.344868], weight: 0.178604 },
        { position: [-97.576103, 30.414383], weight: 0.172381 },
        { position: [-97.769096, 30.152802], weight: 0.171548 },
        { position: [-97.682953, 30.163561], weight: 0.169115 },
        { position: [-97.655106, 30.450336], weight: 0.167978 },
        { position: [-97.527779, 30.377384], weight: 0.165940 },
        { position: [-97.637589, 30.172861], weight: 0.165885 },
        { position: [-97.690369, 30.441420], weight: 0.162103 },
        { position: [-97.928024, 30.394470], weight: 0.159165 },
        { position: [-97.626907, 30.327736], weight: 0.158630 },
        { position: [-97.652382, 30.164621], weight: 0.153525 },
        { position: [-97.824440, 30.297703], weight: 0.151239 },
        { position: [-97.813484, 30.128021], weight: 0.148989 },
        { position: [-97.591469, 30.251698], weight: 0.147368 },
        { position: [-97.825058, 30.228115], weight: 0.145150 },
        { position: [-97.657516, 30.326694], weight: 0.143421 },
        { position: [-97.607986, 30.189331], weight: 0.137834 },
        { position: [-97.623024, 30.266075], weight: 0.135643 },
        { position: [-97.585892, 30.244162], weight: 0.135167 },
        { position: [-97.797424, 30.275475], weight: 0.131231 },
        { position: [-97.647560, 30.411985], weight: 0.128493 },
        { position: [-97.615501, 30.227705], weight: 0.127506 },
        { position: [-97.541290, 30.183868], weight: 0.127092 },
        { position: [-97.747055, 30.447199], weight: 0.124411 },
        { position: [-97.618179, 30.188982], weight: 0.123049 },
        { position: [-97.683929, 30.178984], weight: 0.122746 },
        { position: [-97.621315, 30.320202], weight: 0.120976 },
    ];

    const INCIDENT_POINTS: HeatPoint[] = [
        { position: [-97.735214, 30.262203], weight: 0.400909 },
        { position: [-97.740807, 30.269735], weight: 0.398782 },
        { position: [-97.695679, 30.364008], weight: 0.318901 },
        { position: [-97.751015, 30.269379], weight: 0.293355 },
        { position: [-97.766472, 30.191547], weight: 0.281844 },
        { position: [-97.730606, 30.270092], weight: 0.273627 },
        { position: [-97.745415, 30.261848], weight: 0.267062 },
        { position: [-97.790482, 30.167503], weight: 0.249672 },
        { position: [-97.707558, 30.309525], weight: 0.248548 },
        { position: [-97.768448, 30.222397], weight: 0.244395 },
        { position: [-97.690086, 30.356476], weight: 0.236192 },
        { position: [-97.699310, 30.340710], weight: 0.233938 },
        { position: [-97.752655, 30.215223], weight: 0.193987 },
        { position: [-97.669937, 30.442114], weight: 0.193754 },
        { position: [-97.676239, 30.380121], weight: 0.184585 },
        { position: [-97.723045, 30.231718], weight: 0.183032 },
        { position: [-97.755615, 30.261490], weight: 0.182601 },
        { position: [-97.800140, 30.476254], weight: 0.175804 },
        { position: [-97.715111, 30.347891], weight: 0.174635 },
        { position: [-97.734230, 30.246782], weight: 0.167197 },
        { position: [-97.651207, 30.388695], weight: 0.166357 },
        { position: [-97.739822, 30.254316], weight: 0.161166 },
        { position: [-97.704216, 30.417780], weight: 0.160737 },
        { position: [-97.641968, 30.404451], weight: 0.160495 },
        { position: [-97.728638, 30.239250], weight: 0.159232 },
        { position: [-97.672600, 30.403414], weight: 0.156963 },
        { position: [-97.716095, 30.363306], weight: 0.151409 },
        { position: [-97.794533, 30.468725], weight: 0.149707 },
        { position: [-97.744087, 30.400961], weight: 0.146887 },
        { position: [-97.776672, 30.191187], weight: 0.146134 },
        { position: [-97.766815, 30.276552], weight: 0.143992 },
        { position: [-97.661156, 30.303396], weight: 0.143786 },
        { position: [-97.708534, 30.324942], weight: 0.142158 },
        { position: [-97.712166, 30.301640], weight: 0.140387 },
        { position: [-97.691063, 30.371889], weight: 0.140038 },
        { position: [-97.725998, 30.277981], weight: 0.138144 },
        { position: [-97.678200, 30.410946], weight: 0.136959 },
        { position: [-97.703926, 30.332827], weight: 0.135578 },
        { position: [-97.741798, 30.285156], weight: 0.135209 },
        { position: [-97.750023, 30.253958], weight: 0.129883 },
        { position: [-97.686447, 30.379772], weight: 0.129707 },
        { position: [-97.743446, 30.231005], weight: 0.128366 },
        { position: [-97.746407, 30.277267], weight: 0.124138 },
        { position: [-97.743103, 30.385546], weight: 0.123019 },
        { position: [-97.826668, 30.173941], weight: 0.121859 },
        { position: [-97.668961, 30.426702], weight: 0.119423 },
        { position: [-97.761871, 30.199440], weight: 0.119127 },
        { position: [-97.721695, 30.370836], weight: 0.117396 },
        { position: [-97.758247, 30.222755], weight: 0.116516 },
        { position: [-97.775681, 30.175760], weight: 0.115650 },
    ];

    // RGBA stops (0-255). First stop is lowest intensity, last is hottest.
    const COLLISION_COLOR_RANGE: [number, number, number, number][] = [
        [255, 80, 80, 0],
        [255, 80, 80, 40],
        [255, 80, 80, 90],
        [255, 80, 80, 140],
        [255, 80, 80, 200],
        [255, 80, 80, 255],
    ];

    const INCIDENT_COLOR_RANGE: [number, number, number, number][] = [
        [80, 200, 255, 0],
        [80, 200, 255, 40],
        [80, 200, 255, 90],
        [80, 200, 255, 140],
        [80, 200, 255, 200],
        [80, 200, 255, 255],
    ];


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
                id: "austin-incident-points",
                data: INCIDENT_POINTS,
                getPosition: (d) => d.position,
                getWeight: (d) => d.weight ?? 1,
                radiusPixels: 50,
                intensity: 1,
                threshold: 0.03,
                aggregation: "SUM",
                colorRange: INCIDENT_COLOR_RANGE,
            }),
            new HeatmapLayer<HeatPoint>({
                id: "austin-collision-hotspots",
                data: COLLISION_POINTS,
                getPosition: (d) => d.position,
                getWeight: (d) => d.weight ?? 1,
                radiusPixels: 50,
                intensity: 1,
                threshold: 0.03,
                aggregation: "SUM",
                colorRange: COLLISION_COLOR_RANGE,
            }),
        ],
        [BASELINE_POINTS, INCIDENT_POINTS, COLLISION_POINTS]
    );


    // Fetch the latest incidents every minute
    useEffect(() => {
        let isMounted = true;

        const fetchIncidents = async () => {
            try {
                const res = await fetch(
                    "https://data.austintexas.gov/resource/dx9v-zd7x.json?$order=published_date DESC&$limit=10"
                );
                if (!res.ok) return;

                const data = await res.json();
                if (isMounted) {
                    console.log("Latest incidents:", data);
                    // setState(data) if needed
                }
            } catch {
                // ignore network errors
            }
        };

        // run immediately
        fetchIncidents();

        // then every minute
        const intervalId = setInterval(fetchIncidents, 60_000);

        return () => {
            isMounted = false;
            clearInterval(intervalId);
        };
    }, []);

    useEffect(() => {
        let raf = 0;
        let last = 0;
        const start = performance.now();

        const tick = (t: number) => {
            if (t - last > 400) { // update ~2.5 FPS for perf
                last = t;

                const nowS = (t - start) / 1000;
                setSimTime(nowS);

                // prune completed trips
                setTrips((prev) => prev.filter((tr) => nowS <= tr.endTime + 0.25));
            }
            raf = requestAnimationFrame(tick);
        };

        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, []);


    useEffect(() => {
        const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
        if (!token) return;

        if (!latestCrashPos || !latestCrash?.traffic_report_id) return;

        const targetId = latestCrash.traffic_report_id;

        const start: [number, number] = [AUSTIN_CENTER.longitude, AUSTIN_CENTER.latitude];
        const end: [number, number] = latestCrashPos;

        const url =
            `https://api.mapbox.com/directions/v5/mapbox/driving/` +
            `${start[0]},${start[1]};${end[0]},${end[1]}` +
            `?geometries=geojson&overview=full&access_token=${token}`;

        let cancelled = false;

        (async () => {
            try {
                const res = await fetch(url);
                const json = await res.json();

                const coords: [number, number][] = json?.routes?.[0]?.geometry?.coordinates ?? [];
                if (!coords.length || cancelled) return;

                const timestamps = buildTimestamps(coords, 60);

                arrivedRef.current = false;
                setActiveTripIncidentId(targetId);
                setTrip({ path: coords, timestamps });
                setCurrentTime(0);
            } catch {
                // ignore
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [latestCrashPos, latestCrash?.traffic_report_id]);


    // // optimized for browser performance
    // useEffect(() => {
    //     if (!trip) return;

    //     let raf = 0;
    //     let last = 0;
    //     const start = performance.now();
    //     const durationS = trip.timestamps.at(-1) ?? 20;

    //     const tick = (t: number) => {
    //         if (t - last > 400) {
    //             last = t;

    //             const elapsedS = (t - start) / 1000;
    //             const nextTime = Math.min(elapsedS, durationS);
    //             setCurrentTime(nextTime);

    //             // ✅ Arrived
    //             if (nextTime >= durationS && !arrivedRef.current) {
    //                 arrivedRef.current = true;

    //                 if (activeTripIncidentId) {
    //                     setIncidentMarkers((prev) =>
    //                         prev.filter((i) => i.traffic_report_id !== activeTripIncidentId)
    //                     );
    //                 }

    //                 // Optional: clear the trip after arrival (or keep it visible)
    //                 // setTrip(null);
    //             }

    //             // stop the animation once we hit the end
    //             if (nextTime >= durationS) return;
    //         }

    //         raf = requestAnimationFrame(tick);
    //     };

    //     raf = requestAnimationFrame(tick);
    //     return () => cancelAnimationFrame(raf);
    // }, [trip, activeTripIncidentId]);

    useEffect(() => {
        const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
        if (!token) return;

        // Only run this dispatcher in heatmap view (optional — remove this guard if you want always-on)
        if (mapView !== "heatmap") return;

        let cancelled = false;

        const dispatchOne = async (incident: Incident) => {
            const targetId = incident.traffic_report_id;
            if (!targetId) return;

            // already dispatched?
            if (dispatchedIdsRef.current.has(targetId)) return;
            dispatchedIdsRef.current.add(targetId);

            const end = incidentLonLat(incident);
            if (!end) return;

            const kind: "collision" | "incident" = isCrashIncident(incident) ? "collision" : "incident";
            const pool = kind === "collision" ? collisionAssets : incidentAssets;
            if (!pool.length) return;

            const asset = closestAsset(pool, end);
            if (!asset) return;

            const start: [number, number] = [asset.lon, asset.lat];

            try {
                const coords = await routeTrip(token, start, end);
                if (cancelled || !coords?.length) return;

                // timestamps relative, then shifted to global simTime
                const rel = buildTimestamps(coords, 60);               // tweak speed
                const startTime = simTime;
                const abs = rel.map((x) => x + startTime);
                const endTime = abs[abs.length - 1] ?? startTime;

                const newTrip: DispatchTrip = {
                    id: `${kind}-${asset.asset_id}-${targetId}-${Math.round(startTime * 1000)}`,
                    kind,
                    targetId,
                    assetId: asset.asset_id,
                    path: coords,
                    timestamps: abs,
                    endTime,
                };

                setTrips((prev) => {
                    // keep max 10
                    const next = [...prev, newTrip];
                    return next.length > 10 ? next.slice(next.length - 10) : next;
                });

                // Optional: when a trip completes, you might want to remove the marker.
                // If you DO want that, do it in the pruning step by comparing simTime > endTime
                // and removing incidentMarkers there (more work). For now: trip disappears, marker stays.
            } catch {
                // if routing fails, allow a future re-dispatch by removing from dispatched set
                dispatchedIdsRef.current.delete(targetId);
            }
        };

        // Dispatch for any newly-seen incident markers (your list is bounded to 100)
        (async () => {
            for (const inc of incidentMarkers) {
                await dispatchOne(inc);

                // Stop if we hit the max active trips
                // (so we don't create 50 routes in one render)
                if (cancelled) return;
                if (trips.length >= 10) return;
            }
        })();

        return () => {
            cancelled = true;
        };
        // IMPORTANT: include only stable deps; incidentMarkers changes should trigger dispatch
    }, [incidentMarkers, collisionAssets, incidentAssets, mapView, simTime, trips.length]);




    const tripsLayer = useMemo(() => {
        if (!trips.length) return null;

        return new TripsLayer<DispatchTrip>({
            id: "dispatch-trips",
            data: trips,
            getPath: (d) => d.path,
            getTimestamps: (d) => d.timestamps,
            currentTime: simTime,

            // trail length in seconds of *global clock*.
            // Use a fixed trailing window (looks good with multiple trips).
            trailLength: 25,
            fadeTrail: true,

            widthMinPixels: 6,
            opacity: 0.95,

            getColor: (d) => (d.kind === "collision" ? [255, 80, 80] : [80, 200, 255]),
        });
    }, [trips, simTime]);




    const incidentMarkerLayer = useMemo(() => {
        return new ScatterplotLayer<Incident>({
            id: "incident-markers",
            data: incidentMarkers,
            getPosition: (d) => {
                const c = d.location?.coordinates;
                if (c?.length === 2) return [c[0], c[1], 0];

                const lon = d.longitude ? Number(d.longitude) : NaN;
                const lat = d.latitude ? Number(d.latitude) : NaN;
                return [lon, lat, 0];
            },
            getRadius: 18,              // meters
            radiusUnits: "meters",
            radiusMinPixels: 6,
            radiusMaxPixels: 18,
            getFillColor: (d) => {
                // red-ish for crashes, blue-ish otherwise
                const t = (d.issue_reported ?? "").toLowerCase();
                const isCrash = t.includes("crash") || t.includes("collision");
                return isCrash ? [255, 80, 80, 220] : [80, 200, 255, 220];
            },
            pickable: true,
            onClick: (info) => {
                const d = info.object;
                if (!d) return;
                alert(`${d.issue_reported ?? "Incident"}\n${d.address ?? ""}\n${d.agency?.trim() ?? ""}`);
            },
            updateTriggers: {
                getFillColor: incidentMarkers, // safe: list is bounded
            },
        });
    }, [incidentMarkers]);




    // -------- SUMO stuff ----------

    function sumoAngleToDeckYaw(angle: number) {
        return (90 - angle + 360) % 360;
    }

    // --- Digital twin layers (multiple types) ---
    const digitalTwinLayers = useMemo(() => {
        // Per-model offsets/scales. Tweak yawOffset/pitch/roll to match your GLB "forward".
        const MODEL: Record<VehicleType, { url: string; sizeScale: number; yawOffset: number; flip: number; pitch: number; roll: number }> = {
            ambulance: { url: "/models/ambulance.glb", sizeScale: 1, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
            police: { url: "/models/police-car.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
            "red-car": { url: "/models/red-car.glb", sizeScale: 10, yawOffset: 270, flip: 0, pitch: 0, roll: 90 },
            "sports-car": { url: "/models/sports-car.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
            cybertruck: { url: "/models/cybertruck.glb", sizeScale: 50, yawOffset: 20, flip: 0, pitch: 0, roll: 90 },
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
            <CardHeader className="pt-4">
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
                    <div className="relative h-[600px] w-full">
                        <DeckGL
                            initialViewState={{ latitude: AUSTIN_CENTER.latitude, longitude: AUSTIN_CENTER.longitude, zoom: 11.5 }}
                            controller
                            // layers={tripLayer ? [...heatmapLayers, tripLayer] : heatmapLayers}
                            layers={[
                                ...heatmapLayers,
                                tripsLayer,
                                incidentMarkerLayer,
                            ].filter(Boolean) as any}
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
                    <div className="relative h-[600px] w-full">
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
