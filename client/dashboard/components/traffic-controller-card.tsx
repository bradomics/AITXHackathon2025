"use client";

import * as React from "react";
import { useEffect, useMemo, useState } from "react";
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

type VehicleType =
  | "ambulance"
  | "police"
  | "red-car"
  | "sports-car"
  | "cybertruck"
  | "orange-sports-car"
  | "pickup-truck"
  | "suv"
  | "dodge-challenger"
  | "nissan-gtr";

type Vehicle = {
  "vehicle-id": string | number;
  lat: number;
  lon: number;
  heading: number; // degrees (SUMO: 0=N, 90=E, ...)
  type?: string; // backend-provided vType string
};

type VehicleMessage = {
  t?: number;
  vehicles: Vehicle[];
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
  id: string;
  kind: "collision" | "incident";
  targetId: string;
  assetId: string;
  path: [number, number][];
  timestamps: number[];
  endTime: number;
};

const ASSET_PLACEMENT = assetPlacementSafetyOutput as any;

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
  const ts: number[] = [0];
  let acc = 0;
  for (let i = 1; i < path.length; i++) {
    acc += haversineMeters(path[i - 1], path[i]) / metersPerSecond;
    ts.push(acc);
  }
  return ts;
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

function normalizeVehicleType(raw: unknown, vehicleId?: any): VehicleType {
  const t = typeof raw === "string" ? raw.trim().toLowerCase() : "";

  const idNum =
    typeof vehicleId === "number"
      ? vehicleId
      : typeof vehicleId === "string"
        ? parseInt(vehicleId.replace(/\D+/g, ""), 10)
        : undefined;

  // ✅ Exact matches from backend / SUMO vType ids
  if (t === "ambulance") return "ambulance";
  if (t === "police") return "police";
  if (t === "red-car" || t === "redcar" || t === "red_car") return "red-car";
  if (t === "sports-car" || t === "sportscar" || t === "sports_car") return "sports-car";
  if (t === "cybertruck") return "cybertruck";
  if (t === "orange-sports-car" || t === "orange-sportscar" || t === "orange_sports_car") return "orange-sports-car";
  if (t === "pickup-truck" || t === "pickuptruck" || t === "pickup_truck") return "pickup-truck";
  if (t === "suv") return "suv";
  if (t === "dodge-challenger" || t === "dodgechallenger" || t === "dodge_challenger") return "dodge-challenger";
  if (t === "nissan-gtr" || t === "nissangtr" || t === "nissan_gtr") return "nissan-gtr";

  // Common variants
  if (t === "police-car" || t === "policecar" || t === "cop") return "police";
  if (t === "ambulance-car" || t === "ems") return "ambulance";

  // SUMO generic types (if your routes/flows still use passenger/truck, etc.)
  if (t === "passenger" || t === "car" || t === "sedan") return "red-car";
  if (t === "truck" || t === "delivery") return "pickup-truck"; // better than cybertruck fallback for “truck”

  // Deterministic fallback (so you always see a mix)
  if (typeof idNum === "number") {
    const mod = idNum % 10;
    if (mod === 0) return "ambulance";
    if (mod === 1) return "police";
    if (mod <= 5) return "red-car"; // 50%
    if (mod <= 7) return "sports-car"; // 20%
    if (mod === 8) return "suv"; // 10%
    return "pickup-truck"; // 20%
  }

  return "red-car";
}

function sumoAngleToDeckYaw(angle: number) {
  // SUMO angle: 0=N,90=E,180=S,270=W
  // Deck yaw we want: East=0-ish; this converts well for most glTFs
  return (90 - angle + 360) % 360;
}

export function AustinTrafficControllerCard() {
  const [mapView, setMapView] = useState<"heatmap" | "digital-twin" | "composite-view">("digital-twin");
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");

  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  // --- Incident markers / trips (your existing logic preserved) ---
  const seenIncidentIdsRef = React.useRef<Set<string>>(new Set());
  const [incidentMarkers, setIncidentMarkers] = useState<Incident[]>([]);
  const completedTripIdsRef = React.useRef<Set<string>>(new Set());

  const [simTime, setSimTime] = useState(0);
  const [trips, setTrips] = useState<DispatchTrip[]>([]);
  const dispatchedIdsRef = React.useRef<Set<string>>(new Set());

  // --- Vehicles / rendering ---
  const latestVehiclesRef = React.useRef<(Vehicle & { type: VehicleType })[]>([]);
  const rafRef = React.useRef<number | null>(null);
  const [renderTick, setRenderTick] = useState(0);

  // Debug: counts by type we actually receive
  const [typeCounts, setTypeCounts] = useState<Record<string, number>>({});

  function scheduleRender() {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setRenderTick((x) => (x + 1) % 1_000_000);
    });
  }

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

  // Fetch incidents every minute (unchanged)
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
          setIncidentMarkers((prev) => [...newOnes, ...prev].slice(0, 100));
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

  // Global clock + prune trips (unchanged)
  useEffect(() => {
    let raf = 0;
    let last = 0;
    const start = performance.now();

    const tick = (t: number) => {
      if (t - last > 400) {
        last = t;
        const nowS = (t - start) / 1000;
        setSimTime(nowS);

        setTrips((prev) => {
          const completed = prev.filter((tr) => nowS > tr.endTime);

          if (completed.length) {
            const newlyCompletedTargetIds: string[] = [];
            for (const tr of completed) {
              if (!completedTripIdsRef.current.has(tr.id)) {
                completedTripIdsRef.current.add(tr.id);
                newlyCompletedTargetIds.push(tr.targetId);
              }
            }

            if (newlyCompletedTargetIds.length) {
              setIncidentMarkers((markers) => markers.filter((m) => !newlyCompletedTargetIds.includes(m.traffic_report_id)));
              newlyCompletedTargetIds.forEach((id) => dispatchedIdsRef.current.delete(id));
            }
          }

          return prev.filter((tr) => nowS <= tr.endTime + 0.25);
        });
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  function closestAsset(assets: Asset[], target: [number, number]) {
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

  async function routeTrip(token: string, start: [number, number], end: [number, number]) {
    const url =
      `https://api.mapbox.com/directions/v5/mapbox/driving/` +
      `${start[0]},${start[1]};${end[0]},${end[1]}` +
      `?geometries=geojson&overview=full&access_token=${token}`;

    const res = await fetch(url);
    const json = await res.json();
    return (json?.routes?.[0]?.geometry?.coordinates ?? []) as [number, number][];
  }

  // Dispatch trips in heatmap view (unchanged)
  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token) return;
    if (mapView !== "heatmap") return;

    let cancelled = false;

    const dispatchOne = async (incident: Incident) => {
      const targetId = incident.traffic_report_id;
      if (!targetId) return;

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

        const rel = buildTimestamps(coords, 60);
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
          const next = [...prev, newTrip];
          return next.length > 10 ? next.slice(next.length - 10) : next;
        });
      } catch {
        dispatchedIdsRef.current.delete(targetId);
      }
    };

    (async () => {
      for (const inc of incidentMarkers) {
        await dispatchOne(inc);
        if (cancelled) return;
        if (trips.length >= 10) return;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [incidentMarkers, collisionAssets, incidentAssets, mapView, simTime, trips.length]);

  // --- Heatmap layers (using your baseline; keep your existing INCIDENT/COLLISION arrays if you want) ---
  const BASELINE_POINTS: HeatPoint[] = useMemo(() => makeAustinBaselineDense(0.005, 0.006), []);
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
    ],
    [BASELINE_POINTS]
  );

  const tripsLayer = useMemo(() => {
    if (!trips.length) return null;
    return new TripsLayer<DispatchTrip>({
      id: "dispatch-trips",
      data: trips,
      getPath: (d) => d.path,
      getTimestamps: (d) => d.timestamps,
      currentTime: simTime,
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
      getRadius: 18,
      radiusUnits: "meters",
      radiusMinPixels: 6,
      radiusMaxPixels: 18,
      getFillColor: (d) => (isCrashIncident(d) ? [255, 80, 80, 220] : [80, 200, 255, 220]),
      pickable: true,
    });
  }, [incidentMarkers]);

  // ---- Vehicles for render ----
  const vehiclesForRender = useMemo(() => latestVehiclesRef.current, [renderTick]);

  function bucketByType(list: (Vehicle & { type: VehicleType })[]) {
    const b: Record<VehicleType, (Vehicle & { type: VehicleType })[]> = {
      ambulance: [],
      police: [],
      "red-car": [],
      "sports-car": [],
      cybertruck: [],
      "orange-sports-car": [],
      "pickup-truck": [],
      suv: [],
      "dodge-challenger": [],
      "nissan-gtr": [],
    };
    for (const v of list) b[v.type].push(v);
    return b;
  }

  const buckets = useMemo(() => bucketByType(vehiclesForRender), [vehiclesForRender]);

  const digitalTwinLayers = useMemo(() => {
    const MODEL: Record<
      VehicleType,
      { url: string; sizeScale: number; yawOffset: number; flip: number; pitch: number; roll: number }
    > = {
      ambulance: { url: "/models/ambulance.glb", sizeScale: 1, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
      police: { url: "/models/police-car.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
      "red-car": { url: "/models/red-car.glb", sizeScale: 10, yawOffset: 270, flip: 0, pitch: 0, roll: 90 },
      "sports-car": { url: "/models/sports-car.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },

      cybertruck: { url: "/models/cybertruck.glb", sizeScale: 50, yawOffset: 345, flip: 180, pitch: 0, roll: 90 },
      "orange-sports-car": {
        url: "/models/orange-sports-car.glb",
        sizeScale: 5,
        yawOffset: 90,
        flip: 0,
        pitch: 0,
        roll: 90,
      },
      "dodge-challenger": {
        url: "/models/dodge-challenger.glb",
        sizeScale: 2,
        yawOffset: 90,
        flip: 0,
        pitch: 0,
        roll: 90,
      },
      "nissan-gtr": { url: "/models/nissan-gtr.glb", sizeScale: 3, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
      "pickup-truck": {
        url: "/models/pickup-truck.glb",
        sizeScale: 5,
        yawOffset: 90,
        flip: 0,
        pitch: 0,
        roll: 90,
      },
      suv: { url: "/models/suv.glb", sizeScale: 5, yawOffset: 90, flip: 0, pitch: 0, roll: 90 },
    };

    const mkLayer = (type: VehicleType) => {
      const cfg = MODEL[type];
      const data = buckets[type];

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

    return [
      mkLayer("ambulance"),
      mkLayer("police"),
      mkLayer("red-car"),
      mkLayer("sports-car"),
      mkLayer("cybertruck"),
      mkLayer("orange-sports-car"),
      mkLayer("dodge-challenger"),
      mkLayer("nissan-gtr"),
      mkLayer("pickup-truck"),
      mkLayer("suv"),
    ];
  }, [buckets]);

  // ✅ WS connect only in digital-twin/composite
  useEffect(() => {
    const shouldConnect = mapView === "digital-twin" || mapView === "composite-view";
    if (!shouldConnect) {
      setWsStatus("disconnected");
      latestVehiclesRef.current = [];
      setTypeCounts({});
      return;
    }

    const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8765";

    let ws: WebSocket | null = null;
    let isClosed = false;

    setWsStatus("connecting");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => !isClosed && setWsStatus("connected");
    ws.onerror = () => !isClosed && setWsStatus("error");
    ws.onclose = () => !isClosed && setWsStatus("disconnected");

    ws.onmessage = (evt) => {
      try {
        const msg: VehicleMessage = JSON.parse(evt.data);

        const normalized = (msg?.vehicles ?? []).map((v) => {
          const type = normalizeVehicleType((v as any).type, v["vehicle-id"]);
          return { ...v, type };
        });

        latestVehiclesRef.current = normalized;

        // ✅ debug counts so you can confirm you’re receiving the new types
        const counts: Record<string, number> = {};
        for (const v of normalized) counts[v.type] = (counts[v.type] ?? 0) + 1;
        setTypeCounts(counts);

        scheduleRender();
      } catch {
        // ignore
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

  const infoLine = useMemo(() => {
    const total = Object.values(typeCounts).reduce((a, b) => a + b, 0);
    const keys = Object.keys(typeCounts).sort();
    const top = keys.map((k) => `${k}:${typeCounts[k]}`).join("  ");
    return { total, top };
  }, [typeCounts]);

  return (
    <Card className="overflow-hidden p-0">
      <CardHeader className="pt-4">
        <CardTitle>Traffic Controller View</CardTitle>
        <CardDescription className="pb-0">
          {mapView === "digital-twin"
            ? `Digital Twin (WS: ${wsStatus}, vehicles: ${infoLine.total})`
            : mapView === "composite-view"
              ? `Composite View (WS: ${wsStatus}, vehicles: ${infoLine.total})`
              : "Incident & Collision Hotspots"}
          {infoLine.top ? <div className="mt-1 text-xs opacity-80">{infoLine.top}</div> : null}
        </CardDescription>

        {/* ✅ FIXED toggle buttons */}
        <div className="flex flex-col items-start pb-0">
          <ButtonGroup>
            <Button
              onClick={() => setMapView("digital-twin")}
              variant={mapView === "digital-twin" ? "default" : "outline"}
            >
              Digital Twin
            </Button>
            <Button
              onClick={() => setMapView("heatmap")}
              variant={mapView === "heatmap" ? "default" : "outline"}
            >
              Heatmap
            </Button>
            <Button
              onClick={() => setMapView("composite-view")}
              variant={mapView === "composite-view" ? "default" : "outline"}
            >
              Composite
            </Button>
          </ButtonGroup>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {mapView === "heatmap" && (
          <div className="relative h-[600px] w-full">
            {mounted && (
              <DeckGL
                initialViewState={{ latitude: AUSTIN_CENTER.latitude, longitude: AUSTIN_CENTER.longitude, zoom: 11.5 }}
                controller
                layers={[...heatmapLayers, tripsLayer, incidentMarkerLayer].filter(Boolean) as any}
                useDevicePixels={1}
              >
                <Map reuseMaps mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN} mapStyle="mapbox://styles/mapbox/dark-v11" />
              </DeckGL>
            )}
          </div>
        )}

        {mapView === "digital-twin" && (
          <div className="relative h-[600px] w-full">
            {mounted && (
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
                useDevicePixels={1}
              >
                <Map reuseMaps mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN} mapStyle="mapbox://styles/mapbox/dark-v11" />
              </DeckGL>
            )}
          </div>
        )}

        {mapView === "composite-view" && (
          <div className="relative h-[600px] w-full">
            {mounted && (
              <DeckGL
                initialViewState={{
                  latitude: AUSTIN_CENTER.latitude,
                  longitude: AUSTIN_CENTER.longitude,
                  zoom: 12,
                  pitch: 45,
                  bearing: 0,
                }}
                controller
                layers={[...heatmapLayers, tripsLayer, incidentMarkerLayer, ...digitalTwinLayers].filter(Boolean) as any}
                useDevicePixels={1}
              >
                <Map reuseMaps mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN} mapStyle="mapbox://styles/mapbox/dark-v11" />
              </DeckGL>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
