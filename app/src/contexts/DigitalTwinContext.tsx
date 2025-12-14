import React, { createContext, useContext, useState, ReactNode } from "react";
import { Vehicle, RiskPoint, IncidentPoint, Hotspot } from "../types";

interface DigitalTwinContextType {
  vehicles: Vehicle[];
  collisionPoints: RiskPoint[];
  incidentPoints: IncidentPoint[];
  hotspots: Hotspot[];
  setVehicles: (vehicles: Vehicle[]) => void;
  setCollisionPoints: (points: RiskPoint[]) => void;
  setIncidentPoints: (points: IncidentPoint[]) => void;
  setHotspots: (hotspots: Hotspot[]) => void;
}

const DigitalTwinContext = createContext<DigitalTwinContextType | undefined>(undefined);

// Dummy collision points data (red circles) - from web app
const DUMMY_COLLISION_POINTS: RiskPoint[] = [
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
];

// Dummy incident points data (blue circles) - from web app
// Generate timestamps relative to now (varying from 1 min to 2 hours ago)
const now = Date.now();
const DUMMY_INCIDENT_POINTS: IncidentPoint[] = [
  { position: [-97.735214, 30.262203], weight: 0.400909, type: "collision", timestamp: now - 1 * 60 * 1000 }, // 1 min ago
  { position: [-97.740807, 30.269735], weight: 0.398782, type: "collision", timestamp: now - 5 * 60 * 1000 }, // 5 mins ago
  { position: [-97.695679, 30.364008], weight: 0.318901, type: "incident", timestamp: now - 15 * 60 * 1000 }, // 15 mins ago
  { position: [-97.751015, 30.269379], weight: 0.293355, type: "incident", timestamp: now - 30 * 60 * 1000 }, // 30 mins ago
  { position: [-97.766472, 30.191547], weight: 0.281844, type: "incident", timestamp: now - 45 * 60 * 1000 }, // 45 mins ago
  { position: [-97.730606, 30.270092], weight: 0.273627, type: "incident", timestamp: now - 60 * 60 * 1000 }, // 1 hr ago
  { position: [-97.745415, 30.261848], weight: 0.267062, type: "incident", timestamp: now - 90 * 60 * 1000 }, // 1.5 hrs ago
];

// Dummy hotspots data - best response time spots for highest risk areas
// This will come from backend in production
const DUMMY_HOTSPOTS: Hotspot[] = [
  { id: "1", position: [-97.665321, 30.449991], risk: 0.582160, responseTime: 120, priority: 1 }, // 2 min
  { id: "2", position: [-97.536064, 30.346228], risk: 0.451347, responseTime: 180, priority: 2 }, // 3 min
  { id: "3", position: [-97.660706, 30.457869], risk: 0.437221, responseTime: 150, priority: 3 }, // 2.5 min
  { id: "4", position: [-97.666298, 30.465401], risk: 0.435874, responseTime: 210, priority: 4 }, // 3.5 min
  { id: "5", position: [-97.670921, 30.457523], risk: 0.424278, responseTime: 165, priority: 5 }, // 2.75 min
  { id: "6", position: [-97.556465, 30.345551], risk: 0.416964, responseTime: 195, priority: 6 }, // 3.25 min
  { id: "7", position: [-97.669937, 30.442114], risk: 0.411175, responseTime: 240, priority: 7 }, // 4 min
  { id: "8", position: [-97.693146, 30.163204], risk: 0.404382, responseTime: 270, priority: 8 }, // 4.5 min
];

export function DigitalTwinProvider({ children }: { children: ReactNode }) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [collisionPoints, setCollisionPoints] = useState<RiskPoint[]>(DUMMY_COLLISION_POINTS);
  const [incidentPoints, setIncidentPoints] = useState<IncidentPoint[]>(DUMMY_INCIDENT_POINTS);
  const [hotspots, setHotspots] = useState<Hotspot[]>(DUMMY_HOTSPOTS);

  return (
    <DigitalTwinContext.Provider
      value={{
        vehicles,
        collisionPoints,
        incidentPoints,
        hotspots,
        setVehicles,
        setCollisionPoints,
        setIncidentPoints,
        setHotspots,
      }}
    >
      {children}
    </DigitalTwinContext.Provider>
  );
}

export function useDigitalTwin() {
  const context = useContext(DigitalTwinContext);
  if (context === undefined) {
    throw new Error("useDigitalTwin must be used within a DigitalTwinProvider");
  }
  return context;
}



