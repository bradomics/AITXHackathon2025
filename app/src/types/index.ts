// Shared types for the application

export type UserRole = "ems" | "roadside" | "public-safety";

export interface Vehicle {
  "vehicle-id": string | number;
  lat: number;
  lon: number;
  heading: number;
  speed?: number;
  type?: string;
}

export interface RiskPoint {
  position: [number, number]; // [longitude, latitude]
  weight: number;
}

export interface IncidentPoint extends RiskPoint {
  id?: string;
  type?: "collision" | "incident";
  timestamp?: number;
  description?: string;
}

export interface RecommendedSpot {
  id: string;
  position: [number, number];
  type: "hospital" | "service-station" | "police-station" | "fire-station" | "other";
  name: string;
  distance?: number; // in meters
}

export interface Hotspot {
  id: string;
  position: [number, number]; // [longitude, latitude]
  risk: number; // risk level (0-1)
  responseTime: number; // response time in seconds
  priority?: number; // priority ranking
}



