import { UserRole } from "../../types";

export interface LayerConfig {
  showVehicles: boolean;
  showCollisionPoints: boolean;
  showIncidentPoints: boolean;
  showCustomPOIs: boolean;
  vehicleFilter?: (vehicle: any) => boolean;
  incidentFilter?: (incident: any) => boolean;
  customPOIs?: Array<{
    id: string;
    position: [number, number];
    type: string;
    name: string;
  }>;
}

export const roleLayerConfigs: Record<UserRole, LayerConfig> = {
  ems: {
    showVehicles: true,
    showCollisionPoints: true,
    showIncidentPoints: true,
    showCustomPOIs: true,
    vehicleFilter: (vehicle) => {
      // Show all vehicles, but prioritize EMS vehicles
      return true;
    },
    incidentFilter: (incident) => {
      // Show medical emergencies and accidents
      return true; // Show all incidents for EMS
    },
    customPOIs: [
      // Hospitals and medical facilities
      { id: "hosp1", position: [-97.7431, 30.2672], type: "hospital", name: "Austin General Hospital" },
      { id: "hosp2", position: [-97.75, 30.28], type: "hospital", name: "Central Medical Center" },
    ],
  },
  roadside: {
    showVehicles: true,
    showCollisionPoints: true,
    showIncidentPoints: true,
    showCustomPOIs: true,
    vehicleFilter: (vehicle) => {
      // Show all vehicles for roadside assistance
      return true;
    },
    incidentFilter: (incident) => {
      // Focus on vehicle breakdowns and collisions
      const type = incident.type?.toLowerCase() || "";
      return type.includes("breakdown") || type.includes("collision") || type.includes("accident");
    },
    customPOIs: [
      // Service stations and towing companies
      { id: "serv1", position: [-97.74, 30.27], type: "service-station", name: "Quick Service Station" },
      { id: "serv2", position: [-97.75, 30.26], type: "service-station", name: "24/7 Towing" },
    ],
  },
  "public-safety": {
    showVehicles: true,
    showCollisionPoints: true,
    showIncidentPoints: true,
    showCustomPOIs: true,
    vehicleFilter: (vehicle) => {
      // Show all vehicles for public safety view
      return true;
    },
    incidentFilter: (incident) => {
      // Show all incidents for public safety
      return true;
    },
    customPOIs: [
      // Police and fire stations
      { id: "police1", position: [-97.74, 30.27], type: "police-station", name: "Central Police Station" },
      { id: "fire1", position: [-97.75, 30.28], type: "fire-station", name: "Fire Station #1" },
    ],
  },
};

export function getLayerConfig(role: UserRole | null): LayerConfig | null {
  if (!role) return null;
  return roleLayerConfigs[role];
}



