import React, { createContext, useContext, useState, useEffect, useRef, useMemo, ReactNode } from "react";
import { Vehicle, RiskPoint, IncidentPoint, Hotspot } from "../types";
import { fetchIncidents, fetchSafetyAssets, fetchRiskSpots } from "../services/api";

interface DigitalTwinContextType {
  vehicles: Vehicle[];
  collisionPoints: RiskPoint[]; // Collision RISK SPOTS from API (for visualization)
  incidentRiskSpots: RiskPoint[]; // Incident RISK SPOTS from API (for visualization)
  incidentPoints: IncidentPoint[]; // Live incidents from Austin API (actual incidents)
  hotspots: Hotspot[];
  isLoading: {
    incidents: boolean;
    safetyAssets: boolean;
    riskSpots: boolean;
  };
  errors: {
    incidents: string | null;
    safetyAssets: string | null;
    riskSpots: string | null;
  };
  setVehicles: (vehicles: Vehicle[]) => void;
  setIncidentPoints: (points: IncidentPoint[]) => void;
  setHotspots: (hotspots: Hotspot[]) => void;
}

const DigitalTwinContext = createContext<DigitalTwinContextType | undefined>(undefined);

export function DigitalTwinProvider({ children }: { children: ReactNode }) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  // Live incidents from Austin API (actual incidents that happened)
  const [incidentPoints, setIncidentPoints] = useState<IncidentPoint[]>([]);
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  
  // RISK SPOTS from API (predicted risk areas, not actual incidents)
  // These are used for heatmap visualization to show predicted risk areas
  const [collisionPoints, setCollisionPoints] = useState<RiskPoint[]>([]);
  const [incidentRiskSpots, setIncidentRiskSpots] = useState<RiskPoint[]>([]);
  
  const [isLoading, setIsLoading] = useState({
    incidents: false,
    safetyAssets: false,
    riskSpots: false,
  });
  const [errors, setErrors] = useState<{
    incidents: string | null;
    safetyAssets: string | null;
    riskSpots: string | null;
  }>({
    incidents: null,
    safetyAssets: null,
    riskSpots: null,
  });

  // Refs to track if component is mounted and prevent state updates after unmount
  const isMountedRef = useRef(true);
  
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Fetch incidents from Austin API (poll every 60 seconds)
  useEffect(() => {
    const fetchIncidentsData = async () => {
      if (!isMountedRef.current) return;
      setIsLoading((prev) => ({ ...prev, incidents: true }));
      setErrors((prev) => ({ ...prev, incidents: null }));

      try {
        const data = await fetchIncidents();
        if (isMountedRef.current) {
          setIncidentPoints(data);
          // Clear any previous errors since we got a response (even if empty)
          setErrors((prev) => ({ ...prev, incidents: null }));
        }
      } catch (error) {
        // This should rarely happen now since API functions return empty arrays
        // Only catches unexpected errors
        console.warn("Unexpected error fetching incidents:", error);
        if (isMountedRef.current) {
          // Don't set error for empty data - just log it
          setErrors((prev) => ({ ...prev, incidents: null }));
        }
      } finally {
        if (isMountedRef.current) {
          setIsLoading((prev) => ({ ...prev, incidents: false }));
        }
      }
    };

    // Fetch immediately
    fetchIncidentsData();

    // Then poll every 60 seconds
    const intervalId = setInterval(fetchIncidentsData, 60 * 1000);

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  // Fetch safety assets (hotspots) from server API (poll every 10 minutes)
  useEffect(() => {
    const fetchSafetyAssetsData = async () => {
      if (!isMountedRef.current) return;
      setIsLoading((prev) => ({ ...prev, safetyAssets: true }));
      setErrors((prev) => ({ ...prev, safetyAssets: null }));

      try {
        const data = await fetchSafetyAssets();
        if (isMountedRef.current) {
          setHotspots(data);
          // Clear any previous errors since we got a response (even if empty)
          setErrors((prev) => ({ ...prev, safetyAssets: null }));
        }
      } catch (error) {
        // This should rarely happen now since API functions return empty arrays
        // Only catches unexpected errors
        console.warn("Unexpected error fetching safety assets:", error);
        if (isMountedRef.current) {
          // Don't set error for empty data - just log it
          setErrors((prev) => ({ ...prev, safetyAssets: null }));
        }
      } finally {
        if (isMountedRef.current) {
          setIsLoading((prev) => ({ ...prev, safetyAssets: false }));
        }
      }
    };

    // Fetch immediately
    fetchSafetyAssetsData();

    // Then poll every 10 minutes (600 seconds) - ML model outputs update less frequently
    const intervalId = setInterval(fetchSafetyAssetsData, 10 * 60 * 1000);

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  // Fetch risk spots (collision and incident) from server API (poll every 5 minutes)
  useEffect(() => {
    const fetchRiskSpotsData = async () => {
      if (!isMountedRef.current) return;
      setIsLoading((prev) => ({ ...prev, riskSpots: true }));
      setErrors((prev) => ({ ...prev, riskSpots: null }));

      try {
        const data = await fetchRiskSpots();
        if (isMountedRef.current) {
          setCollisionPoints(data.collisionPoints);
          setIncidentRiskSpots(data.incidentPoints);
          // Clear any previous errors since we got a response (even if empty)
          setErrors((prev) => ({ ...prev, riskSpots: null }));
        }
      } catch (error) {
        console.warn("Unexpected error fetching risk spots:", error);
        if (isMountedRef.current) {
          // Don't set error for empty data - just log it
          setErrors((prev) => ({ ...prev, riskSpots: null }));
        }
      } finally {
        if (isMountedRef.current) {
          setIsLoading((prev) => ({ ...prev, riskSpots: false }));
        }
      }
    };

    // Fetch immediately
    fetchRiskSpotsData();

    // Then poll every 5 minutes (300 seconds)
    const intervalId = setInterval(fetchRiskSpotsData, 5 * 60 * 1000);

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return (
    <DigitalTwinContext.Provider
      value={{
        vehicles,
        collisionPoints,
        incidentRiskSpots,
        incidentPoints,
        hotspots,
        isLoading,
        errors,
        setVehicles,
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



