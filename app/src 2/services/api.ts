import { RiskPoint, IncidentPoint, Hotspot } from "../types";

// API Configuration
const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || "http://localhost:8000";
const RISK_SPOTS_API_URL = process.env.EXPO_PUBLIC_RISK_SPOTS_API_URL || "http://localhost:8001";
const AUSTIN_API_URL = "https://data.austintexas.gov/resource/dx9v-zd7x.json";

// Austin API response type
interface AustinIncident {
  traffic_report_id: string;
  published_date?: string;
  issue_reported?: string;
  address?: string;
  agency?: string;
  latitude?: string;
  longitude?: string;
  location?: { type: "Point"; coordinates: [number, number] };
}

// Safety Assets API response type
interface SafetyAsset {
  asset_id: string;
  asset_type: string;
  lat: number;
  lon: number;
  expected_hit?: number;
  mean_distance_km?: number;
  covers?: Array<{ lat: number; lon: number; weight: number; distance_km: number }>;
}

interface SafetyAssetsResponse {
  assets: {
    type1_collisions: SafetyAsset[];
    type2_incidents: SafetyAsset[];
  };
}

/**
 * Fetch traffic incidents from Austin SODA API
 */
export async function fetchIncidents(): Promise<IncidentPoint[]> {
  try {
    // Simple query - get most recent incidents
    // Austin SODA API may not support complex $where clauses, so we'll fetch recent ones
    // and filter client-side if needed
    const url = `${AUSTIN_API_URL}?$order=published_date DESC&$limit=100`;
    const response = await fetch(url);
    
    if (!response.ok) {
      // If API returns error, log details and return empty array
      const errorText = await response.text().catch(() => 'Unable to read error response');
      console.warn(`Austin API returned ${response.status}: ${response.statusText}`);
      console.warn(`Error details: ${errorText}`);
      return [];
    }
    
    const data: AustinIncident[] = await response.json();
    
    // If no data, return empty array
    if (!Array.isArray(data) || data.length === 0) {
      return [];
    }
    
    // Filter to incidents from last hour (client-side filtering)
    const oneHourAgo = Date.now() - 60 * 60 * 1000;
    const recentIncidents = data.filter(incident => {
      if (!incident.published_date) return false;
      const publishedTime = new Date(incident.published_date).getTime();
      return !isNaN(publishedTime) && publishedTime >= oneHourAgo;
    });
    
    // Transform to IncidentPoint[]
    return recentIncidents.map((incident): IncidentPoint => {
      // Extract position from location.coordinates or longitude/latitude
      let position: [number, number] | null = null;
      
      if (incident.location?.coordinates && incident.location.coordinates.length === 2) {
        position = [incident.location.coordinates[0], incident.location.coordinates[1]];
      } else if (incident.longitude && incident.latitude) {
        const lon = parseFloat(incident.longitude);
        const lat = parseFloat(incident.latitude);
        if (!isNaN(lon) && !isNaN(lat)) {
          position = [lon, lat];
        }
      }
      
      // Default position if missing (shouldn't happen, but handle gracefully)
      if (!position) {
        position = [0, 0];
      }
      
      // Determine type from issue_reported
      const issueLower = (incident.issue_reported || "").toLowerCase();
      const type: "collision" | "incident" = 
        issueLower.includes("crash") || issueLower.includes("collision") 
          ? "collision" 
          : "incident";
      
      // Convert published_date to timestamp (milliseconds)
      let timestamp: number | undefined;
      if (incident.published_date) {
        const date = new Date(incident.published_date);
        timestamp = isNaN(date.getTime()) ? undefined : date.getTime();
      }
      
      return {
        id: incident.traffic_report_id,
        position,
        weight: 1.0, // Austin API doesn't provide weight, use default
        type,
        timestamp,
        description: incident.issue_reported || undefined,
      };
    }).filter(incident => incident.position[0] !== 0 || incident.position[1] !== 0); // Filter out invalid positions
  } catch (error) {
    // Network errors or JSON parsing errors - return empty array instead of throwing
    console.warn("Error fetching incidents:", error);
    return [];
  }
}

/**
 * Fetch safety assets and transform to Hotspot[]
 */
export async function fetchSafetyAssets(): Promise<Hotspot[]> {
  try {
    const url = `${API_BASE_URL}/api/safety-assets`;
    const response = await fetch(url);
    
    if (!response.ok) {
      // If API returns error or 404, return empty array instead of throwing
      console.warn(`Safety assets API returned ${response.status}: ${response.statusText}`);
      return [];
    }
    
    const data: SafetyAssetsResponse = await response.json();
    
    // If no assets data, return empty array
    if (!data.assets || (!data.assets.type1_collisions && !data.assets.type2_incidents)) {
      return [];
    }
    
    const hotspots: Hotspot[] = [];
    
    // Process type1_collisions (collision assets)
    const collisionAssets = data.assets?.type1_collisions || [];
    collisionAssets.forEach((asset, index) => {
      // Calculate risk from expected_hit or average weight from covers
      let risk = asset.expected_hit || 0;
      if (risk === 0 && asset.covers && asset.covers.length > 0) {
        const avgWeight = asset.covers.reduce((sum, cover) => sum + cover.weight, 0) / asset.covers.length;
        risk = avgWeight;
      }
      
      // Calculate response time from mean_distance_km (assume 60 km/h average speed)
      // responseTime in seconds = (distance_km / speed_kmh) * 3600
      let responseTime = 120; // default 2 minutes
      if (asset.mean_distance_km) {
        const speedKmh = 60; // 60 km/h average
        responseTime = Math.round((asset.mean_distance_km / speedKmh) * 3600);
      }
      
      hotspots.push({
        id: asset.asset_id,
        position: [asset.lon, asset.lat],
        risk,
        responseTime,
        priority: index + 1,
      });
    });
    
    // Process type2_incidents (incident assets)
    const incidentAssets = data.assets?.type2_incidents || [];
    const collisionCount = collisionAssets.length;
    incidentAssets.forEach((asset, index) => {
      let risk = asset.expected_hit || 0;
      if (risk === 0 && asset.covers && asset.covers.length > 0) {
        const avgWeight = asset.covers.reduce((sum, cover) => sum + cover.weight, 0) / asset.covers.length;
        risk = avgWeight;
      }
      
      let responseTime = 120;
      if (asset.mean_distance_km) {
        const speedKmh = 60;
        responseTime = Math.round((asset.mean_distance_km / speedKmh) * 3600);
      }
      
      hotspots.push({
        id: asset.asset_id,
        position: [asset.lon, asset.lat],
        risk,
        responseTime,
        priority: collisionCount + index + 1,
      });
    });
    
    return hotspots;
  } catch (error) {
    // Network errors or JSON parsing errors - return empty array instead of throwing
    console.warn("Error fetching safety assets:", error);
    return [];
  }
}

// Risk spots API response type
interface RiskSpotsResponse {
  collision_points: RiskPoint[];
  incident_points: RiskPoint[];
}

/**
 * Fetch risk spots (collision and incident) from the risk spots API
 */
export async function fetchRiskSpots(): Promise<{ collisionPoints: RiskPoint[]; incidentPoints: RiskPoint[] }> {
  try {
    const url = `${RISK_SPOTS_API_URL}/api/risk-spots`;
    const response = await fetch(url);
    
    if (!response.ok) {
      console.warn(`Risk spots API returned ${response.status}: ${response.statusText}`);
      return { collisionPoints: [], incidentPoints: [] };
    }
    
    const data: RiskSpotsResponse = await response.json();
    
    return {
      collisionPoints: data.collision_points || [],
      incidentPoints: data.incident_points || [],
    };
  } catch (error) {
    console.warn("Error fetching risk spots:", error);
    return { collisionPoints: [], incidentPoints: [] };
  }
}

