import React, { useState, useCallback, useRef, useEffect } from "react";
import { View, StyleSheet, Platform } from "react-native";
import MapView, { Circle, Marker, Polyline, PROVIDER_GOOGLE, PROVIDER_DEFAULT } from "react-native-maps";
import { useApp } from "../../contexts/AppContext";
import { useDigitalTwin } from "../../contexts/DigitalTwinContext";
import { getLayerConfig } from "./RoleLayerConfig";
import { DEFAULT_VIEW_STATE } from "../../utils/map";
import { getMapStyle } from "../../utils/mapStyles";
import { IncidentPoint, Vehicle } from "../../types";
import { IncidentMarker } from "./IncidentMarker";
import { VehicleMarker } from "./VehicleMarker";

interface MapViewProps {
  onIncidentPress?: (incident: IncidentPoint) => void;
  onVehiclePress?: (vehicle: Vehicle) => void;
  onLongPress?: (coordinate: { latitude: number; longitude: number }) => void;
}

// Track vehicle paths for animation
interface VehiclePath {
  vehicleId: string | number;
  coordinates: Array<{ latitude: number; longitude: number }>;
  timestamp: number;
}

export function MapViewComponent({ onIncidentPress, onVehiclePress, onLongPress }: MapViewProps) {
  const { role, effectiveTheme } = useApp();
  const { vehicles, collisionPoints, incidentRiskSpots, incidentPoints } = useDigitalTwin();
  const [region, setRegion] = useState({
    latitude: DEFAULT_VIEW_STATE.latitude,
    longitude: DEFAULT_VIEW_STATE.longitude,
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  });
  const mapRef = useRef<MapView>(null);
  const layerConfig = getLayerConfig(role);
  const [vehiclePaths, setVehiclePaths] = useState<Map<string | number, VehiclePath>>(new Map());

  // Filter vehicles and incidents based on role
  const filteredVehicles = layerConfig?.vehicleFilter
    ? vehicles.filter(layerConfig.vehicleFilter)
    : vehicles;

  const filteredIncidents = layerConfig?.incidentFilter
    ? incidentPoints.filter(layerConfig.incidentFilter)
    : incidentPoints;

  // Track vehicle paths for drawing trails - only update when vehicles actually change
  const previousVehiclesStringRef = useRef<string>("");
  
  useEffect(() => {
    // Only update if vehicles actually changed (by comparing serialized data)
    const currentVehiclesString = JSON.stringify(
      filteredVehicles.map(v => ({ id: v["vehicle-id"], lat: v.lat, lon: v.lon }))
    );
    
    if (currentVehiclesString === previousVehiclesStringRef.current) {
      return; // No change, skip update
    }
    
    previousVehiclesStringRef.current = currentVehiclesString;

    setVehiclePaths((prevPaths) => {
      const newPaths = new Map(prevPaths);
      
      filteredVehicles.forEach((vehicle) => {
        const vehicleId = vehicle["vehicle-id"];
        const existingPath = newPaths.get(vehicleId) || {
          vehicleId,
          coordinates: [],
          timestamp: Date.now(),
        };

        // Only add if position actually changed
        const lastCoord = existingPath.coordinates[existingPath.coordinates.length - 1];
        const hasChanged = !lastCoord || 
          lastCoord.latitude !== vehicle.lat || 
          lastCoord.longitude !== vehicle.lon;

        if (hasChanged) {
          // Add current position to path
          existingPath.coordinates.push({
            latitude: vehicle.lat,
            longitude: vehicle.lon,
          });

          // Keep only last 50 points for performance
          if (existingPath.coordinates.length > 50) {
            existingPath.coordinates.shift();
          }
        }

        existingPath.timestamp = Date.now();
        newPaths.set(vehicleId, existingPath);
      });

      // Remove paths for vehicles that are no longer present (after 30 seconds)
      const now = Date.now();
      newPaths.forEach((path, vehicleId) => {
        if (now - path.timestamp > 30000) {
          newPaths.delete(vehicleId);
        }
      });

      return newPaths;
    });
  }, [filteredVehicles]); // Depend on filteredVehicles, but check for actual changes inside

  // Use Apple Maps on iOS, Google Maps on Android
  const mapProvider = Platform.OS === "ios" ? PROVIDER_DEFAULT : PROVIDER_GOOGLE;
  
  // Get custom map style based on theme (works best with Google Maps)
  const customMapStyle = Platform.OS === "android" ? getMapStyle(effectiveTheme) : undefined;

  const handleIncidentPress = useCallback(
    (incident: IncidentPoint) => {
      onIncidentPress?.(incident);
    },
    [onIncidentPress]
  );

  const handleVehiclePress = useCallback(
    (vehicle: Vehicle) => {
      onVehiclePress?.(vehicle);
    },
    [onVehiclePress]
  );

  // Get vehicle color based on type
  const getVehicleColor = (vehicle: Vehicle): string => {
    const type = vehicle.type?.toLowerCase() || "";
    if (type.includes("ambulance") || type.includes("ems")) return "#FF0000"; // Red
    if (type.includes("police")) return "#0000FF"; // Blue
    if (type.includes("fire")) return "#FF8C00"; // Orange
    return "#737373"; // Neutral gray
  };

  // Get incident alert color based on type
  const getIncidentColor = (incident: IncidentPoint): string => {
    switch (incident.type) {
      case "collision":
        return "#EF4444"; // Red
      case "incident":
        return "#3B82F6"; // Blue
      default:
        return "#3B82F6"; // Blue
    }
  };

  // Scale weight to radius in meters
  // Weights typically range from 0.1 to 0.6, map to 100-600 meters radius
  const scaleWeightToRadius = (weight: number, minRadius: number = 100, maxRadius: number = 600): number => {
    // Normalize weight to 0-1 range (assuming weights are typically 0.1-0.6)
    const minWeight = 0.1;
    const maxWeight = 0.6;
    const normalizedWeight = Math.max(0, Math.min(1, (weight - minWeight) / (maxWeight - minWeight)));
    
    // Map to radius range
    return minRadius + (normalizedWeight * (maxRadius - minRadius));
  };

  // Scale weight to opacity (higher weight = more visible)
  const scaleWeightToOpacity = (weight: number, minOpacity: number = 0.2, maxOpacity: number = 0.5): number => {
    const minWeight = 0.1;
    const maxWeight = 0.6;
    const normalizedWeight = Math.max(0, Math.min(1, (weight - minWeight) / (maxWeight - minWeight)));
    return minOpacity + (normalizedWeight * (maxOpacity - minOpacity));
  };

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        provider={mapProvider}
        mapType="standard"
        customMapStyle={customMapStyle}
        initialRegion={region}
        onRegionChangeComplete={setRegion}
        onLongPress={(event) => {
          const { latitude, longitude } = event.nativeEvent.coordinate;
          onLongPress?.({ latitude, longitude });
        }}
        showsUserLocation={true}
        showsMyLocationButton={true}
        showsCompass={true}
        showsScale={true}
      >
        {/* Collision Hotspots - Red Transparent Circles */}
        {layerConfig?.showCollisionPoints &&
          collisionPoints.map((point, index) => {
            const [lng, lat] = point.position;
            const radius = scaleWeightToRadius(point.weight, 150, 900); // Scaled 1.5x (was 100-600)
            const opacity = scaleWeightToOpacity(point.weight, 0.2, 0.5);
            const strokeOpacity = Math.min(1, opacity + 0.3);
            
            return (
              <Circle
                key={`collision-${index}`}
                center={{
                  latitude: lat,
                  longitude: lng,
                }}
                radius={radius}
                fillColor={`rgba(255, 0, 0, ${opacity})`} // Red transparent, scaled by weight
                strokeColor={`rgba(255, 0, 0, ${strokeOpacity})`}
                strokeWidth={Math.max(2, Math.round(point.weight * 5))} // Thicker stroke for higher weight
                zIndex={1}
              />
            );
          })}

        {/* Incident Risk Spots - Blue Transparent Circles (hardcoded risk predictions) */}
        {layerConfig?.showIncidentPoints &&
          incidentRiskSpots.map((point, index) => {
            const [lng, lat] = point.position;
            const radius = scaleWeightToRadius(point.weight, 120, 750); // Scaled 1.5x (was 80-500)
            const opacity = scaleWeightToOpacity(point.weight, 0.2, 0.5);
            const strokeOpacity = Math.min(1, opacity + 0.3);
            
            return (
              <Circle
                key={`incident-risk-${index}`}
                center={{
                  latitude: lat,
                  longitude: lng,
                }}
                radius={radius}
                fillColor={`rgba(59, 130, 246, ${opacity})`} // Blue transparent, scaled by weight
                strokeColor={`rgba(59, 130, 246, ${strokeOpacity})`}
                strokeWidth={Math.max(2, Math.round(point.weight * 5))} // Thicker stroke for higher weight
                zIndex={1}
              />
            );
          })}

        {/* Vehicle Paths - Draw trails showing where vehicles have traveled */}
        {layerConfig?.showVehicles &&
          Array.from(vehiclePaths.values()).map((path) => {
            if (path.coordinates.length < 2) return null;

            const vehicle = filteredVehicles.find((v) => v["vehicle-id"] === path.vehicleId);
            if (!vehicle) return null;

            const pathColor = getVehicleColor(vehicle);

            return (
              <Polyline
                key={`path-${path.vehicleId}`}
                coordinates={path.coordinates}
                strokeColor={pathColor}
                strokeWidth={3}
                lineDashPattern={[5, 5]} // Dashed line for trail effect
                zIndex={2}
              />
            );
          })}

        {/* Vehicles - Current positions with markers */}
        {layerConfig?.showVehicles &&
          filteredVehicles.map((vehicle, index) => (
            <Marker
              key={`vehicle-${vehicle["vehicle-id"]}-${index}`}
              coordinate={{
                latitude: vehicle.lat,
                longitude: vehicle.lon,
              }}
              rotation={vehicle.heading}
              onPress={() => handleVehiclePress(vehicle)}
              anchor={{ x: 0.5, y: 0.5 }}
            >
              <VehicleMarker vehicle={vehicle} />
            </Marker>
          ))}

        {/* Incident Alerts - Markers for live traffic incidents */}
        {layerConfig?.showIncidentPoints &&
          filteredIncidents.map((incident, index) => {
            const [lng, lat] = incident.position;
            const alertColor = getIncidentColor(incident);

            return (
              <React.Fragment key={`incident-alert-${index}`}>
                {/* Alert circle around incident */}
                <Circle
                  center={{
                    latitude: lat,
                    longitude: lng,
                  }}
                  radius={100} // Alert radius in meters
                  fillColor={`${alertColor}20`} // Very transparent
                  strokeColor={alertColor}
                  strokeWidth={3}
                  zIndex={3}
                />
                {/* Incident marker */}
                <Marker
                  coordinate={{
                    latitude: lat,
                    longitude: lng,
                  }}
                  onPress={() => handleIncidentPress(incident)}
                  anchor={{ x: 0.5, y: 0.5 }}
                >
                  <IncidentMarker incident={incident} />
                </Marker>
              </React.Fragment>
            );
          })}

        {/* Custom POIs */}
        {layerConfig?.showCustomPOIs &&
          layerConfig.customPOIs?.map((poi) => {
            const [lng, lat] = poi.position;

            return (
              <Marker
                key={poi.id}
                coordinate={{
                  latitude: lat,
                  longitude: lng,
                }}
                title={poi.name}
                description={poi.type}
              >
                <View
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: 12,
                    backgroundColor:
                      poi.type === "hospital"
                        ? "#FF0000"
                        : poi.type === "service-station"
                        ? "#00FF00"
                        : poi.type === "police-station"
                        ? "#0000FF"
                        : poi.type === "fire-station"
                        ? "#FF8C00"
                        : "#808080",
                    borderWidth: 2,
                    borderColor: "#FFFFFF",
                  }}
                />
              </Marker>
            );
          })}
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    flex: 1,
  },
});
