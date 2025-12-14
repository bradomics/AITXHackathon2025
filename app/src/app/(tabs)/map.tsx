import { useState, useEffect, useRef } from "react";
import { View, Modal, Text, ScrollView, TouchableOpacity, StyleSheet, Animated, Platform, Linking } from "react-native";
import { StatusBar } from "expo-status-bar";
import * as Location from "expo-location";
import { useApp } from "../../contexts/AppContext";
// WebSocket disabled - mobile app doesn't need vehicle simulation data
// import { useWebSocket } from "../../hooks/useWebSocket";
import { useNotifications } from "../../hooks/useNotifications";
import { useDigitalTwin } from "../../contexts/DigitalTwinContext";
import { MapViewComponent as MapView } from "../../components/map/MapView";
import { NotificationBanner } from "../../components/notifications/NotificationBanner";
import { NavigationPopup } from "../../components/map/NavigationPopup";
import { TutorialOverlay } from "../../components/tutorial/TutorialOverlay";
import { IncidentPoint, Vehicle, RecommendedSpot, Hotspot } from "../../types";
import { Button } from "../../components/ui/Button";

export default function MapScreen() {
  const { role, effectiveTheme } = useApp();
  const { incidentPoints, hotspots } = useDigitalTwin();
  const [selectedIncident, setSelectedIncident] = useState<IncidentPoint | null>(null);
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);
  const [recommendedSpots, setRecommendedSpots] = useState<RecommendedSpot[]>([]);
  const [navigationTarget, setNavigationTarget] = useState<{ latitude: number; longitude: number } | null>(null);
  const incidentSlideAnim = useRef(new Animated.Value(300)).current;
  const vehicleSlideAnim = useRef(new Animated.Value(300)).current;
  
  // Track ALL incidents we've ever seen (persists across re-renders)
  // Only notify for incidents we haven't seen before
  const seenIncidentIds = useRef<Set<string>>(new Set());
  const hasInitialized = useRef(false);

  const getTimeSince = (timestamp?: number): string => {
    if (!timestamp) return "Unknown";
    
    const now = Date.now();
    const diffMs = now - timestamp;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min${diffMins !== 1 ? "s" : ""}`;
    if (diffHours < 24) return `${diffHours} hr${diffHours !== 1 ? "s" : ""}`;
    return `${diffDays} day${diffDays !== 1 ? "s" : ""}`;
  };

  // WebSocket disabled - mobile app doesn't need vehicle simulation data
  // The web client uses this for simulation, but mobile app only needs incidents/hotspots

  // Set up notifications - ONLY for new incidents (batched)
  const { notifyNewIncidentsBatch } = useNotifications();

  // Request location permissions for user location on map
  useEffect(() => {
    const requestLocationPermissions = async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        console.warn("Location permission not granted");
      }
    };
    requestLocationPermissions();
  }, []);

  // Convert hotspots to recommended spots
  useEffect(() => {
    const spots: RecommendedSpot[] = hotspots.map((hotspot: Hotspot) => ({
      id: hotspot.id,
      position: hotspot.position,
      type: "other", // Hotspots don't have specific types, use "other"
      name: `High-Risk Area (${(hotspot.risk * 100).toFixed(0)}% risk)`,
      distance: undefined, // Could calculate if we have user location
    }));
    setRecommendedSpots(spots);
  }, [hotspots]);

  // Notify ONLY for truly new incidents (not seen before)
  useEffect(() => {
    // Skip if no incidents yet
    if (incidentPoints.length === 0) {
      return;
    }

    // On first load, mark all current incidents as "seen" without notifying
    if (!hasInitialized.current) {
      incidentPoints.forEach(inc => {
        const id = inc.id || "";
        if (id) {
          seenIncidentIds.current.add(id);
        }
      });
      hasInitialized.current = true;
      console.log(`[Notifications] Initial load: marked ${seenIncidentIds.current.size} incidents as seen`);
      return;
    }

    // Find incidents we haven't seen before
    const newIncidents = incidentPoints.filter(inc => {
      const id = inc.id || "";
      if (!id) return false;
      
      // If we haven't seen this ID before, it's new
      if (!seenIncidentIds.current.has(id)) {
        seenIncidentIds.current.add(id); // Mark as seen immediately
        return true;
      }
      return false;
    });

    // Batch notify for all new incidents in a single notification
    if (newIncidents.length > 0) {
      console.log(`[Notifications] Found ${newIncidents.length} new incident(s) - sending batched notification`);
      // Send ONE notification for all new incidents
      notifyNewIncidentsBatch(newIncidents.length, newIncidents).catch(err => {
        console.error("Error sending batched notification:", err);
      });
    }
  }, [incidentPoints, notifyNewIncidentsBatch]);

  // NO notifications for hotspots - only incidents!

  const handleIncidentPress = (incident: IncidentPoint) => {
    setSelectedIncident(incident);
    // Animate content sliding up
    Animated.spring(incidentSlideAnim, {
      toValue: 0,
      useNativeDriver: true,
      tension: 65,
      friction: 11,
    }).start();
  };

  const handleVehiclePress = (vehicle: Vehicle) => {
    setSelectedVehicle(vehicle);
    // Animate content sliding up
    Animated.spring(vehicleSlideAnim, {
      toValue: 0,
      useNativeDriver: true,
      tension: 65,
      friction: 11,
    }).start();
  };

  const handleCloseIncident = () => {
    Animated.timing(incidentSlideAnim, {
      toValue: 300,
      duration: 250,
      useNativeDriver: true,
    }).start(() => {
      setSelectedIncident(null);
      incidentSlideAnim.setValue(300); // Reset for next open
    });
  };

  const handleCloseVehicle = () => {
    Animated.timing(vehicleSlideAnim, {
      toValue: 300,
      duration: 250,
      useNativeDriver: true,
    }).start(() => {
      setSelectedVehicle(null);
      vehicleSlideAnim.setValue(300); // Reset for next open
    });
  };

  const handleDismissSpot = (spotId: string) => {
    setRecommendedSpots((prev) => prev.filter((spot) => spot.id !== spotId));
  };

  const handleSpotPress = (spot: RecommendedSpot) => {
    // Navigate to spot or show details
    console.log("Spot pressed:", spot);
  };

  const handleMapLongPress = (coordinate: { latitude: number; longitude: number }) => {
    setNavigationTarget(coordinate);
  };

  const handleNavigateIncident = (incident: IncidentPoint) => {
    const latitude = incident.position[1];
    const longitude = incident.position[0];

    if (Platform.OS === "ios") {
      // Open Apple Maps
      const url = `http://maps.apple.com/?daddr=${latitude},${longitude}&dirflg=d`;
      Linking.openURL(url).catch((err) => console.error("Error opening Apple Maps:", err));
    } else {
      // Open Google Maps
      const url = `https://www.google.com/maps/dir/?api=1&destination=${latitude},${longitude}`;
      Linking.openURL(url).catch((err) => console.error("Error opening Google Maps:", err));
    }
    
    // Close the modal after opening navigation
    handleCloseIncident();
  };

  return (
    <View style={styles.container}>
      <StatusBar style={effectiveTheme === "dark" ? "light" : "dark"} />
      <MapView
        onIncidentPress={handleIncidentPress}
        onVehiclePress={handleVehiclePress}
        onLongPress={handleMapLongPress}
      />

      {/* Notification Banner */}
      <NotificationBanner
        recommendedSpots={recommendedSpots}
        onDismiss={handleDismissSpot}
        onPress={handleSpotPress}
      />

      {/* Navigation Popup */}
      {navigationTarget && (
        <NavigationPopup
          latitude={navigationTarget.latitude}
          longitude={navigationTarget.longitude}
          onClose={() => setNavigationTarget(null)}
          theme={effectiveTheme}
        />
      )}

      {/* Tutorial Overlay */}
      <TutorialOverlay />

      {/* WebSocket Status Indicator - Disabled since mobile app doesn't need vehicle simulation */}
      {/* Removed - mobile app doesn't use vehicle simulation data */}

      {/* Incident Detail Modal */}
      <Modal
        visible={!!selectedIncident}
        transparent
        animationType="fade"
        onRequestClose={handleCloseIncident}
      >
        <View style={styles.modalOverlay}>
          <Animated.View
            style={{
              transform: [{ translateY: incidentSlideAnim }],
            }}
            className={`rounded-t-3xl p-6 ${
              effectiveTheme === "dark" ? "bg-neutral-800" : "bg-white"
            }`}
          >
            <View className="flex-row justify-between items-center mb-4">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50">
                Incident Details
              </Text>
              <TouchableOpacity onPress={handleCloseIncident}>
                <Text className="text-2xl text-neutral-600 dark:text-neutral-400">×</Text>
              </TouchableOpacity>
            </View>

            {selectedIncident && (
              <ScrollView>
                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Type</Text>
                  <Text className={`text-base font-semibold capitalize ${
                    selectedIncident.type === "collision" 
                      ? "text-red-600 dark:text-red-400" 
                      : selectedIncident.type === "incident"
                      ? "text-blue-600 dark:text-blue-400"
                      : "text-neutral-900 dark:text-neutral-50"
                  }`}>
                    {selectedIncident.type || "Unknown"}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Time Since</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {getTimeSince(selectedIncident.timestamp)}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Location</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {selectedIncident.position[1].toFixed(6)}, {selectedIncident.position[0].toFixed(6)}
                  </Text>
                </View>

                {selectedIncident.description && (
                  <View className="mb-4">
                    <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Description</Text>
                    <Text className="text-base text-neutral-900 dark:text-neutral-50">
                      {selectedIncident.description}
                    </Text>
                  </View>
                )}

                <View className="flex-row gap-3 mt-4">
                  <View className="flex-1">
                    <Button
                      className="dark:bg-white bg-neutral-800"
                      textClassName="dark:text-neutral-950 text-white"
                      title="Navigate"
                      onPress={() => selectedIncident && handleNavigateIncident(selectedIncident)}
                      variant="primary"
                      size="lg"
                    />
                  </View>
                  <View className="flex-1">
                    <Button
                      className="dark:bg-white bg-neutral-800"
                      textClassName="dark:text-neutral-950 text-white"
                      title="Close"
                      onPress={handleCloseIncident}
                      variant="primary"
                      size="lg"
                    />
                  </View>
                </View>
              </ScrollView>
            )}
          </Animated.View>
        </View>
      </Modal>

      {/* Vehicle Detail Modal */}
      <Modal
        visible={!!selectedVehicle}
        transparent
        animationType="fade"
        onRequestClose={handleCloseVehicle}
      >
        <View style={styles.modalOverlay}>
          <Animated.View
            style={{
              transform: [{ translateY: vehicleSlideAnim }],
            }}
            className={`rounded-t-3xl p-6 ${
              effectiveTheme === "dark" ? "bg-neutral-800" : "bg-white"
            }`}
          >
            <View className="flex-row justify-between items-center mb-4">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50">
                Vehicle Details
              </Text>
              <TouchableOpacity onPress={handleCloseVehicle}>
                <Text className="text-2xl text-neutral-600 dark:text-neutral-400">×</Text>
              </TouchableOpacity>
            </View>

            {selectedVehicle && (
              <ScrollView>
                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Vehicle ID</Text>
                  <Text className="text-base font-semibold text-neutral-900 dark:text-neutral-50">
                    {selectedVehicle["vehicle-id"]}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Type</Text>
                  <Text className="text-base font-semibold text-neutral-900 dark:text-neutral-50">
                    {selectedVehicle.type || "Unknown"}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Speed</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {selectedVehicle.speed ? `${selectedVehicle.speed} km/h` : "N/A"}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Heading</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {selectedVehicle.heading}°
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Location</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {selectedVehicle.lat.toFixed(6)}, {selectedVehicle.lon.toFixed(6)}
                  </Text>
                </View>

                <Button
                  className="dark:bg-white bg-neutral-800 mt-4"
                  textClassName="dark:text-neutral-950 text-white"
                  title="Close"
                  onPress={handleCloseVehicle}
                  variant="primary"
                  size="lg"
                />
              </ScrollView>
            )}
          </Animated.View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "flex-end",
  },
});



