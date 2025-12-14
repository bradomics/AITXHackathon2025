import { useState, useEffect, useRef } from "react";
import { View, Modal, Text, ScrollView, TouchableOpacity, StyleSheet, Animated, Platform, Linking } from "react-native";
import { StatusBar } from "expo-status-bar";
import { useApp } from "../../contexts/AppContext";
import { useWebSocket } from "../../hooks/useWebSocket";
import { useNotifications } from "../../hooks/useNotifications";
import { useDigitalTwin } from "../../contexts/DigitalTwinContext";
import { MapViewComponent as MapView } from "../../components/map/MapView";
import { NotificationBanner } from "../../components/notifications/NotificationBanner";
import { NavigationPopup } from "../../components/map/NavigationPopup";
import { TutorialOverlay } from "../../components/tutorial/TutorialOverlay";
import { IncidentPoint, Vehicle, RecommendedSpot } from "../../types";
import { Button } from "../../components/ui/Button";

export default function MapScreen() {
  const { role, effectiveTheme } = useApp();
  const { incidentPoints } = useDigitalTwin();
  const [selectedIncident, setSelectedIncident] = useState<IncidentPoint | null>(null);
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);
  const [recommendedSpots, setRecommendedSpots] = useState<RecommendedSpot[]>([]);
  const [navigationTarget, setNavigationTarget] = useState<{ latitude: number; longitude: number } | null>(null);
  const incidentSlideAnim = useRef(new Animated.Value(300)).current;
  const vehicleSlideAnim = useRef(new Animated.Value(300)).current;

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

  // Connect to WebSocket
  const { status: wsStatus, isConnected } = useWebSocket({
    enabled: !!role, // Only connect when role is set
  });

  // Set up notifications
  const { notifyRecommendedSpot } = useNotifications();

  // Simulate recommended spots (in real app, this would come from the server)
  // Only run once when connected, not on every render
  const hasInitializedSpots = useRef(false);
  
  useEffect(() => {
    if (role && isConnected && !hasInitializedSpots.current) {
      hasInitializedSpots.current = true;
      
      // Example: Generate some recommended spots based on role
      const spots: RecommendedSpot[] = [
        {
          id: "spot1",
          position: [-97.7431, 30.2672],
          type: role === "ems" ? "hospital" : role === "roadside" ? "service-station" : "police-station",
          name: role === "ems" ? "Austin General Hospital" : role === "roadside" ? "Quick Service" : "Central Police",
          distance: 500,
        },
      ];
      setRecommendedSpots(spots);

      // Notify about recommended spots
      spots.forEach((spot) => {
        notifyRecommendedSpot(spot);
      });
    }
    
    // Reset when disconnected
    if (!isConnected) {
      hasInitializedSpots.current = false;
    }
  }, [role, isConnected]); // Remove notifyRecommendedSpot from deps

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

      {/* WebSocket Status Indicator */}
      {!isConnected && (
        <View className="absolute bottom-20 left-4 right-4 bg-yellow-500 dark:bg-yellow-600 rounded-lg p-3">
          <Text className="text-white font-semibold text-sm">
            {wsStatus === "connecting" ? "Connecting..." : "Disconnected from server"}
          </Text>
        </View>
      )}

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



