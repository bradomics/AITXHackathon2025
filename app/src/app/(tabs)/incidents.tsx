import { useState, useRef } from "react";
import { View, Text, ScrollView, TouchableOpacity, Modal, Platform, Linking, Animated } from "react-native";
import { StatusBar } from "expo-status-bar";
import { useApp } from "../../contexts/AppContext";
import { useDigitalTwin } from "../../contexts/DigitalTwinContext";
import { IncidentPoint } from "../../types";
import { Button } from "../../components/ui/Button";
import { IncidentCard } from "../../components/incidents/IncidentCard";

export default function IncidentsScreen() {
  const { effectiveTheme } = useApp();
  const { incidentPoints } = useDigitalTwin();
  const [selectedIncident, setSelectedIncident] = useState<IncidentPoint | null>(null);
  const incidentSlideAnim = useRef(new Animated.Value(300)).current;

  const handleNavigate = (incident: IncidentPoint) => {
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

  const handleOpenIncident = (incident: IncidentPoint) => {
    setSelectedIncident(incident);
    // Animate content sliding up
    Animated.spring(incidentSlideAnim, {
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

  const getTypeColor = (type?: string) => {
    switch (type?.toLowerCase()) {
      case "collision":
        return "text-red-600 dark:text-red-400";
      case "incident":
        return "text-blue-600 dark:text-blue-400";
      default:
        return "text-neutral-600 dark:text-neutral-400";
    }
  };

  return (
    <View className="flex-1 bg-white dark:bg-neutral-900">
      <StatusBar style={effectiveTheme === "dark" ? "light" : "dark"} />
      {/* Fixed Header */}
      <View className="px-4 pt-16 pb-4 bg-white dark:bg-neutral-900">
        <Text className="text-3xl font-bold text-neutral-900 dark:text-neutral-50 mb-2">
          Incidents
        </Text>
        <Text className="text-base text-neutral-600 dark:text-neutral-400">
          {incidentPoints.length} active incident{incidentPoints.length !== 1 ? "s" : ""}
        </Text>
      </View>
      
      {/* Scrollable Content */}
      <ScrollView className="flex-1 px-4">
        {incidentPoints.length === 0 ? (
          <View className="flex-1 items-center justify-center py-12">
            <Text className="text-lg text-neutral-600 dark:text-neutral-400">
              No incidents reported
            </Text>
          </View>
        ) : (
          <View className="gap-3">
            {incidentPoints.map((incident, index) => (
              <IncidentCard
                key={index}
                incident={incident}
                onPress={handleOpenIncident}
                getTimeSince={getTimeSince}
              />
            ))}
          </View>
        )}
      </ScrollView>

      {/* Incident Detail Modal */}
      <Modal
        visible={!!selectedIncident}
        transparent
        animationType="fade"
        onRequestClose={handleCloseIncident}
      >
        <View className="flex-1 bg-black/50 justify-end">
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
                  <Text className={`text-base font-semibold capitalize ${getTypeColor(selectedIncident.type)}`}>
                    {selectedIncident.type || "Unknown"}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Time Since</Text>
                  <Text className="text-base font-semibold text-neutral-900 dark:text-neutral-50">
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
                      onPress={() => selectedIncident && handleNavigate(selectedIncident)}
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
    </View>
  );
}

