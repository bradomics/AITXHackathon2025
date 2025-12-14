import { useState, useRef } from "react";
import { View, Text, ScrollView, TouchableOpacity, Modal, Platform, Linking, Animated } from "react-native";
import { StatusBar } from "expo-status-bar";
import { useApp } from "../../contexts/AppContext";
import { useDigitalTwin } from "../../contexts/DigitalTwinContext";
import { Button } from "../../components/ui/Button";
import { Hotspot } from "../../types";

export default function HotspotScreen() {
  const { effectiveTheme } = useApp();
  const { hotspots } = useDigitalTwin();
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);
  const hotspotSlideAnim = useRef(new Animated.Value(300)).current;

  const handleNavigate = (position: [number, number]) => {
    const latitude = position[1];
    const longitude = position[0];

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
    handleCloseHotspot();
  };

  const handleOpenHotspot = (hotspot: Hotspot) => {
    setSelectedHotspot(hotspot);
    // Animate content sliding up
    Animated.spring(hotspotSlideAnim, {
      toValue: 0,
      useNativeDriver: true,
      tension: 65,
      friction: 11,
    }).start();
  };

  const handleCloseHotspot = () => {
    Animated.timing(hotspotSlideAnim, {
      toValue: 300,
      duration: 250,
      useNativeDriver: true,
    }).start(() => {
      setSelectedHotspot(null);
      hotspotSlideAnim.setValue(300); // Reset for next open
    });
  };

  const formatResponseTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (remainingSeconds === 0) {
      return `${minutes} min`;
    }
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getResponseTimeColor = (responseTime: number) => {
    if (responseTime <= 120) return "text-green-600 dark:text-green-400"; // <= 2 min
    if (responseTime <= 240) return "text-yellow-600 dark:text-yellow-400"; // <= 4 min
    return "text-red-600 dark:text-red-400"; // > 4 min
  };

  return (
    <View className="flex-1 bg-white dark:bg-neutral-900">
      <StatusBar style={effectiveTheme === "dark" ? "light" : "dark"} />
      {/* Fixed Header */}
      <View className="px-4 pt-16 pb-4 bg-white dark:bg-neutral-900">
        <Text className="text-3xl font-bold text-neutral-900 dark:text-neutral-50 mb-2">
          Hotspots
        </Text>
        <Text className="text-base text-neutral-600 dark:text-neutral-400">
          {hotspots.length} hotspot{hotspots.length !== 1 ? "s" : ""} for optimal response time
        </Text>
      </View>
      
      {/* Scrollable Content */}
      <ScrollView className="flex-1 px-4">
        {hotspots.length === 0 ? (
          <View className="flex-1 items-center justify-center py-12">
            <Text className="text-lg text-neutral-600 dark:text-neutral-400">
              No hotspots identified
            </Text>
          </View>
        ) : (
          <View className="gap-3">
            {hotspots.map((hotspot) => (
              <TouchableOpacity
                key={hotspot.id}
                onPress={() => handleOpenHotspot(hotspot)}
                className="bg-white dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700"
              >
                <View className="flex-row justify-between items-start mb-2">
                  <Text className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                    Hotspot #{hotspot.priority || hotspot.id}
                  </Text>
                  <Text className={`text-sm font-semibold ${getResponseTimeColor(hotspot.responseTime)}`}>
                    {formatResponseTime(hotspot.responseTime)}
                  </Text>
                </View>
                <Text className="text-xs text-neutral-500 dark:text-neutral-500">
                  {hotspot.position[1].toFixed(4)}, {hotspot.position[0].toFixed(4)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </ScrollView>

      {/* Hotspot Detail Modal */}
      <Modal
        visible={!!selectedHotspot}
        transparent
        animationType="fade"
        onRequestClose={handleCloseHotspot}
      >
        <View className="flex-1 bg-black/50 justify-end">
          <Animated.View
            style={{
              transform: [{ translateY: hotspotSlideAnim }],
            }}
            className={`rounded-t-3xl p-6 ${
              effectiveTheme === "dark" ? "bg-neutral-800" : "bg-white"
            }`}
          >
            <View className="flex-row justify-between items-center mb-4">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50">
                Hotspot Details
              </Text>
              <TouchableOpacity onPress={handleCloseHotspot}>
                <Text className="text-2xl text-neutral-600 dark:text-neutral-400">×</Text>
              </TouchableOpacity>
            </View>

            {selectedHotspot && (
              <View>
                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Response Time</Text>
                  <Text className={`text-2xl font-bold ${getResponseTimeColor(selectedHotspot.responseTime)}`}>
                    {formatResponseTime(selectedHotspot.responseTime)}
                  </Text>
                </View>

                <View className="mb-4">
                  <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Location</Text>
                  <Text className="text-base text-neutral-900 dark:text-neutral-50">
                    {selectedHotspot.position[1].toFixed(6)}, {selectedHotspot.position[0].toFixed(6)}
                  </Text>
                </View>

                {selectedHotspot.priority && (
                  <View className="mb-4">
                    <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Priority</Text>
                    <Text className="text-base text-neutral-900 dark:text-neutral-50">
                      #{selectedHotspot.priority}
                    </Text>
                  </View>
                )}

                <View className="flex-row gap-3 mt-4">
                  <View className="flex-1">
                    <Button
                      className="dark:bg-white bg-neutral-800"
                      textClassName="dark:text-neutral-950 text-white"
                      title="Navigate"
                      onPress={() => selectedHotspot && handleNavigate(selectedHotspot.position)}
                      variant="primary"
                      size="lg"
                    />
                  </View>
                  <View className="flex-1">
                    <Button
                      className="dark:bg-white bg-neutral-800"
                      textClassName="dark:text-neutral-950 text-white"
                      title="Close"
                      onPress={handleCloseHotspot}
                      variant="primary"
                      size="lg"
                    />
                  </View>
                </View>
              </View>
            )}
          </Animated.View>
        </View>
      </Modal>
    </View>
  );
}

