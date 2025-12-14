import { Tabs } from "expo-router";
import { useApp } from "../../contexts/AppContext";
import { MapIcon } from "../../components/navigation/MapIcon";
import { IncidentsIcon } from "../../components/navigation/IncidentsIcon";
import { RiskIcon } from "../../components/navigation/RiskIcon";
import { SettingsIcon } from "../../components/navigation/SettingsIcon";

export default function TabsLayout() {
  const { effectiveTheme } = useApp();

  // Use white for active, dark neutral for inactive
  const activeColor = "#FFFFFF";
  const inactiveColor = effectiveTheme === "dark" ? "#404040" : "#737373";

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: activeColor,
        tabBarInactiveTintColor: inactiveColor,
        tabBarStyle: {
          backgroundColor: effectiveTheme === "dark" ? "#262626" : "#ffffff",
          borderTopColor: effectiveTheme === "dark" ? "#404040" : "#d4d4d4",
        },
      }}
    >
      <Tabs.Screen
        name="map"
        options={{
          title: "Map",
          tabBarIcon: ({ focused, color }) => (
            <MapIcon size={24} color={focused ? activeColor : inactiveColor} />
          ),
        }}
      />
      <Tabs.Screen
        name="incidents"
        options={{
          title: "Incidents",
          tabBarIcon: ({ focused, color }) => (
            <IncidentsIcon size={24} color={focused ? activeColor : inactiveColor} />
          ),
        }}
      />
      <Tabs.Screen
        name="risk"
        options={{
          title: "Hotspots",
          tabBarIcon: ({ focused, color }) => (
            <RiskIcon size={24} color={focused ? activeColor : inactiveColor} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: "Settings",
          tabBarIcon: ({ focused, color }) => (
            <SettingsIcon size={24} color={focused ? activeColor : inactiveColor} />
          ),
        }}
      />
    </Tabs>
  );
}

