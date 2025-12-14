import { View, Text, TouchableOpacity } from "react-native";
import { IncidentPoint } from "../../types";

interface IncidentCardProps {
  incident: IncidentPoint;
  onPress: (incident: IncidentPoint) => void;
  getTimeSince: (timestamp?: number) => string;
}

export function IncidentCard({ incident, onPress, getTimeSince }: IncidentCardProps) {
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
    <TouchableOpacity
      onPress={() => onPress(incident)}
      className="bg-white dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700"
    >
      <View className="flex-row justify-between items-start mb-2">
        <Text className="text-lg font-semibold text-neutral-900 dark:text-neutral-50 flex-1 capitalize">
          {incident.type || "Incident"}
        </Text>
        {incident.type && (
          <Text className={`text-sm font-semibold capitalize ${getTypeColor(incident.type)}`}>
            {incident.type}
          </Text>
        )}
      </View>
      {incident.description && (
        <Text className="text-sm text-neutral-600 dark:text-neutral-400 mb-2" numberOfLines={2}>
          {incident.description}
        </Text>
      )}
      <View className="flex-row justify-between items-center">
        <Text className="text-xs text-neutral-500 dark:text-neutral-500">
          {incident.position[1].toFixed(4)}, {incident.position[0].toFixed(4)}
        </Text>
        <Text className="text-xs text-neutral-500 dark:text-neutral-500">
          {getTimeSince(incident.timestamp)}
        </Text>
      </View>
    </TouchableOpacity>
  );
}

