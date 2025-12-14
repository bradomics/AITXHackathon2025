import { View } from "react-native";
import { IncidentPoint } from "../../types";
import { WarningIcon } from "./icons/WarningIcon";
import { ExclamationIcon } from "./icons/ExclamationIcon";

interface IncidentMarkerProps {
  incident: IncidentPoint;
}

export function IncidentMarker({ incident }: IncidentMarkerProps) {
  const isCollision = incident.type === "collision";
  const iconColor = isCollision ? "#EF4444" : "#3B82F6"; // Red for collision, blue for incident

  return (
    <View className="items-center justify-center">
      {isCollision ? (
        <WarningIcon size={32} color={iconColor} />
      ) : (
        <ExclamationIcon size={32} color={iconColor} />
      )}
    </View>
  );
}



