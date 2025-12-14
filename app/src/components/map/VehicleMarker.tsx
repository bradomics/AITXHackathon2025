import { View } from "react-native";
import { Vehicle } from "../../types";

interface VehicleMarkerProps {
  vehicle: Vehicle;
}

export function VehicleMarker({ vehicle }: VehicleMarkerProps) {
  const getVehicleColor = () => {
    const type = vehicle.type?.toLowerCase() || "";
    if (type.includes("ambulance") || type.includes("ems")) return "bg-red-500";
    if (type.includes("police")) return "bg-blue-500";
    if (type.includes("fire")) return "bg-orange-500";
    return "bg-neutral-500";
  };

  return (
    <View
      className={`w-4 h-4 rounded-full ${getVehicleColor()} border border-white dark:border-neutral-900`}
      style={{
        transform: [{ rotate: `${vehicle.heading}deg` }],
      }}
    />
  );
}



