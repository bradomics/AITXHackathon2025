import { View, Text, TouchableOpacity, Platform, Linking } from "react-native";

interface NavigationPopupProps {
  latitude: number;
  longitude: number;
  onClose: () => void;
  theme: "light" | "dark";
}

export function NavigationPopup({ latitude, longitude, onClose, theme }: NavigationPopupProps) {
  const openAppleMaps = () => {
    const url = `http://maps.apple.com/?daddr=${latitude},${longitude}&dirflg=d`;
    Linking.openURL(url).catch((err) => console.error("Error opening Apple Maps:", err));
    onClose();
  };

  const openGoogleMaps = () => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${latitude},${longitude}`;
    Linking.openURL(url).catch((err) => console.error("Error opening Google Maps:", err));
    onClose();
  };

  const openWaze = () => {
    const url = `waze://?ll=${latitude},${longitude}&navigate=yes`;
    Linking.canOpenURL(url).then((supported) => {
      if (supported) {
        Linking.openURL(url).catch((err) => console.error("Error opening Waze:", err));
      } else {
        // Fallback to App Store or web
        Linking.openURL("https://waze.com/ul?ll=" + latitude + "," + longitude + "&navigate=yes").catch(
          (err) => console.error("Error opening Waze web:", err)
        );
      }
      onClose();
    });
  };

  const copyCoordinates = () => {
    // For React Native, we'd need expo-clipboard or similar
    const coords = `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
    console.log("Coordinates:", coords);
    // TODO: Add clipboard functionality if needed
    onClose();
  };

  return (
    <View className="absolute bottom-[100px] left-4 right-4 rounded-2xl p-4 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 shadow-lg z-[1000]">
      <View className="flex-row justify-between items-center mb-3">
        <Text className="text-lg font-bold text-neutral-900 dark:text-neutral-50">Navigate To</Text>
        <TouchableOpacity onPress={onClose} className="w-8 h-8 justify-center items-center">
          <Text className="text-2xl leading-6 text-neutral-900 dark:text-neutral-50">×</Text>
        </TouchableOpacity>
      </View>

      <View className="mb-4 py-2 px-3 bg-neutral-100 dark:bg-neutral-700/50 rounded-lg">
        <Text className={`text-sm ${Platform.OS === "ios" ? "font-mono" : ""} text-neutral-900 dark:text-neutral-50`}>
          {latitude.toFixed(6)}, {longitude.toFixed(6)}
        </Text>
      </View>

      <View className="gap-2">
        {Platform.OS === "ios" ? (
          <TouchableOpacity
            className="py-3 px-4 rounded-lg dark:bg-white bg-neutral-800 active:opacity-80 items-center"
            onPress={openAppleMaps}
          >
            <Text className="dark:text-neutral-950 text-white text-base font-semibold">Open in Apple Maps</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            className="py-3 px-4 rounded-lg dark:bg-white bg-neutral-800 active:opacity-80 items-center"
            onPress={openGoogleMaps}
          >
            <Text className="dark:text-neutral-950 text-white text-base font-semibold">Open in Google Maps</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          className="py-3 px-4 rounded-lg dark:bg-white bg-neutral-800 active:opacity-80 items-center"
          onPress={openWaze}
        >
          <Text className="dark:text-neutral-950 text-white text-base font-semibold">Open in Waze</Text>
        </TouchableOpacity>

        <TouchableOpacity
          className="py-3 px-4 rounded-lg dark:bg-white bg-neutral-800 active:opacity-80 items-center"
          onPress={copyCoordinates}
        >
          <Text className="dark:text-neutral-950 text-white text-base font-semibold">Copy Coordinates</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

