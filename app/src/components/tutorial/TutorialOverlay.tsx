import { useState } from "react";
import { View, Text, TouchableOpacity, Modal } from "react-native";
import { useApp } from "../../contexts/AppContext";

type TutorialStep = "longPress" | "clickPoint" | "complete";

interface TutorialOverlayProps {}

export function TutorialOverlay({}: TutorialOverlayProps) {
  const { hasCompletedTutorial, setHasCompletedTutorial, setHasCompletedOnboarding } = useApp();
  const [currentStep, setCurrentStep] = useState<TutorialStep>("longPress");

  // If tutorial already completed, don't show
  if (hasCompletedTutorial) {
    return null;
  }

  const handleNext = () => {
    if (currentStep === "longPress") {
      setCurrentStep("clickPoint");
    } else if (currentStep === "clickPoint") {
      setCurrentStep("complete");
    }
  };

  const handleCompleteTutorial = async () => {
    await Promise.all([
      setHasCompletedTutorial(true),
      setHasCompletedOnboarding(true),
    ]);
  };

  return (
    <Modal
      visible={!hasCompletedTutorial}
      transparent
      animationType="fade"
      statusBarTranslucent
    >
      <View className="flex-1 bg-black/60">
        {currentStep === "longPress" && (
          <View className="flex-1 justify-center items-center px-6">
            <View className="bg-white dark:bg-neutral-800 rounded-2xl p-6 max-w-sm">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50 mb-4 text-center">
                Step 1: Navigation
              </Text>
              <Text className="text-base text-neutral-600 dark:text-neutral-400 mb-6 text-center">
                Press and hold anywhere on the map to open navigation options. You can choose to navigate with Apple Maps, Google Maps, or Waze.
              </Text>
              <TouchableOpacity
                onPress={handleNext}
                className="bg-neutral-800 dark:bg-white rounded-lg py-3 px-6"
              >
                <Text className="text-white dark:text-neutral-950 text-center font-semibold">
                  Next
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {currentStep === "clickPoint" && (
          <View className="flex-1 justify-center items-center px-6">
            <View className="bg-white dark:bg-neutral-800 rounded-2xl p-6 max-w-sm">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50 mb-4 text-center">
                Step 2: View Details
              </Text>
              <Text className="text-base text-neutral-600 dark:text-neutral-400 mb-6 text-center">
                Tap on any vehicle or incident marker on the map to see detailed information including location, type, and other relevant data.
              </Text>
              <TouchableOpacity
                onPress={handleNext}
                className="bg-neutral-800 dark:bg-white rounded-lg py-3 px-6"
              >
                <Text className="text-white dark:text-neutral-950 text-center font-semibold">
                  Next
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {currentStep === "complete" && (
          <View className="flex-1 justify-center items-center px-6">
            <View className="bg-white dark:bg-neutral-800 rounded-2xl p-6 max-w-sm">
              <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50 mb-4 text-center">
                You're all set!
              </Text>
              <Text className="text-base text-neutral-600 dark:text-neutral-400 mb-6 text-center">
                You now know how to navigate and view details on the map
              </Text>
              <TouchableOpacity
                onPress={handleCompleteTutorial}
                className="bg-neutral-800 dark:bg-white rounded-lg py-3 px-6"
              >
                <Text className="text-white dark:text-neutral-950 text-center font-semibold">
                  Get Started
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </Modal>
  );
}
