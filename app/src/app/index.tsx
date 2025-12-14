import { useEffect, useState } from "react";
import { View, Text, ScrollView, KeyboardAvoidingView, Platform } from "react-native";
import { useRouter } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useApp } from "../contexts/AppContext";
import { RoleSelector } from "../components/onboarding/RoleSelector";
import { Button } from "../components/ui/Button";
import { UserRole } from "../types";

export default function OnboardingScreen() {
  const router = useRouter();
  const { hasCompletedOnboarding, setRole, setHasCompletedOnboarding, role, effectiveTheme } =
    useApp();
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(role);

  useEffect(() => {
    // If already completed onboarding, redirect to map
    if (hasCompletedOnboarding && role) {
      router.replace("/(tabs)/map");
    }
    // If role is set but onboarding not complete, go to map for tutorial
    if (role && !hasCompletedOnboarding) {
      router.replace("/(tabs)/map");
    }
  }, [hasCompletedOnboarding, role, router]);

  const handleComplete = async () => {
    if (!selectedRole) {
      // Show error or prevent completion
      return;
    }

    // Only set role, don't mark onboarding as complete yet
    // Tutorial will complete onboarding
    await setRole(selectedRole);

    router.replace("/(tabs)/map");
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      className="flex-1 bg-white dark:bg-neutral-900"
    >
      <StatusBar style={effectiveTheme === "dark" ? "light" : "dark"} />
      <ScrollView
        className="flex-1"
        contentContainerClassName="flex-1 justify-center px-6"
        showsVerticalScrollIndicator={false}
      >
        <View className="mb-8">
          <Text className="text-6xl text-center font-extrabold font-monospace text-neutral-900 dark:text-neutral-50 mb-2">
            Welcome
          </Text>
          <Text className="text-lg text-center text-neutral-600 dark:text-neutral-400">
            Let's set up your app to get started
          </Text>
        </View>

        <View className="mb-8">
          <RoleSelector selectedRole={selectedRole} onSelectRole={setSelectedRole} />
        </View>

        <View>
          <Button
            className="dark:bg-white bg-neutral-800"  
            textClassName="dark:text-neutral-950 text-white"
            title="Continue"
            onPress={handleComplete}
            disabled={!selectedRole}
            variant="primary"
            size="lg"
          />
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
