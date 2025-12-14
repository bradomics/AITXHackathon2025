import { View, Text, ScrollView, KeyboardAvoidingView, Platform, TouchableOpacity } from "react-native";
import { StatusBar } from "expo-status-bar";
import { useRouter } from "expo-router";
import { useApp } from "../../contexts/AppContext";
import { RoleSelector } from "../../components/onboarding/RoleSelector";
import { UserRole } from "../../types";

export default function SettingsScreen() {
  const router = useRouter();
  const { role, setRole, setHasCompletedOnboarding, setHasCompletedTutorial, effectiveTheme } = useApp();

  const handleRoleChange = async (newRole: UserRole) => {
    await setRole(newRole);
  };

  const handleResetOnboarding = async () => {
    await Promise.all([
      setHasCompletedOnboarding(false),
      setHasCompletedTutorial(false),
      setRole(null), // Clear role so onboarding screen shows
    ]);
    // Navigate to onboarding screen
    router.replace("/");
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      className="flex-1 bg-white dark:bg-neutral-900"
    >
      <StatusBar style={effectiveTheme === "dark" ? "light" : "dark"} />
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-6 pt-16"
        showsVerticalScrollIndicator={false}
      >
        <View className="mb-8">
          <Text className="text-6xl text-center font-extrabold font-monospace text-neutral-900 dark:text-neutral-50 mb-2">
            Settings
          </Text>
          <Text className="text-lg text-center text-neutral-600 dark:text-neutral-400">
            Manage your app preferences
          </Text>
        </View>

        <View className="mb-8">
          <RoleSelector selectedRole={role} onSelectRole={handleRoleChange} />
        </View>
      </ScrollView>
      
      {/* Reset Onboarding Button - Bottom Left */}
      <TouchableOpacity
        onPress={handleResetOnboarding}
        className="absolute bottom-4 left-4"
      >
        <Text className="text-xs text-neutral-500 dark:text-neutral-400 underline">
          Reset Onboarding
        </Text>
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}



