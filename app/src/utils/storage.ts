import AsyncStorage from "@react-native-async-storage/async-storage";
import { UserRole, ThemeMode } from "../types";

const STORAGE_KEYS = {
  HAS_COMPLETED_ONBOARDING: "@app:hasCompletedOnboarding",
  HAS_COMPLETED_TUTORIAL: "@app:hasCompletedTutorial",
  USER_ROLE: "@app:userRole",
  THEME_MODE: "@app:themeMode",
} as const;

export const storage = {
  // Onboarding
  async getHasCompletedOnboarding(): Promise<boolean> {
    try {
      const value = await AsyncStorage.getItem(STORAGE_KEYS.HAS_COMPLETED_ONBOARDING);
      return value === "true";
    } catch (error) {
      console.error("Error reading onboarding status:", error);
      return false;
    }
  },

  async setHasCompletedOnboarding(value: boolean): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.HAS_COMPLETED_ONBOARDING, String(value));
    } catch (error) {
      console.error("Error saving onboarding status:", error);
    }
  },

  // Tutorial
  async getHasCompletedTutorial(): Promise<boolean> {
    try {
      const value = await AsyncStorage.getItem(STORAGE_KEYS.HAS_COMPLETED_TUTORIAL);
      return value === "true";
    } catch (error) {
      console.error("Error reading tutorial status:", error);
      return false;
    }
  },

  async setHasCompletedTutorial(value: boolean): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.HAS_COMPLETED_TUTORIAL, String(value));
    } catch (error) {
      console.error("Error saving tutorial status:", error);
    }
  },

  // User Role
  async getUserRole(): Promise<UserRole | null> {
    try {
      const value = await AsyncStorage.getItem(STORAGE_KEYS.USER_ROLE);
      return value as UserRole | null;
    } catch (error) {
      console.error("Error reading user role:", error);
      return null;
    }
  },

  async setUserRole(role: UserRole): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.USER_ROLE, role);
    } catch (error) {
      console.error("Error saving user role:", error);
    }
  },

  async clearUserRole(): Promise<void> {
    try {
      await AsyncStorage.removeItem(STORAGE_KEYS.USER_ROLE);
    } catch (error) {
      console.error("Error clearing user role:", error);
    }
  },

  // Theme Mode
  async getThemeMode(): Promise<ThemeMode> {
    try {
      const value = await AsyncStorage.getItem(STORAGE_KEYS.THEME_MODE);
      return (value as ThemeMode) || "system";
    } catch (error) {
      console.error("Error reading theme mode:", error);
      return "system";
    }
  },

  async setThemeMode(mode: ThemeMode): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.THEME_MODE, mode);
    } catch (error) {
      console.error("Error saving theme mode:", error);
    }
  },

  // Clear all app data
  async clearAll(): Promise<void> {
    try {
      await AsyncStorage.multiRemove(Object.values(STORAGE_KEYS));
    } catch (error) {
      console.error("Error clearing storage:", error);
    }
  },
};



