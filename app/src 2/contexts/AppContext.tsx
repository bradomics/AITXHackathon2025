import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { UserRole } from "../types";
import { storage } from "../utils/storage";
import { useColorScheme } from "react-native";

interface AppContextType {
  role: UserRole | null;
  hasCompletedOnboarding: boolean;
  hasCompletedTutorial: boolean;
  setRole: (role: UserRole | null) => Promise<void>;
  setHasCompletedOnboarding: (completed: boolean) => Promise<void>;
  setHasCompletedTutorial: (completed: boolean) => Promise<void>;
  effectiveTheme: "light" | "dark"; // Always uses system theme
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const systemColorScheme = useColorScheme();
  const [role, setRoleState] = useState<UserRole | null>(null);
  const [hasCompletedOnboarding, setHasCompletedOnboardingState] = useState(false);
  const [hasCompletedTutorial, setHasCompletedTutorialState] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load persisted data on mount
  useEffect(() => {
    const loadPersistedData = async () => {
      try {
        const [savedRole, completed, tutorialCompleted] = await Promise.all([
          storage.getUserRole(),
          storage.getHasCompletedOnboarding(),
          storage.getHasCompletedTutorial(),
        ]);

        setRoleState(savedRole);
        setHasCompletedOnboardingState(completed);
        setHasCompletedTutorialState(tutorialCompleted);
      } catch (error) {
        console.error("Error loading persisted data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadPersistedData();
  }, []);

  // Always use system theme
  const effectiveTheme: "light" | "dark" = systemColorScheme === "dark" ? "dark" : "light";

  const setRole = async (newRole: UserRole | null) => {
    setRoleState(newRole);
    if (newRole) {
      await storage.setUserRole(newRole);
    } else {
      await storage.clearUserRole();
    }
  };

  const setHasCompletedOnboarding = async (completed: boolean) => {
    setHasCompletedOnboardingState(completed);
    await storage.setHasCompletedOnboarding(completed);
  };

  const setHasCompletedTutorial = async (completed: boolean) => {
    setHasCompletedTutorialState(completed);
    await storage.setHasCompletedTutorial(completed);
  };

  // Don't render children until we've loaded persisted data
  if (isLoading) {
    return null; // Or a loading screen
  }

  return (
    <AppContext.Provider
      value={{
        role,
        hasCompletedOnboarding,
        hasCompletedTutorial,
        setRole,
        setHasCompletedOnboarding,
        setHasCompletedTutorial,
        effectiveTheme,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
}



