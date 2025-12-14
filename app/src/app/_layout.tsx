import "../global.css";

import { Slot } from "expo-router";
import { AppProvider } from "../contexts/AppContext";
import { DigitalTwinProvider } from "../contexts/DigitalTwinContext";

export default function Layout() {
  return (
    <AppProvider>
      <DigitalTwinProvider>
        <Slot />
      </DigitalTwinProvider>
    </AppProvider>
  );
}
