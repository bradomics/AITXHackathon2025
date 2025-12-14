import { useEffect, useRef } from "react";
import * as Notifications from "expo-notifications";
import { RecommendedSpot, IncidentPoint, Hotspot } from "../types";

// Configure notification handler
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

export function useNotifications() {
  const notificationListener = useRef<Notifications.Subscription | null>(null);
  const responseListener = useRef<Notifications.Subscription | null>(null);

  useEffect(() => {
    // Request permissions for push notifications
    const requestPermissions = async () => {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;

      if (existingStatus !== "granted") {
        const { status } = await Notifications.requestPermissionsAsync({
          ios: {
            allowAlert: true,
            allowBadge: true,
            allowSound: true,
            allowAnnouncements: true,
          },
        });
        finalStatus = status;
      }

      if (finalStatus !== "granted") {
        console.warn("Notification permissions not granted - push notifications will not work");
        return;
      }
    };

    requestPermissions();

    // Listen for notifications received while app is foregrounded
    notificationListener.current = Notifications.addNotificationReceivedListener((notification) => {
      console.log("Notification received:", notification);
    });

    // Listen for user tapping on notifications
    responseListener.current = Notifications.addNotificationResponseReceivedListener((response) => {
      console.log("Notification response:", response);
    });

    return () => {
      if (notificationListener.current) {
        notificationListener.current.remove();
      }
      if (responseListener.current) {
        responseListener.current.remove();
      }
    };
  }, []);

  const scheduleNotification = async (title: string, body: string, data?: any) => {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: data?.sound || true,
        priority: data?.priority || "default",
        // Ensure notification shows even when app is in background
        badge: 1,
      },
      trigger: null, // Show immediately (works as push notification)
    });
  };

  const notifyRecommendedSpot = async (spot: RecommendedSpot) => {
    await scheduleNotification(
      "Recommended Location",
      `${spot.name} is nearby (${spot.distance ? `${Math.round(spot.distance)}m` : "nearby"})`,
      { type: "recommended-spot", spotId: spot.id }
    );
  };

  const notifyNewIncident = async (incident: IncidentPoint) => {
    const typeLabel = incident.type === "collision" ? "🚨 Collision" : "⚠️ Traffic Incident";
    const description = incident.description || "New traffic incident reported";
    
    // Send push notification (works even when app is in background)
    await scheduleNotification(
      `New ${typeLabel}`,
      description,
      { 
        type: "incident", 
        incidentId: incident.id,
        // Add sound and priority for push notifications
        sound: "default",
        priority: "high",
      }
    );
  };

  const notifyNewIncidentsBatch = async (count: number, incidents: IncidentPoint[]) => {
    // Count collisions vs other incidents
    const collisions = incidents.filter(inc => inc.type === "collision").length;
    const otherIncidents = count - collisions;
    
    let title = "";
    let body = "";
    
    if (count === 1) {
      // Single incident - use detailed notification
      const incident = incidents[0];
      const typeLabel = incident.type === "collision" ? "🚨 Collision" : "⚠️ Traffic Incident";
      title = `New ${typeLabel}`;
      body = incident.description || "New traffic incident reported";
    } else {
      // Multiple incidents - batch notification
      title = `🚨 ${count} New Incident${count !== 1 ? "s" : ""}`;
      if (collisions > 0 && otherIncidents > 0) {
        body = `${collisions} collision${collisions !== 1 ? "s" : ""} and ${otherIncidents} other incident${otherIncidents !== 1 ? "s" : ""} reported`;
      } else if (collisions > 0) {
        body = `${collisions} new collision${collisions !== 1 ? "s" : ""} reported`;
      } else {
        body = `${otherIncidents} new traffic incident${otherIncidents !== 1 ? "s" : ""} reported`;
      }
    }
    
    // Send batched push notification
    await scheduleNotification(
      title,
      body,
      { 
        type: "incidents-batch", 
        count,
        incidentIds: incidents.map(inc => inc.id).filter(Boolean),
        sound: "default",
        priority: "high",
      }
    );
  };

  const notifyNewHotspot = async (hotspot: Hotspot) => {
    await scheduleNotification(
      "New High-Risk Area",
      `High-risk area detected (risk: ${(hotspot.risk * 100).toFixed(0)}%)`,
      { type: "hotspot", hotspotId: hotspot.id }
    );
  };

  return {
    scheduleNotification,
    notifyRecommendedSpot,
    notifyNewIncident,
    notifyNewIncidentsBatch,
    notifyNewHotspot,
  };
}



