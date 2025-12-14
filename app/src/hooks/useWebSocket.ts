import { useEffect, useRef, useState, useCallback } from "react";
import { useDigitalTwin } from "../contexts/DigitalTwinContext";
import { Vehicle, RiskPoint, IncidentPoint } from "../types";

type WebSocketStatus = "disconnected" | "connecting" | "connected" | "error";

interface UseWebSocketOptions {
  url?: string;
  enabled?: boolean;
  onMessage?: (data: any) => void;
  onError?: (error: Event) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = process.env.EXPO_PUBLIC_WS_URL || "ws://localhost:8765",
    enabled = true,
    onMessage: externalOnMessage,
    onError: externalOnError,
    onConnect: externalOnConnect,
    onDisconnect: externalOnDisconnect,
  } = options;

  const { setVehicles, setCollisionPoints, setIncidentPoints } = useDigitalTwin();
  const [status, setStatus] = useState<WebSocketStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds

  // Store context setters in refs to avoid dependency issues
  const settersRef = useRef({ setVehicles, setCollisionPoints, setIncidentPoints });
  settersRef.current = { setVehicles, setCollisionPoints, setIncidentPoints };

  // Store callbacks in refs to avoid dependency issues
  const callbacksRef = useRef({
    onMessage: externalOnMessage,
    onError: externalOnError,
    onConnect: externalOnConnect,
    onDisconnect: externalOnDisconnect,
  });
  callbacksRef.current = {
    onMessage: externalOnMessage,
    onError: externalOnError,
    onConnect: externalOnConnect,
    onDisconnect: externalOnDisconnect,
  };

  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      setStatus("connecting");
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setStatus("connected");
        reconnectAttempts.current = 0;
        callbacksRef.current.onConnect?.();
      };

      ws.onerror = (error) => {
        setStatus("error");
        callbacksRef.current.onError?.(error);
      };

      ws.onclose = () => {
        setStatus("disconnected");
        callbacksRef.current.onDisconnect?.();
        wsRef.current = null;

        // Attempt to reconnect
        if (enabled && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          callbacksRef.current.onMessage?.(data);

          // Try to parse common message formats
          // Adjust these based on your actual WebSocket message structure
          if (data.vehicles && Array.isArray(data.vehicles)) {
            settersRef.current.setVehicles(data.vehicles);
          }
          if (data.collisionPoints && Array.isArray(data.collisionPoints)) {
            settersRef.current.setCollisionPoints(data.collisionPoints);
          }
          if (data.incidentPoints && Array.isArray(data.incidentPoints)) {
            settersRef.current.setIncidentPoints(data.incidentPoints);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("Error creating WebSocket:", error);
      setStatus("error");
    }
  }, [url, enabled]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, []);

  const sendMessage = useCallback((message: string | object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const data = typeof message === "string" ? message : JSON.stringify(message);
      wsRef.current.send(data);
    } else {
      console.warn("WebSocket is not connected");
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]); // Only depend on enabled, connect/disconnect are stable

  return {
    status,
    connect,
    disconnect,
    sendMessage,
    isConnected: status === "connected",
  };
}



