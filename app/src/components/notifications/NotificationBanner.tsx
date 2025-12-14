import { View, Text, TouchableOpacity, Animated, PanResponder } from "react-native";
import { useEffect, useState } from "react";
import { RecommendedSpot } from "../../types";

interface NotificationBannerProps {
  recommendedSpots: RecommendedSpot[];
  onDismiss?: (spotId: string) => void;
  onPress?: (spot: RecommendedSpot) => void;
}

interface NotificationItem {
  spot: RecommendedSpot;
  translateX: Animated.Value;
  translateY: Animated.Value;
  opacity: Animated.Value;
  panResponder: ReturnType<typeof PanResponder.create>;
}

export function NotificationBanner({
  recommendedSpots,
  onDismiss,
  onPress,
}: NotificationBannerProps) {
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);

  const handleDismiss = (item: NotificationItem) => {
    Animated.parallel([
      Animated.timing(item.translateX, {
        toValue: -400,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(item.opacity, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setNotifications((prev) => prev.filter((n) => n.spot.id !== item.spot.id));
      onDismiss?.(item.spot.id);
    });
  };

  useEffect(() => {
    if (recommendedSpots.length > 0) {
      const newSpots = recommendedSpots.slice(0, 5); // Show max 5 at a time
      
      setNotifications((prev) => {
        const existingIds = new Set(prev.map(n => n.spot.id));
        const newItems = newSpots
          .filter(spot => !existingIds.has(spot.id))
          .map(spot => {
            const translateX = new Animated.Value(0);
            const translateY = new Animated.Value(-100); // Start above screen
            const opacity = new Animated.Value(0); // Start invisible
            
            // Animate in from top
            Animated.parallel([
              Animated.spring(translateY, {
                toValue: 0,
                useNativeDriver: true,
                tension: 65,
                friction: 11,
              }),
              Animated.timing(opacity, {
                toValue: 1,
                duration: 300,
                useNativeDriver: true,
              }),
            ]).start();
            
            const panResponder = PanResponder.create({
              onStartShouldSetPanResponder: () => true,
              onMoveShouldSetPanResponder: (_, gestureState) => {
                return Math.abs(gestureState.dx) > 10;
              },
              onPanResponderMove: (_, gestureState) => {
                if (gestureState.dx < 0) {
                  translateX.setValue(gestureState.dx);
                }
              },
              onPanResponderRelease: (_, gestureState) => {
                if (gestureState.dx < -100) {
                  // Swipe left enough to dismiss
                  Animated.parallel([
                    Animated.timing(translateX, {
                      toValue: -400,
                      duration: 300,
                      useNativeDriver: true,
                    }),
                    Animated.timing(opacity, {
                      toValue: 0,
                      duration: 300,
                      useNativeDriver: true,
                    }),
                  ]).start(() => {
                    setNotifications((current) => current.filter((n) => n.spot.id !== spot.id));
                    onDismiss?.(spot.id);
                  });
                } else {
                  // Spring back
                  Animated.spring(translateX, {
                    toValue: 0,
                    useNativeDriver: true,
                    tension: 65,
                    friction: 11,
                  }).start();
                }
              },
            });
            
            return {
              spot,
              translateX,
              translateY,
              opacity,
              panResponder,
            };
          });
        
        return [...newItems, ...prev].slice(0, 5);
      });
    }
  }, [recommendedSpots, onDismiss]);

  if (notifications.length === 0) {
    return null;
  }

  return (
    <View className="absolute top-12 left-4 right-4 z-50" pointerEvents="box-none">
      {notifications.map((item, index) => {
        const stackOffset = index * 70; // Visual stacking offset (height of card)
        const scale = 1 - index * 0.03; // Slight scale down for depth
        const zIndex = 50 - index; // Higher z-index for newer notifications

        return (
          <Animated.View
            key={item.spot.id}
            style={{
              position: 'absolute',
              top: stackOffset,
              left: 0,
              right: 0,
              transform: [
                { translateX: item.translateX },
                { translateY: item.translateY },
                { scale: scale },
              ],
              opacity: item.opacity,
              zIndex: zIndex,
            }}
            {...item.panResponder.panHandlers}
          >
            <TouchableOpacity
              onPress={() => onPress?.(item.spot)}
              activeOpacity={0.8}
              className="bg-blue-500 dark:bg-blue-600 rounded-lg p-4 shadow-lg flex-row items-center justify-between"
            >
              <View className="flex-1">
                <Text className="text-white font-semibold text-base">{item.spot.name}</Text>
                <Text className="text-blue-100 text-sm mt-1">
                  {item.spot.distance ? `${Math.round(item.spot.distance)}m away` : "Nearby"}
                </Text>
              </View>
              {onDismiss && (
                <TouchableOpacity
                  onPress={() => handleDismiss(item)}
                  className="ml-4 p-2"
                >
                  <Text className="text-white text-lg font-bold">×</Text>
                </TouchableOpacity>
              )}
            </TouchableOpacity>
          </Animated.View>
        );
      })}
    </View>
  );
}



