import React from "react";
import { Marker } from "react-native-maps";
import { BaseLayer, LayerProps } from "./BaseLayer";
import { Image, View } from "react-native";

export interface IconLayerProps<T = any> extends LayerProps {
  data: T[];
  getPosition: (d: T) => [number, number];
  getIcon?: (d: T) => string | number | { uri: string };
  getSize?: (d: T) => number;
  getColor?: (d: T) => [number, number, number, number];
  iconAtlas?: any;
  iconMapping?: any;
  sizeScale?: number;
  sizeMinPixels?: number;
  sizeMaxPixels?: number;
  renderIcon?: (d: T, index: number) => React.ReactNode;
}

/**
 * IconLayer - mimics Deck.gl's IconLayer
 * Renders icons/images at specific positions
 */
export class IconLayer<T = any> extends BaseLayer<IconLayerProps<T>> {
  static defaultProps = {
    sizeScale: 1,
    sizeMinPixels: 1,
    sizeMaxPixels: 100,
    visible: true,
    opacity: 1,
  };

  render() {
    const {
      data,
      getPosition,
      getIcon,
      getSize,
      getColor,
      sizeScale = 1,
      sizeMinPixels = 1,
      sizeMaxPixels = 100,
      visible = true,
      opacity = 1,
      renderIcon,
      onClick,
    } = this.props;

    if (!visible || !data || data.length === 0) {
      return null;
    }

    return (
      <>
        {data.map((point, index) => {
          const position = getPosition(point);
          const size = getSize
            ? Math.max(
                sizeMinPixels,
                Math.min(sizeMaxPixels, getSize(point) * sizeScale)
              )
            : 24;

          const icon = getIcon ? getIcon(point) : null;
          const color = getColor ? getColor(point) : [255, 255, 255, 255];
          const [r, g, b, a] = color;

          if (renderIcon) {
            return (
              <Marker
                key={`icon-${index}`}
                coordinate={{
                  latitude: position[1],
                  longitude: position[0],
                }}
                onPress={() => onClick?.({ object: point, index })}
              >
                {renderIcon(point, index)}
              </Marker>
            );
          }

          // Default: render icon if provided
          if (icon) {
            const iconSource =
              typeof icon === "string"
                ? { uri: icon }
                : typeof icon === "number"
                ? icon
                : icon;

            return (
              <Marker
                key={`icon-${index}`}
                coordinate={{
                  latitude: position[1],
                  longitude: position[0],
                }}
                onPress={() => onClick?.({ object: point, index })}
              >
                <Image
                  source={iconSource}
                  style={{
                    width: size,
                    height: size,
                    tintColor: `rgba(${r}, ${g}, ${b}, ${(a / 255) * opacity})`,
                  }}
                  resizeMode="contain"
                />
              </Marker>
            );
          }

          // Fallback: render as colored circle
          return (
            <Marker
              key={`icon-${index}`}
              coordinate={{
                latitude: position[1],
                longitude: position[0],
              }}
              onPress={() => onClick?.({ object: point, index })}
            >
              <View
                style={{
                  width: size,
                  height: size,
                  borderRadius: size / 2,
                  backgroundColor: `rgba(${r}, ${g}, ${b}, ${(a / 255) * opacity})`,
                }}
              />
            </Marker>
          );
        })}
      </>
    );
  }
}

