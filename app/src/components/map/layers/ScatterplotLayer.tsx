import React from "react";
import { Marker } from "react-native-maps";
import { BaseLayer, LayerProps } from "./BaseLayer";
import { View } from "react-native";

export interface ScatterplotLayerProps<T = any> extends LayerProps {
  data: T[];
  getPosition: (d: T) => [number, number];
  getRadius?: (d: T) => number;
  getColor?: (d: T) => [number, number, number, number];
  getFillColor?: (d: T) => [number, number, number, number];
  radiusScale?: number;
  radiusMinPixels?: number;
  radiusMaxPixels?: number;
  stroked?: boolean;
  getLineColor?: (d: T) => [number, number, number, number];
  lineWidthMinPixels?: number;
  renderMarker?: (d: T, index: number) => React.ReactNode;
}

/**
 * ScatterplotLayer - mimics Deck.gl's ScatterplotLayer
 * Renders points as circles or custom markers
 */
export class ScatterplotLayer<T = any> extends BaseLayer<ScatterplotLayerProps<T>> {
  static defaultProps = {
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 100,
    stroked: false,
    lineWidthMinPixels: 1,
    visible: true,
    opacity: 1,
  };

  render() {
    const {
      data,
      getPosition,
      getRadius,
      getColor,
      getFillColor,
      radiusScale = 1,
      radiusMinPixels = 1,
      radiusMaxPixels = 100,
      stroked = false,
      getLineColor,
      lineWidthMinPixels = 1,
      visible = true,
      opacity = 1,
      renderMarker,
      onClick,
    } = this.props;

    if (!visible || !data || data.length === 0) {
      return null;
    }

    return (
      <>
        {data.map((point, index) => {
          const position = getPosition(point);
          const radius = getRadius
            ? Math.max(
                radiusMinPixels,
                Math.min(radiusMaxPixels, getRadius(point) * radiusScale)
              )
            : 8;

          const fillColor = getFillColor
            ? getFillColor(point)
            : getColor
            ? getColor(point)
            : [0, 0, 255, 255];

          const [r, g, b, a] = fillColor;
          const fillColorString = `rgba(${r}, ${g}, ${b}, ${(a / 255) * opacity})`;

          const lineColor = stroked && getLineColor
            ? getLineColor(point)
            : [255, 255, 255, 255];
          const [lr, lg, lb, la] = lineColor;
          const lineColorString = `rgba(${lr}, ${lg}, ${lb}, ${la / 255})`;

          if (renderMarker) {
            return (
              <Marker
                key={`scatter-${index}`}
                coordinate={{
                  latitude: position[1],
                  longitude: position[0],
                }}
                onPress={() => onClick?.({ object: point, index })}
              >
                {renderMarker(point, index)}
              </Marker>
            );
          }

          // Default: render as circle using a custom view
          return (
            <Marker
              key={`scatter-${index}`}
              coordinate={{
                latitude: position[1],
                longitude: position[0],
              }}
              onPress={() => onClick?.({ object: point, index })}
            >
              <View
                style={{
                  width: radius * 2,
                  height: radius * 2,
                  borderRadius: radius,
                  backgroundColor: fillColorString,
                  borderWidth: stroked ? lineWidthMinPixels : 0,
                  borderColor: lineColorString,
                }}
              />
            </Marker>
          );
        })}
      </>
    );
  }
}

