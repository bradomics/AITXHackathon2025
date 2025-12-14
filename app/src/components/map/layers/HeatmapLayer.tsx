import React from "react";
import { Circle, Marker } from "react-native-maps";
import { BaseLayer, LayerProps } from "./BaseLayer";
import { RiskPoint } from "../../../types";

export interface HeatmapLayerProps extends LayerProps {
  data: RiskPoint[];
  getPosition?: (d: RiskPoint) => [number, number];
  getWeight?: (d: RiskPoint) => number;
  radiusPixels?: number;
  intensity?: number;
  threshold?: number;
  colorRange?: [number, number, number, number][];
  radiusScale?: number;
}

/**
 * HeatmapLayer - mimics Deck.gl's HeatmapLayer
 * Renders circles with varying opacity/radius based on weight
 */
export class HeatmapLayer extends BaseLayer<HeatmapLayerProps> {
  static defaultProps = {
    getPosition: (d: RiskPoint) => d.position,
    getWeight: (d: RiskPoint) => d.weight,
    radiusPixels: 50,
    intensity: 1,
    threshold: 0.03,
    radiusScale: 1,
    visible: true,
    opacity: 1,
  };

  render() {
    const {
      data,
      getPosition,
      getWeight,
      radiusPixels = 50,
      intensity = 1,
      threshold = 0.03,
      colorRange,
      radiusScale = 1,
      visible = true,
      opacity = 1,
    } = this.props;

    if (!visible || !data || data.length === 0) {
      return null;
    }

    // Default color range (red gradient)
    const defaultColorRange: [number, number, number, number][] = [
      [255, 80, 80, 0],
      [255, 80, 80, 40],
      [255, 80, 80, 90],
      [255, 80, 80, 140],
      [255, 80, 80, 200],
      [255, 80, 80, 255],
    ];

    const colors = colorRange || defaultColorRange;

    return (
      <>
        {data.map((point, index) => {
          const position = getPosition!(point);
          const weight = getWeight!(point);

          // Skip points below threshold
          if (weight < threshold) {
            return null;
          }

          // Calculate radius based on weight and intensity
          const radius = (radiusPixels * radiusScale * weight * intensity) / 1000; // Convert to meters

          // Interpolate color based on weight
          const colorIndex = Math.min(
            Math.floor((weight / 1.0) * (colors.length - 1)),
            colors.length - 1
          );
          const color = colors[colorIndex];
          const [r, g, b, a] = color;

          return (
            <Circle
              key={`heatmap-${index}`}
              center={{
                latitude: position[1],
                longitude: position[0],
              }}
              radius={radius}
              fillColor={`rgba(${r}, ${g}, ${b}, ${(a / 255) * opacity})`}
              strokeColor={`rgba(${r}, ${g}, ${b}, ${Math.min(255, a + 50) / 255})`}
              strokeWidth={2}
              zIndex={1}
            />
          );
        })}
      </>
    );
  }
}

