import React from "react";
import Svg, { Path } from "react-native-svg";

interface MapIconProps {
  size?: number;
  color?: string;
}

export function MapIcon({ size = 24, color = "#FFFFFF" }: MapIconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill={color}
        fillOpacity="0.1"
      />
    </Svg>
  );
}

