import React from "react";
import Svg, { Path, Circle } from "react-native-svg";

interface IncidentsIconProps {
  size?: number;
  color?: string;
}

export function IncidentsIcon({ size = 24, color = "#FFFFFF" }: IncidentsIconProps) {
  // Use contrasting color for inner elements (white if dark, dark if light)
  const isLight = color === "#FFFFFF" || color === "#737373";
  const innerColor = isLight ? "#262626" : "#FFFFFF";
  
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Circle
        cx="12"
        cy="12"
        r="10"
        fill={color}
      />
      <Path
        d="M12 8v4M12 16h.01"
        stroke={innerColor}
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
      />
    </Svg>
  );
}

