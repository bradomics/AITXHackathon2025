import React from "react";
import Svg, { Path } from "react-native-svg";

interface RiskIconProps {
  size?: number;
  color?: string;
}

export function RiskIcon({ size = 24, color = "#FFFFFF" }: RiskIconProps) {
  // Use contrasting color for inner elements (white if dark, dark if light)
  const isLight = color === "#FFFFFF" || color === "#737373";
  const innerColor = isLight ? "#262626" : "#FFFFFF";
  
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 2L2 22h20L12 2z"
        fill={color}
      />
      <Path
        d="M12 9v4M12 17h.01"
        stroke={innerColor}
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
    </Svg>
  );
}

