import React from "react";
import Svg, { Path } from "react-native-svg";

interface WarningIconProps {
  size?: number;
  color?: string;
}

export function WarningIcon({ size = 24, color = "#EF4444" }: WarningIconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      {/* Triangle background */}
      <Path
        d="M12 2L2 22h20L12 2z"
        fill={color}
        stroke="#FFFFFF"
        strokeWidth="1.5"
      />
      {/* Exclamation mark */}
      <Path
        d="M12 8v4M12 16h.01"
        stroke="#FFFFFF"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </Svg>
  );
}

