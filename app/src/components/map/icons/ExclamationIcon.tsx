import React from "react";
import Svg, { Path, Circle } from "react-native-svg";

interface ExclamationIconProps {
  size?: number;
  color?: string;
}

export function ExclamationIcon({ size = 24, color = "#3B82F6" }: ExclamationIconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      {/* Circle background */}
      <Circle
        cx="12"
        cy="12"
        r="10"
        fill={color}
        stroke="#FFFFFF"
        strokeWidth="1.5"
      />
      {/* Exclamation mark */}
      <Path
        d="M12 6v6M12 16h.01"
        stroke="#FFFFFF"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </Svg>
  );
}

