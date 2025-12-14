import React from "react";
import { View } from "react-native";

/**
 * Base layer interface matching Deck.gl's layer API
 */
export interface LayerProps {
  id: string;
  data: any[];
  visible?: boolean;
  opacity?: number;
  pickable?: boolean;
  onHover?: (info: any) => void;
  onClick?: (info: any) => void;
}

export abstract class BaseLayer<P extends LayerProps = LayerProps> extends React.Component<P> {
  abstract render(): React.ReactNode;
}

