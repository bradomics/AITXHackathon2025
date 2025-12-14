import { TouchableOpacity, Text, ActivityIndicator, ViewStyle, TextStyle } from "react-native";

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: "primary" | "secondary" | "outline";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  textClassName?: string;
}

export function Button({
  title,
  onPress,
  variant = "primary",
  size = "md",
  disabled = false,
  loading = false,
  className = "",
  textClassName = "",
}: ButtonProps) {
  const baseClasses = "rounded-lg items-center justify-center";
  
  const variantClasses = {
    primary: "bg-blue-500 active:bg-blue-600",
    secondary: "bg-neutral-200 dark:bg-neutral-700 active:bg-neutral-300 dark:active:bg-neutral-600",
    outline: "border-2 border-blue-500 bg-transparent",
  };

  const sizeClasses = {
    sm: "px-4 py-2",
    md: "px-6 py-3",
    lg: "px-8 py-4",
  };

  const textVariantClasses = {
    primary: "text-white",
    secondary: "text-neutral-900 dark:text-neutral-50",
    outline: "text-blue-500",
  };

  const textSizeClasses = {
    sm: "text-sm",
    md: "text-base",
    lg: "text-lg",
  };

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || loading}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${
        disabled || loading ? "opacity-50" : ""
      } ${className}`}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === "primary" ? "white" : variant === "outline" ? "#3b82f6" : undefined}
        />
      ) : (
        <Text
          className={`font-semibold ${textVariantClasses[variant]} ${textSizeClasses[size]} ${textClassName}`}
        >
          {title}
        </Text>
      )}
    </TouchableOpacity>
  );
}



