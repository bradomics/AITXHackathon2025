import { View, Text, TouchableOpacity } from "react-native";
import { UserRole } from "../../types";

interface RoleSelectorProps {
  selectedRole: UserRole | null;
  onSelectRole: (role: UserRole) => void;
}

export function RoleSelector({ selectedRole, onSelectRole }: RoleSelectorProps) {
  const roles: { id: UserRole; title: string }[] = [
    {
      id: "ems",
      title: "EMS",
    },
    {
      id: "roadside",
      title: "Roadside Assistance",
    },
    {
      id: "public-safety",
      title: "Public Safety",
    },
  ];

  return (
    <View className="w-full">
      <Text className="text-base text-neutral-600 dark:text-neutral-400 mb-4">
        Which best describes your role?
      </Text>

      <View className="flex-row dark:bg-neutral-800 bg-white rounded-lg">
        {roles.map((role) => (
          <TouchableOpacity
            key={role.id}
            onPress={() => onSelectRole(role.id)}
            className={`flex-1 p-4 rounded-xl items-center justify-center ${
              selectedRole === role.id
                ? "bg-neutral-800 dark:bg-white/10"
                : "bg-transparent"
            }`}
          >
            <Text
              className={`text-center text-base font-semibold ${
                selectedRole === role.id
                  ? "text-white dark:text-white"
                  : "text-neutral-900 dark:text-neutral-50"
              }`}
            >
              {role.title}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}



