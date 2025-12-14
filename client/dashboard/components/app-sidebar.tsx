"use client"

import * as React from "react"
import {
  IconCamera,
  IconChartBar,
  IconDashboard,
  IconDatabase,
  IconFileAi,
  IconFileDescription,
  IconFileWord,
  IconFolder,
  IconHelmet,
  IconHelp,
  IconInnerShadowTop,
  IconListDetails,
  IconReport,
  IconRoad,
  IconSearch,
  IconSettings,
  IconShield,
  IconShieldHalf,
  IconShieldBolt,
  IconUsers,
  IconCarCrane,
} from "@tabler/icons-react"

import { NavDocuments } from "@/components/nav-documents"
import { NavMain } from "@/components/nav-main"
import { NavSecondary } from "@/components/nav-secondary"
import { NavUser } from "@/components/nav-user"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { MapPin } from "lucide-react"

const data = {
  user: {
    name: "admin",
    email: "admin@austin.gov",
    avatar: "/avatars/shadcn.jpg",
  },
  navMain: [
    {
      title: "Public Safety / EMS",
      url: "/public-safety",
      icon: IconShieldHalf,
    },
    {
      title: "Traffic Controller / Dispatcher",
      url: "/traffic-control",
      icon: IconRoad,
    },
    {
      title: "Tow Truck / Roadside Assistance",
      url: "/truck-operator",
      icon: IconCarCrane,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "#",
      icon: IconSettings,
    },
    {
      title: "Get Help",
      url: "#",
      icon: IconHelp,
    },
    {
      title: "Search",
      url: "#",
      icon: IconSearch,
    },
  ],
  documents: [
    {
      name: "Data Library",
      url: "#",
      icon: IconDatabase,
    },
    {
      name: "Reports",
      url: "#",
      icon: IconReport,
    },
    {
      name: "Word Assistant",
      url: "#",
      icon: IconFileWord,
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="#">
                <MapPin className="!size-5" />
                <span className="text-base font-medium" style={{ letterSpacing: '-1px' }}><strong>ETA:</strong> Ensemble Traffic Agents</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  )
}
