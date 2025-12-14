"use client"

import * as React from "react"
import dynamic from "next/dynamic"
import { usePathname } from "next/navigation"

// Keep these imports if other parts of your app depend on them being here.
// (Safe even if unused.)
import { useIsMobile } from "@/hooks/use-mobile"

import { AustinTrafficControllerCard } from "./traffic-controller-card"
export const description = "An interactive area chart"

// ✅ Load the Deck/Mapbox-heavy card only on the client.
// const AustinHeatmapCard = dynamic(
//   () => import("./heatmap-card").then((m) => m.AustinHeatmapCard),
//   { ssr: false, loading: () => <div className="h-[520px] w-full rounded-xl border" /> }
// )

export function TrafficControllerInteractive() {
  const pathname = usePathname()
  const isMobile = useIsMobile()

  const [mounted, setMounted] = React.useState(false)
  const [timeRange, setTimeRange] = React.useState("90d")

  React.useEffect(() => setMounted(true), [])
  React.useEffect(() => {
    if (isMobile) setTimeRange("7d")
  }, [isMobile])

  if (!mounted) return null

  // ✅ This key forces a clean teardown/remount across route transitions,
  // which prevents stale GPU "device/limits" state from being reused.
  return <AustinTrafficControllerCard key={pathname} />
}
