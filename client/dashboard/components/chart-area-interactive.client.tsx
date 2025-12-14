"use client"

import dynamic from "next/dynamic"

// This file is a Client Component, so ssr:false is allowed here.
export const ChartAreaInteractiveClient = dynamic(
  () =>
    import("@/components/chart-area-interactive").then(
      (m) => m.ChartAreaInteractive
    ),
  { ssr: false }
)
