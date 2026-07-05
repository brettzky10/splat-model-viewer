"use client";

import dynamic from "next/dynamic";

const GaussianSplatViewer = dynamic(
  () => import("@/components/global/PLYGaussianSplatViewer"),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#888",
          fontFamily: "sans-serif",
        }}
      >
        Loading viewer…
      </div>
    ),
  }
);

interface SplatViewerProps {
  splatUrl?: string;
  maxSplats?: number;
}

export default function SplatViewer(props: SplatViewerProps) {
  return <GaussianSplatViewer {...props} />;
}
