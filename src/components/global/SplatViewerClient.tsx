"use client";

import dynamic from "next/dynamic";

const GaussianSplatViewer = dynamic(
  () => import("@/components/global/GaussianSplatViewer"),
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

interface SplatViewerClientProps {
  splatUrl?: string;
  maxSplats?: number;
}

export default function SplatViewerClient(props: SplatViewerClientProps) {
  return <GaussianSplatViewer {...props} />;
}
