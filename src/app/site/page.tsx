import SplatViewerClient from "@/components/global/SplatViewerClient";

export default function Home() {
  return (
    <main style={{ width: "100vw", height: "100vh", margin: 0 }}>
      <SplatViewerClient splatUrl="/bonsai.splat" />
    </main>
  );
}