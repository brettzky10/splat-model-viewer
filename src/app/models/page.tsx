import SparkGaussianSplatViewer from "@/components/global/SparkGaussianSplatViewer";
import SplatViewer from "@/components/global/SplatViewer";

export default function Home() {
  return (
    <main style={{ width: "100vw", height: "100vh", margin: 0 }}>
      <SparkGaussianSplatViewer />
    </main>
  );
}