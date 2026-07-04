"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GaussianSplatMesh } from "@zappar/three-gaussian-splat";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface GaussianSplatViewerProps {
  /**
   * Path to the .splat file, relative to the `public` folder,
   * e.g. "/bonsai.splat".
   */
  splatUrl?: string;
  /** Maximum number of splats to load. Defaults to Infinity. */
  maxSplats?: number;
}

export default function GaussianSplatViewer({
  splatUrl = "/bonsai.splat",
  maxSplats = Infinity,
}: GaussianSplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let animationRunning = true;

    const { clientWidth: width, clientHeight: height } = container;

    // --- Core three.js setup -------------------------------------------------
    const camera = new THREE.PerspectiveCamera(
      75,
      width / height,
      0.0001,
      100000
    );
    camera.position.set(0, 2, 6);

    const scene = new THREE.Scene();

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // --- Gaussian splat mesh --------------------------------------------------
    const splat = new GaussianSplatMesh(splatUrl, maxSplats);
    splat.load().catch((err) => {
      console.error("Failed to load splat file:", err);
    });
    splat.position.y = 2;
    splat.scale.setScalar(0.75);
    scene.add(splat);

    // --- Grid helper floor -----------------------------------------------------
    const grid = new THREE.GridHelper(10, 10);
    scene.add(grid);

    // --- Resize handling ---------------------------------------------------
    const onResize = () => {
      const { clientWidth, clientHeight } = container;
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(clientWidth, clientHeight);
    };
    window.addEventListener("resize", onResize);

    // --- Animation loop ------------------------------------------------------
    function animate() {
      if (!animationRunning) return;
      controls.update();
      splat.update(camera, renderer);
      renderer.render(scene, camera);
    }
    renderer.setAnimationLoop(animate);

    // --- Cleanup ---------------------------------------------------------------
    return () => {
      animationRunning = false;
      renderer.setAnimationLoop(null);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      scene.remove(splat, grid);
      splat.geometry.dispose();
      splat.material.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [splatUrl, maxSplats]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative" }}
    />
  );
}
