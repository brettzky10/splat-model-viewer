"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
// npm i @mkkellogg/gaussian-splats-3d
// Requires three@^0.160.0 or newer as a peer dependency.
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// @mediapipe/hands doesn't ship real ESM/CJS exports - it just attaches
// `Hands` (and friends) onto the global object as a side effect of evaluating
// the module, and it's marked `"sideEffects": []` in its package.json, which
// makes bundlers tree-shake away any plain `import "@mediapipe/hands"` (and
// even a dynamic `import()` can be finicky depending on the bundler/CDN combo
// involved). The reliable fix - and what MediaPipe's own docs/demos do - is
// to load it as a plain <script> tag pointed at the CDN, exactly like a
// static HTML page would, and read the resulting global off `window`. We
// only use the npm package here for its TypeScript types (type-only imports
// have zero runtime footprint, so this part is unaffected by any of the
// above).
import type {
  HandsConfig,
  Options as HandsOptions,
  ResultsListener as HandsResultsListener,
  NormalizedLandmark,
} from "@mediapipe/hands";

interface MediapipeHandsInstance {
  onResults(listener: HandsResultsListener): void;
  send(inputs: { image: HTMLVideoElement }): Promise<void>;
  setOptions(options: HandsOptions): void;
  close(): Promise<void>;
}

declare global {
  interface Window {
    Hands: new (config?: HandsConfig) => MediapipeHandsInstance;
  }
}

const HANDS_SCRIPT_SRC = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";

// Loads a script exactly once (safe to call repeatedly, e.g. across toggling
// hand tracking on/off/on - later calls resolve immediately).
function loadScriptOnce(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${src}"]`
    );
    if (existing) {
      if (existing.dataset.loaded === "true") {
        resolve();
      } else {
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () =>
          reject(new Error(`Failed to load script: ${src}`))
        );
      }
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    });
    script.addEventListener("error", () =>
      reject(new Error(`Failed to load script: ${src}`))
    );
    document.head.appendChild(script);
  });
}

// ---------------------------------------------------------------------------
// Splat source handling (.splat / .ply / .ksplat)
// ---------------------------------------------------------------------------

type SplatFormat = "ply" | "splat" | "ksplat";

interface SplatSource {
  /** URL, path relative to `public`, or an object URL for a locally-picked file. */
  url: string;
  /**
   * Only needed when the URL's extension can't be used to infer the format
   * (e.g. a `blob:` URL from a local file picker). GaussianSplats3D can infer
   * the format itself from a normal `/model.ply` style path.
   */
  format?: SplatFormat;
}

function sceneFormatFor(format?: SplatFormat) {
  switch (format) {
    case "ply":
      return GaussianSplats3D.SceneFormat.Ply;
    case "splat":
      return GaussianSplats3D.SceneFormat.Splat;
    case "ksplat":
      return GaussianSplats3D.SceneFormat.KSplat;
    default:
      return undefined;
  }
}

function formatFromFileName(name: string): SplatFormat | null {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "ply" || ext === "splat" || ext === "ksplat") return ext;
  return null;
}

interface GaussianSplatViewerProps {
  /**
   * Path to the .splat, .ply, or .ksplat file, relative to the `public`
   * folder (or a full URL), e.g. "/bonsai.splat" or "/scan.ply".
   */
  splatUrl?: string;
  /** Only needed if `splatUrl`'s extension doesn't already say what it is. */
  splatFormat?: SplatFormat;
  /** Ignore splats with alpha below this value (0-255). Defaults to 5. */
  splatAlphaRemovalThreshold?: number;
}

// ---------------------------------------------------------------------------
// Hand-landmark geometry helpers
// ---------------------------------------------------------------------------

// Standard 21-point MediaPipe hand landmark skeleton edges, used to draw the
// hand overlay (avoids depending on @mediapipe/hands' HAND_CONNECTIONS export,
// which suffers from the same "global-only" export problem noted above).
const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // pinky + palm
];

type Point2D = { x: number; y: number };

function dist(a: Point2D, b: Point2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

interface FingerStates {
  index: boolean;
  middle: boolean;
  ring: boolean;
  pinky: boolean;
}

// A finger counts as "extended" if its tip sits noticeably farther from the
// wrist than its middle (PIP) joint does. This is a cheap, orientation
// tolerant heuristic - good enough for gesture triggers without needing full
// 3D hand-pose reasoning.
function isFingerExtended(
  landmarks: NormalizedLandmark[],
  tipIdx: number,
  pipIdx: number
): boolean {
  const wrist = landmarks[0];
  return dist(landmarks[tipIdx], wrist) > dist(landmarks[pipIdx], wrist) * 1.15;
}

function getFingerStates(landmarks: NormalizedLandmark[]): FingerStates {
  return {
    index: isFingerExtended(landmarks, 8, 6),
    middle: isFingerExtended(landmarks, 12, 10),
    ring: isFingerExtended(landmarks, 16, 14),
    pinky: isFingerExtended(landmarks, 20, 18),
  };
}

function isFist(f: FingerStates): boolean {
  return !f.index && !f.middle && !f.ring && !f.pinky;
}

function isPeaceSign(f: FingerStates): boolean {
  return f.index && f.middle && !f.ring && !f.pinky;
}

// Pinch = thumb tip close to index tip, relative to the hand's own scale
// (distance from wrist to middle-finger knuckle) so it works regardless of
// how close the hand is to the camera.
//
// A closed fist often also brings the thumb tip quite close to the index
// tip (the thumb rests over the curled fingers), which used to get
// misread as a pinch. We gate on `fingers` here so a hand that's fully
// curled is always classified as a fist first, never as a pinch.
function isPinching(
  landmarks: NormalizedLandmark[],
  fingers: FingerStates
): boolean {
  if (isFist(fingers)) return false;
  const handSize = dist(landmarks[0], landmarks[9]) || 0.1;
  const pinchDist = dist(landmarks[4], landmarks[8]);
  return pinchDist < handSize * 0.35;
}

// ---------------------------------------------------------------------------
// Landmark smoothing
// ---------------------------------------------------------------------------
// Raw per-frame landmarks are noisy enough to make both the overlay and the
// camera gestures feel jumpy. We keep an exponential moving average per
// hand (keyed by MediaPipe's "Left"/"Right" handedness label, so smoothing
// carries over correctly even if hands swap array order between frames) and
// feed the smoothed points into everything downstream instead of the raw
// ones.
type HandLabel = "Left" | "Right";
type LabeledHand = { label: HandLabel; landmarks: NormalizedLandmark[] };

// Lower = smoother but laggier, higher = snappier but jumpier.
const LANDMARK_SMOOTHING = 0.4;

function smoothLandmarkList(
  prev: NormalizedLandmark[] | null,
  next: NormalizedLandmark[]
): NormalizedLandmark[] {
  if (!prev || prev.length !== next.length) {
    return next.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
  }
  return next.map((p, i) => ({
    x: prev[i].x + (p.x - prev[i].x) * LANDMARK_SMOOTHING,
    y: prev[i].y + (p.y - prev[i].y) * LANDMARK_SMOOTHING,
    z: (prev[i].z ?? 0) + ((p.z ?? 0) - (prev[i].z ?? 0)) * LANDMARK_SMOOTHING,
  }));
}

// Small per-frame deltas below this are treated as noise rather than
// intentional movement, so a "still" hand doesn't slowly drift the camera.
const ZOOM_DEADZONE = 0.002;
const PAN_DEADZONE = 0.0015;
const ROTATE_DEADZONE = 0.0015;

const MIN_DISTANCE = 0.5;
const MAX_DISTANCE = 80;
const ZOOM_SENSITIVITY = 6;
const PAN_SENSITIVITY = 1.6;
const ROTATE_SENSITIVITY = 6;

function zoomCamera(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  distanceDelta: number
) {
  const offset = camera.position.clone().sub(controls.target);
  const distance = offset.length();
  const scale = 1 - distanceDelta * ZOOM_SENSITIVITY;
  const newDistance = THREE.MathUtils.clamp(
    distance * scale,
    MIN_DISTANCE,
    MAX_DISTANCE
  );
  offset.setLength(newDistance);
  camera.position.copy(controls.target).add(offset);
}

function panCamera(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  dx: number,
  dy: number
) {
  camera.updateMatrixWorld();
  const distance = camera.position.distanceTo(controls.target);
  const panSpeed = distance * PAN_SENSITIVITY;

  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);

  const panOffset = new THREE.Vector3();
  panOffset.addScaledVector(right, -dx * panSpeed);
  panOffset.addScaledVector(up, dy * panSpeed);

  camera.position.add(panOffset);
  controls.target.add(panOffset);
}

function rotateCamera(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  dx: number,
  dy: number
) {
  const offset = camera.position.clone().sub(controls.target);
  const spherical = new THREE.Spherical().setFromVector3(offset);

  spherical.theta -= dx * ROTATE_SENSITIVITY;
  spherical.phi -= dy * ROTATE_SENSITIVITY;

  const EPS = 0.001;
  spherical.phi = THREE.MathUtils.clamp(spherical.phi, EPS, Math.PI - EPS);

  const newOffset = new THREE.Vector3().setFromSpherical(spherical);
  camera.position.copy(controls.target).add(newOffset);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GaussianSplatViewer({
  splatUrl = "/bonsai.splat",
  splatFormat,
  splatAlphaRemovalThreshold = 5,
}: GaussianSplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Populated by the three.js setup effect, read by the gesture handler.
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const viewerRef = useRef<GaussianSplats3D.Viewer | null>(null);

  const handsRef = useRef<MediapipeHandsInstance | null>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [handTrackingEnabled, setHandTrackingEnabled] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [gestureStatus, setGestureStatus] = useState("");
  const [trackingError, setTrackingError] = useState<string | null>(null);

  // Which splat/PLY file is actually loaded: either the `splatUrl` prop, or
  // whatever the user picked with the file input below.
  const [uploadedSource, setUploadedSource] = useState<SplatSource | null>(
    null
  );
  const [splatLoading, setSplatLoading] = useState(true);
  const [splatError, setSplatError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const activeSource: SplatSource = uploadedSource ?? {
    url: splatUrl,
    format: splatFormat,
  };

  // Frame-to-frame gesture memory (kept in a ref so updates don't retrigger
  // React renders on every single video frame).
  const gestureMemory = useRef<{
    prevZoomDist: number | null;
    prevPanCenter: Point2D | null;
    prevRotateCenter: Point2D | null;
  }>({ prevZoomDist: null, prevPanCenter: null, prevRotateCenter: null });

  // Smoothed landmark history per hand label, used to reduce jitter (see
  // `smoothLandmarkList` above).
  const smoothedHands = useRef<Record<HandLabel, NormalizedLandmark[] | null>>(
    { Left: null, Right: null }
  );

  // --- Core three.js scene + GaussianSplats3D viewer setup ------------------
  // We build our own renderer/camera/OrbitControls (so the gesture handler
  // below can drive the same camera the user could otherwise orbit by hand)
  // and hand them to GaussianSplats3D.Viewer in "third-party integration"
  // mode (selfDrivenMode: false, useBuiltInControls: false). That mode also
  // accepts a `threeScene` so ordinary three.js objects (like the grid
  // helper) composite correctly with the splats' depth buffer.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let animationRunning = true;
    let cancelled = false;

    const { clientWidth: width, clientHeight: height } = container;

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

    cameraRef.current = camera;
    controlsRef.current = controls;

    const grid = new THREE.GridHelper(10, 10);
    scene.add(grid);

    const viewer = new GaussianSplats3D.Viewer({
      selfDrivenMode: false,
      renderer,
      camera,
      useBuiltInControls: false,
      threeScene: scene,
      ignoreDevicePixelRatio: false,
      // Avoids requiring cross-origin-isolation (COOP/COEP) response headers
      // for SharedArrayBuffer, which most simple app setups don't send.
      sharedMemoryForWorkers: false,
    });
    viewerRef.current = viewer;

    setSplatLoading(true);
    setSplatError(null);

    const format = sceneFormatFor(activeSource.format);
    viewer
      .addSplatScene(activeSource.url, {
        splatAlphaRemovalThreshold,
        showLoadingUI: true,
        position: [0, 2, 0],
        rotation: [0, 0, 0, 1],
        scale: [0.75, 0.75, 0.75],
        ...(format !== undefined ? { format } : {}),
      })
      .then(() => {
        if (!cancelled) setSplatLoading(false);
      })
      .catch((err: unknown) => {
        console.error("Failed to load splat/PLY scene:", err);
        if (!cancelled) {
          setSplatError(
            err instanceof Error
              ? err.message
              : "Failed to load the splat/PLY file."
          );
          setSplatLoading(false);
        }
      });

    const onResize = () => {
      const { clientWidth, clientHeight } = container;
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(clientWidth, clientHeight);
    };
    window.addEventListener("resize", onResize);

    function animate() {
      if (!animationRunning) return;
      controls.update();
      viewer.update();
      viewer.render();
    }
    renderer.setAnimationLoop(animate);

    return () => {
      cancelled = true;
      animationRunning = false;
      renderer.setAnimationLoop(null);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      try {
        viewer.dispose();
      } catch (err) {
        console.error("Error disposing splat viewer:", err);
      }
      renderer.dispose();
      scene.remove(grid);
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      cameraRef.current = null;
      controlsRef.current = null;
      viewerRef.current = null;
    };
  }, [
    activeSource.url,
    activeSource.format,
    splatAlphaRemovalThreshold,
  ]);

  // Revoke any object URL created for a locally-picked file once it's
  // replaced or the component unmounts, so we don't leak memory.
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  const handleFilePicked = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = ""; // allow re-picking the same file later
      if (!file) return;

      const format = formatFromFileName(file.name);
      if (!format) {
        setSplatError(
          `Unsupported file type: "${file.name}". Please choose a .ply, .splat, or .ksplat file.`
        );
        return;
      }

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setSplatError(null);
      setUploadedSource({ url, format });
    },
    []
  );

  // --- Draw just the hand skeleton (no camera feed) onto the overlay -------
  const drawHandOverlay = useCallback((hands: NormalizedLandmark[][]) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { clientWidth, clientHeight } = canvas;
    if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
      canvas.width = clientWidth;
      canvas.height = clientHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const landmarks of hands) {
      ctx.strokeStyle = "rgba(125, 211, 252, 0.9)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        const from = landmarks[a];
        const to = landmarks[b];
        ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
        ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      }
      ctx.stroke();

      ctx.fillStyle = "rgba(56, 189, 248, 0.95)";
      for (const lm of landmarks) {
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, []);

  // --- Interpret landmarks into camera gestures -----------------------------
  const applyGestures = useCallback((hands: NormalizedLandmark[][]) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const memory = gestureMemory.current;
    if (!camera || !controls) return;

    if (hands.length < 2) {
      memory.prevZoomDist = null;
      memory.prevPanCenter = null;
      memory.prevRotateCenter = null;
      setGestureStatus(
        hands.length === 1
          ? "Show your second hand to control the view"
          : "Show both hands to the camera"
      );
      return;
    }

    const [handA, handB] = hands;
    const fingersA = getFingerStates(handA);
    const fingersB = getFingerStates(handB);
    const pinchA = isPinching(handA, fingersA);
    const pinchB = isPinching(handB, fingersB);
    const fistA = isFist(fingersA);
    const fistB = isFist(fingersB);
    const peaceA = isPeaceSign(fingersA);
    const peaceB = isPeaceSign(fingersB);

    const centerA: Point2D = handA[0];
    const centerB: Point2D = handB[0];

    if (pinchA && pinchB) {
      // Two-hand pinch: distance between hands drives zoom.
      memory.prevPanCenter = null;
      memory.prevRotateCenter = null;

      const currentDist = dist(centerA, centerB);
      if (memory.prevZoomDist !== null) {
        const delta = currentDist - memory.prevZoomDist;
        if (Math.abs(delta) > ZOOM_DEADZONE) {
          zoomCamera(camera, controls, delta);
        }
      }
      memory.prevZoomDist = currentDist;
      setGestureStatus("Zooming");
    } else if (fistA && fistB) {
      // Two fists: their average movement drives pan.
      memory.prevZoomDist = null;
      memory.prevRotateCenter = null;

      const center: Point2D = {
        x: (centerA.x + centerB.x) / 2,
        y: (centerA.y + centerB.y) / 2,
      };
      if (memory.prevPanCenter) {
        const dx = center.x - memory.prevPanCenter.x;
        const dy = center.y - memory.prevPanCenter.y;
        if (Math.abs(dx) > PAN_DEADZONE || Math.abs(dy) > PAN_DEADZONE) {
          panCamera(camera, controls, dx, dy);
        }
      }
      memory.prevPanCenter = center;
      setGestureStatus("Panning");
    } else if ((fistA && peaceB) || (fistB && peaceA)) {
      // One fist + one peace sign: the peace hand's movement drives rotation.
      memory.prevZoomDist = null;
      memory.prevPanCenter = null;

      const peaceCenter = peaceA ? centerA : centerB;
      if (memory.prevRotateCenter) {
        const dx = peaceCenter.x - memory.prevRotateCenter.x;
        const dy = peaceCenter.y - memory.prevRotateCenter.y;
        if (Math.abs(dx) > ROTATE_DEADZONE || Math.abs(dy) > ROTATE_DEADZONE) {
          rotateCamera(camera, controls, dx, dy);
        }
      }
      memory.prevRotateCenter = { x: peaceCenter.x, y: peaceCenter.y };
      setGestureStatus("Rotating");
    } else {
      memory.prevZoomDist = null;
      memory.prevPanCenter = null;
      memory.prevRotateCenter = null;
      setGestureStatus(
        "Pinch both hands to zoom \u00B7 two fists to pan \u00B7 fist + peace sign to rotate"
      );
    }
  }, []);

  // --- Start / stop webcam + MediaPipe Hands whenever the toggle flips -----
  useEffect(() => {
    if (!handTrackingEnabled) return;

    let cancelled = false;

    async function start() {
      const video = videoRef.current;
      if (!video) return;

      setIsStarting(true);
      setTrackingError(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        video.srcObject = stream;
        await video.play();

        await loadScriptOnce(HANDS_SCRIPT_SRC);

        if (typeof window.Hands !== "function") {
          throw new Error(
            "MediaPipe Hands script loaded but window.Hands was not set. " +
              "Try reloading the page."
          );
        }

        const hands = new window.Hands({
          locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
          selfieMode: true,
        });
        hands.onResults((results) => {
          const rawHands = results.multiHandLandmarks ?? [];
          const handedness = results.multiHandedness ?? [];
          const memory = smoothedHands.current;

          const seen = new Set<HandLabel>();
          const labeled: LabeledHand[] = rawHands.map((landmarks, i) => {
            const label: HandLabel = handedness[i]?.label ?? "Right";
            const smoothed = smoothLandmarkList(memory[label], landmarks);
            memory[label] = smoothed;
            seen.add(label);
            return { label, landmarks: smoothed };
          });

          // Clear smoothing memory for any hand that dropped out of frame so
          // it doesn't glide in from a stale position when it reappears.
          (["Left", "Right"] as const).forEach((label) => {
            if (!seen.has(label)) memory[label] = null;
          });

          const smoothedLandmarks = labeled.map((h) => h.landmarks);
          drawHandOverlay(smoothedLandmarks);
          applyGestures(smoothedLandmarks);
        });
        handsRef.current = hands;

        const loop = async () => {
          if (cancelled) return;
          if (video.readyState >= 2) {
            await hands.send({ image: video });
          }
          rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);

        setIsStarting(false);
      } catch (err) {
        console.error("Hand tracking failed to start:", err);
        if (!cancelled) {
          setTrackingError(
            err instanceof Error
              ? err.message
              : "Could not access the webcam."
          );
          setIsStarting(false);
          setHandTrackingEnabled(false);
        }
      }
    }

    start();

    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;

    return () => {
      cancelled = true;

      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }

      handsRef.current?.close();
      handsRef.current = null;

      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;

      if (video) {
        video.pause();
        video.srcObject = null;
      }

      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

      gestureMemory.current = {
        prevZoomDist: null,
        prevPanCenter: null,
        prevRotateCenter: null,
      };
      smoothedHands.current = { Left: null, Right: null };
      setGestureStatus("");
    };
  }, [handTrackingEnabled, drawHandOverlay, applyGestures]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative" }}
    >
      {/* Webcam feed - never shown, only used as the tracking input source */}
      <video
        ref={videoRef}
        playsInline
        muted
        style={{
          position: "absolute",
          width: 1,
          height: 1,
          opacity: 0,
          pointerEvents: "none",
        }}
      />

      {/* Transparent overlay: only the hand skeleton is drawn here */}
      <canvas
        ref={overlayCanvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "absolute",
          top: 16,
          left: 16,
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: 8,
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setHandTrackingEnabled((v) => !v)}
            disabled={isStarting}
            style={{
              padding: "8px 14px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.2)",
              background: handTrackingEnabled
                ? "rgba(56, 189, 248, 0.9)"
                : "rgba(20, 20, 20, 0.75)",
              color: handTrackingEnabled ? "#0a0a0a" : "#fff",
              fontSize: 14,
              fontWeight: 600,
              cursor: isStarting ? "wait" : "pointer",
              backdropFilter: "blur(6px)",
            }}
          >
            {isStarting
              ? "Requesting camera\u2026"
              : handTrackingEnabled
              ? "Disable Hand Tracking"
              : "Enable Hand Tracking"}
          </button>

          <label
            style={{
              padding: "8px 14px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.2)",
              background: "rgba(20, 20, 20, 0.75)",
              color: "#fff",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
              backdropFilter: "blur(6px)",
            }}
          >
            Load .ply / .splat / .ksplat
            <input
              type="file"
              accept=".ply,.splat,.ksplat"
              onChange={handleFilePicked}
              style={{ display: "none" }}
            />
          </label>
        </div>

        {splatLoading && (
          <div
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(20, 20, 20, 0.65)",
              color: "#e5e5e5",
              fontSize: 12,
              backdropFilter: "blur(6px)",
            }}
          >
            {"Loading splat scene\u2026"}
          </div>
        )}

        {splatError && (
          <div
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(127, 29, 29, 0.85)",
              color: "#fecaca",
              fontSize: 12,
              maxWidth: 320,
            }}
          >
            {splatError}
          </div>
        )}

        {handTrackingEnabled && gestureStatus && (
          <div
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(20, 20, 20, 0.65)",
              color: "#e5e5e5",
              fontSize: 12,
              backdropFilter: "blur(6px)",
              maxWidth: 320,
            }}
          >
            {gestureStatus}
          </div>
        )}

        {trackingError && (
          <div
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(127, 29, 29, 0.85)",
              color: "#fecaca",
              fontSize: 12,
              maxWidth: 320,
            }}
          >
            {trackingError}
          </div>
        )}
      </div>
    </div>
  );
}