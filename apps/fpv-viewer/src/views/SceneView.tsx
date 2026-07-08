import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SCENE_BASE } from "../types";
import { videoHref } from "../App";
import {
  ReadOnlySceneViewer,
  type SceneFrame,
  type SceneTimeline,
} from "../three/sceneViewer";
import { SceneCharts } from "../components/SceneCharts";

type FrameMode = "overlay" | "render" | "actual";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// Nearest VGGT frame to a given flight time.
function nearestFrameIndex(frames: SceneFrame[], t: number): number {
  if (!frames.length) return -1;
  let best = 0;
  for (let i = 1; i < frames.length; i += 1) {
    if (Math.abs(frames[i].t - t) < Math.abs(frames[best].t - t)) best = i;
  }
  return best;
}

// Read-only 3D scene with playback on flight (sequence) time — the pauses that
// exist in the source video are removed, so speed/height charts and the camera
// motion are continuous. The corner panel shows the actual frames that were
// sent to VGGT (actual / rendered / overlay), synced to the same clock.
export function SceneView({ video }: { video: VideoRecord }) {
  const holderRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ReadOnlySceneViewer | null>(null);
  const clockRef = useRef<{ playing: boolean; last: number; t: number }>({
    playing: false,
    last: 0,
    t: 0,
  });
  const lastTRef = useRef(-1);
  const timelineRef = useRef<SceneTimeline | null>(null);
  // Corner frame panel: painted imperatively into a <canvas> from preloaded,
  // decoded images so it stays in lockstep with the 3D during playback. Setting
  // <img src> to a fresh remote URL every animation frame can't fetch/decode
  // fast enough, so the picture used to freeze until playback stopped.
  const frameCanvasRef = useRef<HTMLCanvasElement>(null);
  const framesRef = useRef<SceneFrame[]>([]);
  const frameModeRef = useRef<FrameMode>("overlay");
  const imgCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());

  const [status, setStatus] = useState<string | null>("Loading 3D scene…");
  const [stats, setStats] = useState<{ pointCount: number; frames: number } | null>(null);
  const [timeline, setTimeline] = useState<SceneTimeline | null>(null);
  const [frames, setFrames] = useState<SceneFrame[]>([]);
  const [frameMode, setFrameMode] = useState<FrameMode>("overlay");
  const [currentT, setCurrentT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showFrames, setShowFrames] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [showFrusta, setShowFrusta] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [measureText, setMeasureText] = useState<string | null>(null);
  const variants = video.scenePaths ?? (video.scenePath ? [video.scenePath] : []);
  const [scenePath, setScenePath] = useState<string | null>(video.scenePath);

  useEffect(() => setScenePath(video.scenePath), [video.videoFile, video.scenePath]);

  // Build the 3D viewer
  useEffect(() => {
    if (!scenePath || !holderRef.current) return;
    const viewer = new ReadOnlySceneViewer(holderRef.current);
    viewerRef.current = viewer;
    setStatus("Loading 3D scene…");
    viewer
      .load(`${SCENE_BASE}/${scenePath}/viewer`)
      .then((s) => {
        // StrictMode double-mounts: a disposed viewer resolves with empty
        // results — never let it clobber the live viewer's state.
        if (viewerRef.current !== viewer) return;
        setStats(s);
        const t = viewer.timeline();
        setTimeline(t);
        timelineRef.current = t;
        setFrames(viewer.frames());
        if (t) {
          clockRef.current = { playing: false, last: 0, t: t.t0 };
          setCurrentT(t.t0);
        }
        setStatus(null);
      })
      .catch((e) => {
        if (viewerRef.current !== viewer) return;
        setStatus(`Failed to load scene (${e.message ?? e})`);
      });
    return () => {
      viewerRef.current = null;
      viewer.dispose();
    };
  }, [scenePath]);

  // The frame-panel image URL for a given flight time, in the active mode.
  const srcAt = useCallback((t: number): string | null => {
    const fr = framesRef.current;
    const idx = nearestFrameIndex(fr, t);
    if (idx < 0) return null;
    const f = fr[idx];
    return f[frameModeRef.current] ?? f.actual;
  }, []);

  // Paint a (already-decoded, cached) frame into the canvas. No-op until the
  // image has loaded — the preloader redraws it on load.
  const drawFrame = useCallback((src: string | null) => {
    const canvas = frameCanvasRef.current;
    if (!canvas || !src) return;
    const img = imgCacheRef.current.get(src);
    if (!img || !img.complete || img.naturalWidth === 0) return;
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);
  }, []);

  // Keep the refs the animation loop reads in sync with React state.
  useEffect(() => {
    framesRef.current = frames;
  }, [frames]);
  useEffect(() => {
    frameModeRef.current = frameMode;
  }, [frameMode]);

  // Preload + decode every frame of the active mode so swapping between them is
  // instant, then draw whatever frame is current.
  useEffect(() => {
    const cache = imgCacheRef.current;
    for (const f of frames) {
      const s = f[frameMode] ?? f.actual;
      if (!s || cache.has(s)) continue;
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        if (srcAt(clockRef.current.t) === s) drawFrame(s);
      };
      img.src = s;
      cache.set(s, img);
    }
    drawFrame(srcAt(clockRef.current.t));
  }, [frames, frameMode, srcAt, drawFrame]);

  // Redraw on the discrete cases React drives: scrubbing while paused, switching
  // mode, or re-showing the panel. (Smooth playback is handled in the tick.)
  useEffect(() => {
    drawFrame(srcAt(clockRef.current.t));
  }, [currentT, frameMode, showFrames, srcAt, drawFrame]);

  // Playback clock: internal accumulator over flight time.
  useEffect(() => {
    let handle = 0;
    const tick = (now: number) => {
      handle = requestAnimationFrame(tick);
      const c = clockRef.current;
      const tl = timelineRef.current;
      if (!tl || !c.playing) return;
      const dt = (now - c.last) / 1000;
      c.last = now;
      c.t = Math.min(c.t + dt, tl.t1);
      if (c.t >= tl.t1) {
        c.playing = false;
        setPlaying(false);
      }
      if (Math.abs(c.t - lastTRef.current) > 0.02) {
        lastTRef.current = c.t;
        viewerRef.current?.setTime(c.t);
        drawFrame(srcAt(c.t));
        setCurrentT(c.t);
      }
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [drawFrame, srcAt]);

  useEffect(() => viewerRef.current?.setPathVisible(showPath), [showPath]);
  useEffect(() => viewerRef.current?.setGridVisible(showGrid), [showGrid]);
  useEffect(() => viewerRef.current?.setPointsVisible(showPoints), [showPoints]);
  useEffect(() => viewerRef.current?.setFrustaVisible(showFrusta), [showFrusta]);

  const seek = (t: number) => {
    clockRef.current.t = t;
    lastTRef.current = t;
    viewerRef.current?.setTime(t);
    drawFrame(srcAt(t));
    setCurrentT(t);
  };

  const togglePlay = () => {
    const c = clockRef.current;
    const tl = timelineRef.current;
    if (!tl) return;
    if (!c.playing && c.t >= tl.t1 - 0.05) c.t = tl.t0; // replay from start
    c.playing = !c.playing;
    c.last = performance.now();
    setPlaying(c.playing);
  };

  // Spacebar plays/pauses (viewing shortcut, matches the video page).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const tag = (document.activeElement as HTMLElement | null)?.tagName ?? "";
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
      e.preventDefault();
      togglePlay();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleMeasure = () => {
    const next = !measuring;
    setMeasuring(next);
    setMeasureText(next ? "Click two points in the scene" : null);
    viewerRef.current?.setMeasureMode(next, (meters) => {
      if (meters !== null) setMeasureText(`Distance: ${meters.toFixed(1)} m`);
      else if (next) setMeasureText("Click two points in the scene");
    });
  };

  const toggleFullscreen = () => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen().catch(() => undefined);
  };

  // Current VGGT frame (nearest by flight time). Used for the counter and to
  // decide whether the panel has anything to show; the picture itself is painted
  // into the canvas imperatively.
  const frameIndex = useMemo(() => nearestFrameIndex(frames, currentT), [frames, currentT]);
  const frameSrc =
    frameIndex >= 0 ? frames[frameIndex][frameMode] ?? frames[frameIndex].actual : null;

  if (!video.scenePath) {
    return (
      <div>
        <a className="back-link" href="#/">
          ← All videos
        </a>
        <p className="not-found">No 3D scene for this video.</p>
      </div>
    );
  }

  return (
    <div className="scene-view">
      <a className="back-link" href="#/">
        ← All videos
      </a>
      <h1>{video.description || video.videoFile}</h1>
      <p className="view-meta">
        {video.date}
        {video.town ? ` · ${video.town}` : ""} · 3D reconstruction
        {stats ? ` · ${stats.pointCount.toLocaleString()} points · ${stats.frames} camera poses` : ""}
      </p>

      <div className="scene-stage" ref={stageRef}>
        <div className="canvas-holder" ref={holderRef} />
        {status ? <div className="scene-status">{status}</div> : null}
        {showFrames && frameSrc ? (
          <div className="scene-frame-panel">
            <div className="frame-tabs">
              {(["overlay", "render", "actual"] as FrameMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={frameMode === mode ? "active" : ""}
                  onClick={() => setFrameMode(mode)}
                >
                  {mode === "overlay" ? "Overlay" : mode === "render" ? "Render" : "Actual"}
                </button>
              ))}
              <span className="frame-counter">
                {frameIndex + 1}/{frames.length}
              </span>
            </div>
            <canvas
              ref={frameCanvasRef}
              aria-label={`VGGT ${frameMode} frame ${frameIndex + 1}`}
            />
          </div>
        ) : null}
        <button
          type="button"
          className="stage-btn fullscreen-btn"
          onClick={toggleFullscreen}
          title="Fullscreen"
        >
          ⛶
        </button>
      </div>

      {timeline ? (
        <div className="scene-controls">
          <button
            type="button"
            className="play-btn"
            onClick={togglePlay}
            aria-label={playing ? "Pause" : "Play"}
            title="Play/pause (space)"
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <input
            type="range"
            className="scrubber"
            min={timeline.t0}
            max={timeline.t1}
            step={0.01}
            value={Math.min(Math.max(currentT, timeline.t0), timeline.t1)}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Timeline"
          />
          <span className="time-display">
            {fmt(Math.max(currentT - timeline.t0, 0))} / {fmt(timeline.t1 - timeline.t0)}
          </span>
          <span className="scene-toggles">
            <label>
              <input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} />
              Points
            </label>
            <label>
              <input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />
              Path
            </label>
            <label>
              <input type="checkbox" checked={showFrusta} onChange={(e) => setShowFrusta(e.target.checked)} />
              Cameras
            </label>
            <label>
              <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
              Grid
            </label>
            <label>
              <input type="checkbox" checked={showFrames} onChange={(e) => setShowFrames(e.target.checked)} />
              Frames
            </label>
            {variants.length > 1 ? (
              <select
                className="variant-select"
                value={scenePath ?? ""}
                onChange={(e) => setScenePath(e.target.value)}
                aria-label="Scene variant"
                title="Point-cloud variant"
              >
                {variants.map((p) => {
                  const name = p.split("/").pop() ?? p;
                  const label = name.includes("__") ? name.split("__").pop()! : "base";
                  return (
                    <option key={p} value={p}>
                      {label}
                    </option>
                  );
                })}
              </select>
            ) : null}
            <button
              type="button"
              className={`measure-btn${measuring ? " active" : ""}`}
              onClick={toggleMeasure}
              title="Measure the distance between two points"
            >
              Measure
            </button>
          </span>
        </div>
      ) : null}
      {measureText ? <p className="measure-readout">{measureText}</p> : null}

      {timeline ? <SceneCharts timeline={timeline} currentT={currentT} onSeek={seek} /> : null}

      <div className="view-actions" style={{ marginTop: 14 }}>
        <a href={videoHref(video.videoFile)}>← Watch the full video with annotations</a>
      </div>
    </div>
  );
}
