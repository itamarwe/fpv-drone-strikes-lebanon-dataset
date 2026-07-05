import { useEffect, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SCENE_BASE } from "../types";
import { videoHref } from "../App";
import { ReadOnlySceneViewer, type SceneTimeline } from "../three/sceneViewer";
import { SceneCharts } from "../components/SceneCharts";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// Read-only 3D scene with synced playback: the source video (corner overlay)
// is the master clock; the camera marker in the scene and the playhead in the
// height/speed chart follow it. Scrub with the slider or by dragging on the
// chart.
export function SceneView({ video }: { video: VideoRecord }) {
  const holderRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const viewerRef = useRef<ReadOnlySceneViewer | null>(null);
  const internalClockRef = useRef<{ playing: boolean; last: number } | null>(null);
  const lastTRef = useRef(-1);

  const [status, setStatus] = useState<string | null>("Loading 3D scene…");
  const [stats, setStats] = useState<{ pointCount: number; frames: number } | null>(null);
  const [timeline, setTimeline] = useState<SceneTimeline | null>(null);
  const [currentT, setCurrentT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoBroken, setVideoBroken] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showVideo, setShowVideo] = useState(true);

  // Build the 3D viewer
  useEffect(() => {
    if (!video.scenePath || !holderRef.current) return;
    const viewer = new ReadOnlySceneViewer(holderRef.current);
    viewerRef.current = viewer;
    setStatus("Loading 3D scene…");
    viewer
      .load(`${SCENE_BASE}/${video.scenePath}/viewer`)
      .then((s) => {
        setStats(s);
        const t = viewer.timeline();
        setTimeline(t);
        if (t) {
          setCurrentT(t.t0);
          const el = videoRef.current;
          if (el) el.currentTime = t.t0;
        }
        setStatus(null);
      })
      .catch((e) => setStatus(`Failed to load scene (${e.message ?? e})`));
    return () => {
      viewerRef.current = null;
      viewer.dispose();
    };
  }, [video.scenePath]);

  // Playback clock: read the video's currentTime every frame; if the video
  // failed to load, fall back to an internal wall clock.
  useEffect(() => {
    let handle = 0;
    const tick = (now: number) => {
      handle = requestAnimationFrame(tick);
      let t: number | null = null;
      const el = videoRef.current;
      if (!videoBroken && el && el.readyState >= 1) {
        t = el.currentTime;
      } else if (internalClockRef.current?.playing && timeline) {
        const c = internalClockRef.current;
        const dt = (now - c.last) / 1000;
        c.last = now;
        t = Math.min(lastTRef.current < 0 ? timeline.t0 : lastTRef.current + dt, timeline.t1);
      }
      if (t === null) return;
      if (Math.abs(t - lastTRef.current) > 0.02) {
        lastTRef.current = t;
        viewerRef.current?.setTime(t);
        setCurrentT(t);
      }
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [videoBroken, timeline]);

  useEffect(() => viewerRef.current?.setPathVisible(showPath), [showPath]);
  useEffect(() => viewerRef.current?.setGridVisible(showGrid), [showGrid]);

  const seek = (t: number) => {
    const el = videoRef.current;
    if (!videoBroken && el) el.currentTime = t;
    lastTRef.current = t;
    viewerRef.current?.setTime(t);
    setCurrentT(t);
  };

  const togglePlay = () => {
    const el = videoRef.current;
    if (!videoBroken && el) {
      if (el.paused) void el.play().catch(() => setVideoBroken(true));
      else el.pause();
    } else {
      const c = internalClockRef.current ?? { playing: false, last: performance.now() };
      c.playing = !c.playing;
      c.last = performance.now();
      internalClockRef.current = c;
      setPlaying(c.playing);
    }
  };

  const toggleFullscreen = () => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen().catch(() => undefined);
  };

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
        <video
          ref={videoRef}
          className="scene-pip"
          style={{ display: showVideo && !videoBroken ? "block" : "none" }}
          src={video.videoUrl}
          muted
          playsInline
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() => setVideoBroken(true)}
        />
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
              <input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />
              Path
            </label>
            <label>
              <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
              Grid
            </label>
            <label>
              <input
                type="checkbox"
                checked={showVideo}
                disabled={videoBroken}
                onChange={(e) => setShowVideo(e.target.checked)}
              />
              Video
            </label>
          </span>
        </div>
      ) : null}

      {timeline ? <SceneCharts timeline={timeline} currentT={currentT} onSeek={seek} /> : null}

      <div className="view-actions" style={{ marginTop: 14 }}>
        <a href={videoHref(video.videoFile)}>← Watch the full video with annotations</a>
      </div>
    </div>
  );
}
