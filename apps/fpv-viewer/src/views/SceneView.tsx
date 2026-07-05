import { useEffect, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SCENE_BASE, THUMB_BASE } from "../types";
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
  const internalClockRef = useRef<{ playing: boolean; last: number; t: number } | null>(null);
  const lastTRef = useRef(-1);
  const timelineRef = useRef<SceneTimeline | null>(null);

  const [status, setStatus] = useState<string | null>("Loading 3D scene…");
  const [stats, setStats] = useState<{ pointCount: number; frames: number } | null>(null);
  const [timeline, setTimeline] = useState<SceneTimeline | null>(null);
  const [currentT, setCurrentT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoBroken, setVideoBroken] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showVideo, setShowVideo] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [showFrusta, setShowFrusta] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [measureText, setMeasureText] = useState<string | null>(null);

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
        timelineRef.current = t;
        if (t) {
          setCurrentT(t.t0);
          const el = videoRef.current;
          // Seeks are dropped before metadata is loaded; onLoadedMetadata
          // repeats this for the case where the video is slower than the scene.
          if (el && el.readyState >= 1) el.currentTime = t.t0;
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
        const tl = timelineRef.current;
        if (tl) {
          // Keep the video inside the scene's time window: snap back to the
          // start if a pre-metadata seek was dropped, pause at the scene end.
          if (t < tl.t0 - 0.3) {
            el.currentTime = tl.t0;
            t = tl.t0;
          } else if (t > tl.t1 + 0.05 && !el.paused) {
            el.pause();
            t = tl.t1;
          }
        }
      } else if (internalClockRef.current?.playing && timeline) {
        // The clock keeps its own accumulator; lastTRef is only the render
        // threshold and must not feed back into timekeeping.
        const c = internalClockRef.current;
        const dt = (now - c.last) / 1000;
        c.last = now;
        c.t = Math.min(c.t + dt, timeline.t1);
        t = c.t;
        if (t >= timeline.t1) {
          c.playing = false;
          setPlaying(false);
        }
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
  useEffect(() => viewerRef.current?.setPointsVisible(showPoints), [showPoints]);
  useEffect(() => viewerRef.current?.setFrustaVisible(showFrusta), [showFrusta]);

  const toggleMeasure = () => {
    const next = !measuring;
    setMeasuring(next);
    setMeasureText(next ? "Click two points in the scene" : null);
    viewerRef.current?.setMeasureMode(next, (meters) => {
      if (meters !== null) setMeasureText(`Distance: ${meters.toFixed(1)} m`);
      else if (next) setMeasureText("Click two points in the scene");
    });
  };

  const seek = (t: number) => {
    const el = videoRef.current;
    if (!videoBroken && el) el.currentTime = t;
    if (internalClockRef.current) internalClockRef.current.t = t;
    lastTRef.current = t;
    viewerRef.current?.setTime(t);
    setCurrentT(t);
  };

  const markVideoBroken = () => {
    // Video can't be used as the clock (e.g. missing from the CDN, no error
    // event fired) -> hide the PiP and keep playing on the internal clock.
    setVideoBroken(true);
    internalClockRef.current = {
      playing: true,
      last: performance.now(),
      t: lastTRef.current >= 0 ? lastTRef.current : timelineRef.current?.t0 ?? 0,
    };
    setPlaying(true);
  };

  const togglePlay = () => {
    const el = videoRef.current;
    if (!videoBroken && el) {
      if (el.paused) {
        const tl = timelineRef.current;
        // Restart from the scene start when outside the scene window.
        if (tl && (el.currentTime < tl.t0 - 0.3 || el.currentTime >= tl.t1 - 0.05)) {
          el.currentTime = tl.t0;
        }
        void el.play().catch(markVideoBroken);
        // Some failures (403 from the CDN) never fire an error event: if no
        // data has arrived shortly after pressing play, fall back.
        window.setTimeout(() => {
          const now = videoRef.current;
          if (now && !now.paused && now.readyState < 2) markVideoBroken();
        }, 2500);
      } else el.pause();
    } else {
      const c = internalClockRef.current ?? {
        playing: false,
        last: performance.now(),
        t: timelineRef.current?.t0 ?? 0,
      };
      const tl = timelineRef.current;
      if (!c.playing && tl && c.t >= tl.t1 - 0.05) c.t = tl.t0; // replay from start
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
          poster={
            video.thumbWidths?.length
              ? `${THUMB_BASE}/${video.slug}/${video.thumbWidths[video.thumbWidths.length - 1]}.webp`
              : video.thumbnailUrl || undefined
          }
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={(e) => {
            const tl = timelineRef.current;
            if (tl) e.currentTarget.currentTime = tl.t0;
          }}
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
              <input
                type="checkbox"
                checked={showVideo}
                disabled={videoBroken}
                onChange={(e) => setShowVideo(e.target.checked)}
              />
              Video
            </label>
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
