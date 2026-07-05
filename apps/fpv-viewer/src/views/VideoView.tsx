import { useEffect, useMemo, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SEGMENT_TYPES } from "../types";
import { sceneHref } from "../App";

const FPS = 30; // frame-step size for ,/. and the ±1f buttons

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// YouTube-mobile-style layout: the player comes first, then a compact
// title/meta block, actions, and the read-only flight annotations.
export function VideoView({ video }: { video: VideoRecord }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);

  useEffect(() => {
    setDuration(0);
    setCurrentTime(0);
    setPlaying(false);
  }, [video.videoFile]);

  const segments = useMemo(
    () => (video.segments ?? []).slice().sort((a, b) => a.time - b.time),
    [video.segments],
  );

  const usedTypes = useMemo(() => {
    const types = new Set(segments.map((s) => s.type));
    return Object.entries(SEGMENT_TYPES).filter(([key]) => types.has(key));
  }, [segments]);

  const seek = (t: number, andPlay = false) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, Math.min(t, el.duration || t));
    if (andPlay) void el.play().catch(() => undefined);
  };

  const step = (deltaS: number) => {
    const el = videoRef.current;
    if (!el) return;
    el.pause();
    seek(el.currentTime + deltaS);
  };

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play().catch(() => undefined);
    else el.pause();
  };

  const toggleMute = () => {
    const el = videoRef.current;
    if (!el) return;
    el.muted = !el.muted;
    setMuted(el.muted);
  };

  const toggleFullscreen = () => {
    const el = videoRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen().catch(() => undefined);
  };

  // Keyboard shortcuts (viewing only): space play/pause, arrows ±5s,
  // , / . frame step, m mute, f fullscreen, Home/End.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement as HTMLElement | null)?.tagName ?? "";
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowLeft":
          e.preventDefault();
          seek((videoRef.current?.currentTime ?? 0) - 5);
          break;
        case "ArrowRight":
          e.preventDefault();
          seek((videoRef.current?.currentTime ?? 0) + 5);
          break;
        case ",":
          step(-1 / FPS);
          break;
        case ".":
          step(1 / FPS);
          break;
        case "m":
          toggleMute();
          break;
        case "f":
          toggleFullscreen();
          break;
        case "Home":
          e.preventDefault();
          seek(0);
          break;
        case "End":
          e.preventDefault();
          seek(duration);
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration]);

  const onTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - rect.left) / rect.width) * duration, true);
  };

  return (
    <div className="video-view">
      <a className="back-link" href="#/">
        ← All videos
      </a>
      <video
        ref={videoRef}
        src={video.videoUrl}
        playsInline
        preload="metadata"
        onClick={togglePlay}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />

      <div className="transport">
        <div className="t-row t-primary">
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
            min={0}
            max={duration || 1}
            step={0.01}
            value={Math.min(currentTime, duration || 1)}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Seek"
          />
          <span className="time-display">
            {fmt(currentTime)} / {fmt(duration)}
          </span>
        </div>
        <div className="t-row t-secondary">
          <button type="button" onClick={() => step(-10)} title="Back 10s">
            -10s
          </button>
          <button type="button" onClick={() => step(-1 / FPS)} title="Previous frame (,)">
            ‹f
          </button>
          <button type="button" onClick={() => step(1 / FPS)} title="Next frame (.)">
            f›
          </button>
          <button type="button" onClick={() => step(10)} title="Forward 10s">
            +10s
          </button>
          <button type="button" onClick={toggleMute} title="Mute (m)">
            {muted ? "🔇" : "🔊"}
          </button>
          <button type="button" onClick={toggleFullscreen} title="Fullscreen (f)">
            ⛶
          </button>
        </div>
      </div>

      <div className="video-headline">
        <h1>{video.description || video.videoFile}</h1>
        <p className="view-meta">
          {video.date}
          {video.town ? ` · ${video.town}` : ""}
        </p>
        <div className="view-actions">
          {video.scenePath ? <a href={sceneHref(video.videoFile)}>View 3D scene →</a> : null}
          <a
            className="icon-link"
            href={video.videoUrl}
            rel="noreferrer"
            title="Download video"
            aria-label="Download video"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M12 4v11m0 0l-5-5m5 5l5-5" />
              <path d="M4 19h16" />
            </svg>
          </a>
        </div>
      </div>

      {segments.length ? (
        <section>
          <h2 style={{ fontSize: 17 }}>Flight annotations</h2>
          {duration > 0 ? (
            <>
              <div className="timeline" onClick={onTimelineClick} title="Click to seek">
                {segments.map((s, i) => (
                  <span
                    key={i}
                    className="marker"
                    style={{
                      left: `${(s.time / duration) * 100}%`,
                      background: SEGMENT_TYPES[s.type]?.color ?? "#8fa0ad",
                    }}
                  />
                ))}
                <span className="playhead" style={{ left: `${(currentTime / duration) * 100}%` }} />
              </div>
              <div className="legend">
                {usedTypes.map(([key, def]) => (
                  <span key={key} className="chip" style={{ "--chip-color": def.color } as React.CSSProperties}>
                    {def.label}
                  </span>
                ))}
              </div>
            </>
          ) : null}
          <table className="segment-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((s, i) => (
                <tr key={i}>
                  <td>
                    <a
                      className="time-link"
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        seek(s.time, true);
                      }}
                    >
                      {fmt(s.time)}
                    </a>
                  </td>
                  <td>
                    <span
                      className="segment-type"
                      style={{ "--chip-color": SEGMENT_TYPES[s.type]?.color ?? "#8fa0ad" } as React.CSSProperties}
                    >
                      {SEGMENT_TYPES[s.type]?.label ?? s.type}
                    </span>
                  </td>
                  <td style={{ color: "var(--grey)" }}>{s.comment ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {video.annotationAuto ? (
            <p className="annotation-note">Annotations were generated automatically.</p>
          ) : null}
        </section>
      ) : (
        <p className="annotation-note">No annotations for this video yet.</p>
      )}
    </div>
  );
}
