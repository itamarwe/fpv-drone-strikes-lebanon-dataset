import { useEffect, useMemo, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SEGMENT_TYPES } from "../types";
import { sceneHref } from "../App";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

export function VideoView({ video }: { video: VideoRecord }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    setDuration(0);
    setCurrentTime(0);
  }, [video.videoFile]);

  const segments = useMemo(
    () => (video.segments ?? []).slice().sort((a, b) => a.time - b.time),
    [video.segments],
  );

  const usedTypes = useMemo(() => {
    const types = new Set(segments.map((s) => s.type));
    return Object.entries(SEGMENT_TYPES).filter(([key]) => types.has(key));
  }, [segments]);

  const seek = (t: number) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, Math.min(t, el.duration || t));
    el.play().catch(() => undefined);
  };

  const onTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - rect.left) / rect.width) * duration);
  };

  return (
    <div className="video-view">
      <a className="back-link" href="#/">
        ← All videos
      </a>
      <h1 style={{ textTransform: "capitalize" }}>{video.description || video.videoFile}</h1>
      <p className="view-meta">
        {video.date}
        {video.town ? ` · ${video.town}` : ""}
      </p>

      <video
        ref={videoRef}
        src={video.videoUrl}
        controls
        playsInline
        preload="metadata"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
      />

      <div className="view-actions">
        {video.scenePath ? <a href={sceneHref(video.videoFile)}>View 3D scene →</a> : null}
        <a href={video.videoUrl} rel="noreferrer">
          Download video
        </a>
      </div>

      {segments.length ? (
        <section>
          <h2 style={{ fontSize: 18 }}>Flight annotations</h2>
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
                        seek(s.time);
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
