import { useEffect, useRef, useState } from "react";
import type { VideoRecord } from "../types";
import { SCENE_BASE } from "../types";
import { videoHref } from "../App";
import { ReadOnlySceneViewer } from "../three/sceneViewer";

export function SceneView({ video }: { video: VideoRecord }) {
  const holderRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ReadOnlySceneViewer | null>(null);
  const [status, setStatus] = useState<string | null>("Loading 3D scene…");
  const [stats, setStats] = useState<{ pointCount: number; frames: number } | null>(null);
  const [pointScale, setPointScale] = useState(1);

  useEffect(() => {
    if (!video.scenePath || !holderRef.current) return;
    const viewer = new ReadOnlySceneViewer(holderRef.current);
    viewerRef.current = viewer;
    setStatus("Loading 3D scene…");
    viewer
      .load(`${SCENE_BASE}/${video.scenePath}/viewer`)
      .then((s) => {
        setStats(s);
        setStatus(null);
      })
      .catch((e) => setStatus(`Failed to load scene (${e.message ?? e})`));
    return () => {
      viewerRef.current = null;
      viewer.dispose();
    };
  }, [video.scenePath]);

  useEffect(() => {
    viewerRef.current?.setPointScale(pointScale);
  }, [pointScale]);

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
      <h1 style={{ textTransform: "capitalize" }}>{video.description || video.videoFile}</h1>
      <p className="view-meta">
        {video.date}
        {video.town ? ` · ${video.town}` : ""} · 3D reconstruction
        {stats ? ` · ${stats.pointCount.toLocaleString()} points · ${stats.frames} camera poses` : ""}
      </p>
      <div className="canvas-holder" ref={holderRef}>
        {status ? <div className="scene-status">{status}</div> : null}
        {!status ? (
          <div className="scene-hud">
            <span>Point size</span>
            <input
              type="range"
              min="0.3"
              max="3"
              step="0.1"
              value={pointScale}
              onChange={(e) => setPointScale(Number(e.target.value))}
            />
          </div>
        ) : null}
      </div>
      <div className="view-actions" style={{ marginTop: 14 }}>
        <a href={videoHref(video.videoFile)}>← Watch the video</a>
      </div>
    </div>
  );
}
