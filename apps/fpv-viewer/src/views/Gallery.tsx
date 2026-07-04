import { useMemo, useState } from "react";
import type { VideoRecord } from "../types";
import { THUMB_BASE } from "../types";
import { sceneHref, videoHref } from "../App";

function Thumb({ video }: { video: VideoRecord }) {
  const [broken, setBroken] = useState(false);
  const useLocal = Boolean(video.thumbWidths?.length) && !broken;
  const src = useLocal
    ? `${THUMB_BASE}/${video.slug}/${video.thumbWidths![video.thumbWidths!.length - 1]}.webp`
    : video.thumbnailUrl;
  const srcSet = useLocal
    ? video.thumbWidths!.map((w) => `${THUMB_BASE}/${video.slug}/${w}.webp ${w}w`).join(", ")
    : undefined;
  return (
    <a
      className="thumb"
      href={videoHref(video.videoFile)}
      style={video.blur ? { backgroundImage: `url(${video.blur})` } : undefined}
    >
      {src ? (
        <img
          src={src}
          srcSet={srcSet}
          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 280px"
          alt={video.description}
          loading="lazy"
          decoding="async"
          onError={() => setBroken(true)}
        />
      ) : null}
      {video.scenePath ? <span className="badge-3d">3D scene</span> : null}
    </a>
  );
}

export function Gallery({ videos }: { videos: VideoRecord[] }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return videos;
    return videos.filter((v) =>
      `${v.description} ${v.town} ${v.date} ${v.videoFile}`.toLowerCase().includes(q),
    );
  }, [videos, query]);

  const withScene = videos.filter((v) => v.scenePath).length;

  return (
    <div>
      <h1>FPV strike videos</h1>
      <p className="page-lede">
        {videos.length} videos from southern Lebanon · {withScene} with a reconstructed 3D scene.
        Open a video to see its flight annotations, or explore the 3D reconstruction where one
        exists.
      </p>
      <div className="gallery-toolbar">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by description, town, date…"
          aria-label="Search videos"
        />
        <span className="gallery-count">
          {filtered.length === videos.length
            ? `${videos.length} videos`
            : `${filtered.length} of ${videos.length}`}
        </span>
      </div>
      <div className="video-grid">
        {filtered.map((v) => (
          <div className="video-card" key={v.videoFile}>
            <Thumb video={v} />
            <a className="title" href={videoHref(v.videoFile)}>
              {v.description || v.videoFile}
            </a>
            <span className="meta">
              {v.date}
              {v.town ? ` · ${v.town}` : ""}
            </span>
            <span className="links">
              <a href={videoHref(v.videoFile)}>Video</a>
              {v.scenePath ? (
                <a href={sceneHref(v.videoFile)}>3D scene</a>
              ) : (
                <span className="disabled">No scene</span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
