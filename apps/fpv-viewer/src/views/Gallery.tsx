import { useMemo, useState } from "react";
import type { VideoRecord } from "../types";
import { THUMB_BASE } from "../types";
import { sceneHref, videoHref } from "../App";

const PlayIcon = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden>
    <path d="M8 5.14v13.72L19 12 8 5.14z" />
  </svg>
);

const GlobeIcon = () => (
  <svg
    viewBox="0 0 24 24"
    width="14"
    height="14"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    aria-hidden
  >
    <circle cx="12" cy="12" r="9" />
    <ellipse cx="12" cy="12" rx="4" ry="9" />
    <path d="M3.5 9h17M3.5 15h17" />
  </svg>
);

const ArrowIcon = ({ dir }: { dir: "asc" | "desc" }) => (
  <svg
    viewBox="0 0 24 24"
    width="13"
    height="13"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    aria-hidden
  >
    {dir === "desc" ? <path d="M12 4v16m0 0l-6-6m6 6l6-6" /> : <path d="M12 20V4m0 0l-6 6m6-6l6 6" />}
  </svg>
);

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

type SceneFilter = "all" | "with";
type SortDir = "desc" | "asc";

export function Gallery({ videos }: { videos: VideoRecord[] }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortDir>("desc");
  const [sceneFilter, setSceneFilter] = useState<SceneFilter>("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = videos.filter((v) => {
      if (sceneFilter === "with" && !v.scenePath) return false;
      if (!q) return true;
      return `${v.description} ${v.town} ${v.date} ${v.videoFile}`.toLowerCase().includes(q);
    });
    list.sort((a, b) => {
      const cmp = a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
      return sort === "desc" ? -cmp : cmp;
    });
    return list;
  }, [videos, query, sort, sceneFilter]);

  return (
    <div>
      <div className="gallery-toolbar">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by description, town, date…"
          aria-label="Search videos"
        />
        <button
          type="button"
          className="toolbar-btn"
          onClick={() => setSort((s) => (s === "desc" ? "asc" : "desc"))}
          aria-label={`Sort by date, ${sort === "desc" ? "newest" : "oldest"} first`}
          title={sort === "desc" ? "Newest first" : "Oldest first"}
        >
          Date <ArrowIcon dir={sort} />
        </button>
        <button
          type="button"
          className={`toolbar-btn${sceneFilter === "with" ? " active" : ""}`}
          onClick={() => setSceneFilter((f) => (f === "with" ? "all" : "with"))}
          aria-pressed={sceneFilter === "with"}
          title="Only videos with a reconstructed 3D scene"
        >
          <GlobeIcon /> 3D scenes
        </button>
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
            <div className="card-actions">
              <a className="card-btn" href={videoHref(v.videoFile)}>
                <PlayIcon /> Video
              </a>
              {v.scenePath ? (
                <a className="card-btn primary" href={sceneHref(v.videoFile)}>
                  <GlobeIcon /> 3D scene
                </a>
              ) : (
                <span className="card-btn disabled" title="No reconstructed scene yet">
                  <GlobeIcon /> No scene
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      {filtered.length === 0 ? <p className="not-found">No videos match your filters.</p> : null}
    </div>
  );
}
