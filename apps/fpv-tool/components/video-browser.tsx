"use client";

import { useMemo, useState } from "react";
import { VideoCard } from "@/components/video-card";
import type { VideoRecord } from "@/lib/videos";

type SceneFilter = "all" | "with" | "without";
type SortDir = "desc" | "asc";

const controlClass =
  "h-9 rounded-md border border-input bg-secondary px-3 text-[13px] text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring";

export function VideoBrowser({ videos }: { videos: VideoRecord[] }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortDir>("desc");
  const [sceneFilter, setSceneFilter] = useState<SceneFilter>("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = videos.filter((v) => {
      if (sceneFilter === "with" && !v.sceneId) return false;
      if (sceneFilter === "without" && v.sceneId) return false;
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
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by description, town, date…"
          className={`${controlClass} min-w-0 flex-1 sm:max-w-sm`}
          aria-label="Search videos"
        />
        <select
          value={sceneFilter}
          onChange={(e) => setSceneFilter(e.target.value as SceneFilter)}
          className={controlClass}
          aria-label="Filter by scene"
        >
          <option value="all">All videos</option>
          <option value="with">With 3D scene</option>
          <option value="without">Without scene</option>
        </select>
        <button
          type="button"
          onClick={() => setSort((s) => (s === "desc" ? "asc" : "desc"))}
          className={`${controlClass} inline-flex items-center gap-1.5 hover:text-foreground`}
          aria-label={`Sort by date ${sort === "desc" ? "descending" : "ascending"}`}
        >
          Date
          <span aria-hidden className="text-muted-foreground">
            {sort === "desc" ? "↓" : "↑"}
          </span>
        </button>
        <span className="ml-auto text-[13px] text-muted-foreground">
          {filtered.length === videos.length
            ? `${videos.length} videos`
            : `${filtered.length} of ${videos.length}`}
        </span>
      </div>

      {filtered.length === 0 ? (
        <p className="py-16 text-center text-sm text-muted-foreground">No videos match your filters.</p>
      ) : (
        <div className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {filtered.map((video) => (
            <VideoCard key={video.videoFile} video={video} />
          ))}
        </div>
      )}
    </div>
  );
}
