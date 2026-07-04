"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { VideoCard } from "@/components/video-card";
import type { VideoRecord } from "@/lib/videos";

type SceneFilter = "all" | "with" | "without";
type SortDir = "desc" | "asc";

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
        <div className="relative min-w-0 flex-1 sm:max-w-sm">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by description, town, date…"
            aria-label="Search videos"
            className="pl-8"
          />
        </div>

        <Select value={sceneFilter} onValueChange={(v) => setSceneFilter(v as SceneFilter)}>
          <SelectTrigger className="w-[168px]" aria-label="Filter by scene">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All videos</SelectItem>
            <SelectItem value="with">With 3D scene</SelectItem>
            <SelectItem value="without">Without scene</SelectItem>
          </SelectContent>
        </Select>

        <Button
          type="button"
          variant="secondary"
          onClick={() => setSort((s) => (s === "desc" ? "asc" : "desc"))}
          aria-label={`Sort by date ${sort === "desc" ? "descending" : "ascending"}`}
        >
          Date
          {sort === "desc" ? <ArrowDown /> : <ArrowUp />}
        </Button>

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
