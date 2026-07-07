export type SegmentMarker = {
  time: number;
  type: string;
  comment?: string;
};

export type VideoRecord = {
  videoFile: string;
  slug: string;
  date: string;
  description: string;
  town: string;
  videoUrl: string;
  thumbnailUrl: string;
  thumbWidths: number[] | null;
  blur: string | null;
  scenePath: string | null; // "<stem>/<sceneId>" under the scenes base
  scenePaths: string[] | null; // all scene variants for this video (canonical first)
  segments: SegmentMarker[] | null;
  annotationAuto: boolean | null;
};

export type Dataset = {
  generated_at: string;
  videos: VideoRecord[];
};

// Bases are compile-time env for production (CloudFront); the dev server
// proxies the local paths instead.
export const SCENE_BASE = (import.meta.env.VITE_SCENE_BASE ?? "/scenes").replace(/\/$/, "");
export const THUMB_BASE = (import.meta.env.VITE_THUMB_BASE ?? "/thumbnails").replace(/\/$/, "");

// Segment marker colors — same legend as the annotator.
export const SEGMENT_TYPES: Record<string, { label: string; color: string }> = {
  banner_start: { label: "Banner", color: "#4a9eff" },
  flight_start: { label: "Flight start", color: "#20c56b" },
  new_flight_start: { label: "New flight", color: "#00bcd4" },
  pause_start: { label: "Pause", color: "#f59e0b" },
  replay_start: { label: "Replay", color: "#9e77ed" },
  end: { label: "End", color: "#ef4444" },
  video_end: { label: "Video end", color: "#ef4444" },
  other: { label: "Other", color: "#8fa0ad" },
};
