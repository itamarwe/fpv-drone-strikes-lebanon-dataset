import { Button } from "@/components/ui/button";
import { thumbBase, withBasePath } from "@/lib/urls";
import type { VideoRecord } from "@/lib/videos";

// Matches the responsive grid: 1 col on phones, 2 on small, 3 on large, 4 on xl.
const SIZES = "(max-width: 640px) 100vw, (max-width: 1024px) 50vw, (max-width: 1280px) 33vw, 320px";

function thumbSources(video: VideoRecord) {
  if (video.thumbWidths && video.thumbWidths.length) {
    const base = thumbBase ? `${thumbBase}/${video.slug}` : withBasePath(`/thumbnails/${video.slug}`);
    const srcSet = video.thumbWidths.map((w) => `${base}/${w}.webp ${w}w`).join(", ");
    const fallback = `${base}/${video.thumbWidths[video.thumbWidths.length - 1]}.webp`;
    return { src: fallback, srcSet };
  }
  // No locally generated thumbnails yet -> use the remote source directly.
  return { src: video.thumbnailUrl, srcSet: undefined };
}

export function VideoCard({ video }: { video: VideoRecord }) {
  const annotateHref = withBasePath(`/annotate/?v=${encodeURIComponent(video.videoFile)}`);
  const viewerHref = video.sceneId
    ? withBasePath(`/viewer/${encodeURIComponent(video.sceneId)}/`)
    : null;
  const primaryHref = viewerHref ?? annotateHref;
  const { src, srcSet } = thumbSources(video);

  return (
    <div className="group flex flex-col gap-2.5">
      <a
        href={primaryHref}
        className="relative block aspect-video overflow-hidden rounded-xl bg-secondary ring-1 ring-border/60 transition group-hover:ring-ring/50"
        style={
          video.blurDataURL
            ? {
                backgroundImage: `url(${video.blurDataURL})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
              }
            : undefined
        }
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          srcSet={srcSet}
          sizes={SIZES}
          alt={video.description}
          loading="lazy"
          decoding="async"
          className="h-full w-full object-cover transition-transform duration-300 ease-out group-hover:scale-[1.03]"
        />
        {video.sceneId ? (
          <span className="absolute bottom-2 right-2 rounded-md bg-black/75 px-1.5 py-0.5 text-[11px] font-medium text-white backdrop-blur-sm">
            3D scene
          </span>
        ) : null}
      </a>

      <div className="flex min-w-0 flex-col gap-0.5">
        <h3 className="line-clamp-2 text-[15px] font-medium capitalize leading-snug text-foreground">
          {video.description || video.videoFile}
        </h3>
        <p className="truncate text-[13px] text-muted-foreground">
          {video.date}
          {video.town ? ` · ${video.town}` : ""}
        </p>
      </div>

      <div className="mt-0.5 flex gap-2">
        <Button asChild size="sm" variant="secondary" className="flex-1">
          <a href={annotateHref}>Annotate</a>
        </Button>
        {viewerHref ? (
          <Button asChild size="sm" className="flex-1">
            <a href={viewerHref}>Scene viewer</a>
          </Button>
        ) : (
          <Button
            size="sm"
            variant="outline"
            className="flex-1"
            disabled
            title="No reconstructed scene yet"
          >
            No scene
          </Button>
        )}
      </div>
    </div>
  );
}
