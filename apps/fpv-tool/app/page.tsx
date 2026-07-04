import { SiteHeader } from "@/components/site-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardTitle } from "@/components/ui/card";
import { withBasePath } from "@/lib/config";
import { loadVideos } from "@/lib/videos";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const videos = loadVideos();
  const withScene = videos.filter((v) => v.sceneId).length;

  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-7xl px-4 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">FPV strike videos</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {videos.length} videos · {withScene} with a reconstructed 3D scene. Annotate
            a clip or open its scene viewer.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {videos.map((video) => {
            const annotateHref = withBasePath(
              `/annotate/?v=${encodeURIComponent(video.videoFile)}`,
            );
            const viewerHref = video.sceneId
              ? withBasePath(`/viewer/${encodeURIComponent(video.sceneId)}/`)
              : null;
            return (
              <Card key={video.videoFile} className="group">
                <a
                  href={viewerHref ?? annotateHref}
                  className="relative block aspect-video overflow-hidden bg-secondary"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={video.thumbnailUrl}
                    alt={video.description}
                    loading="lazy"
                    className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
                  />
                  {video.sceneId ? (
                    <span className="absolute right-2 top-2 rounded-full bg-primary/90 px-2 py-0.5 text-[11px] font-medium text-primary-foreground">
                      3D scene
                    </span>
                  ) : null}
                </a>
                <CardContent className="flex flex-1 flex-col gap-1.5 pt-3">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <time>{video.date}</time>
                    {video.town ? (
                      <>
                        <span aria-hidden>·</span>
                        <span className="truncate">{video.town}</span>
                      </>
                    ) : null}
                  </div>
                  <CardTitle className="line-clamp-2 text-sm capitalize">
                    {video.description || video.videoFile}
                  </CardTitle>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="secondary" size="sm" className="flex-1">
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
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </main>
    </div>
  );
}
