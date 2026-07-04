import { SiteHeader } from "@/components/site-header";
import { VideoBrowser } from "@/components/video-browser";
import { loadVideos } from "@/lib/videos";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const videos = loadVideos();
  const withScene = videos.filter((v) => v.sceneId).length;

  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6">
        <div className="mb-5">
          <h1 className="text-xl font-semibold tracking-tight">FPV strike videos</h1>
          <p className="mt-1 text-[13px] text-muted-foreground">
            {videos.length} videos · {withScene} with a reconstructed 3D scene
          </p>
        </div>
        <VideoBrowser videos={videos} />
      </main>
    </div>
  );
}
