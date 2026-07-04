import { SiteHeader } from "@/components/site-header";
import { VideoCard } from "@/components/video-card";
import { loadVideos } from "@/lib/videos";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const videos = loadVideos();
  const withScene = videos.filter((v) => v.sceneId).length;

  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6">
        <div className="mb-6">
          <h1 className="text-xl font-semibold tracking-tight">FPV strike videos</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {videos.length} videos · {withScene} with a reconstructed 3D scene
          </p>
        </div>

        <div className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {videos.map((video) => (
            <VideoCard key={video.videoFile} video={video} />
          ))}
        </div>
      </main>
    </div>
  );
}
