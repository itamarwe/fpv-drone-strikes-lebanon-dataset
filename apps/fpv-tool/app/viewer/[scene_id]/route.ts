import { serveViewerBySceneId } from "@/lib/scene-files";

type RouteContext = { params: Promise<{ scene_id: string }> };

export async function GET(_request: Request, context: RouteContext) {
  const { scene_id } = await context.params;
  const response = await serveViewerBySceneId(decodeURIComponent(scene_id));
  return response ?? new Response("Scene not found", { status: 404 });
}
