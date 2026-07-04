import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import type { Dataset, VideoRecord } from "./types";
import { Header } from "./components/Header";
import { Gallery } from "./views/Gallery";
import { VideoView } from "./views/VideoView";

// three.js only loads when someone opens a 3D scene.
const SceneView = lazy(() =>
  import("./views/SceneView").then((m) => ({ default: m.SceneView })),
);

// Tiny hash router: "#/", "#/video/<videoFile>", "#/scene/<videoFile>".
// Hash routing keeps deep links working from any static mount point
// (e.g. itamarweiss.com/fpv/#/video/...), with no server rewrites.
type Route =
  | { view: "gallery" }
  | { view: "video"; videoFile: string }
  | { view: "scene"; videoFile: string };

function parseHash(): Route {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const [view, ...rest] = hash.split("/");
  const arg = decodeURIComponent(rest.join("/"));
  if (view === "video" && arg) return { view: "video", videoFile: arg };
  if (view === "scene" && arg) return { view: "scene", videoFile: arg };
  return { view: "gallery" };
}

export function videoHref(videoFile: string): string {
  return `#/video/${encodeURIComponent(videoFile)}`;
}

export function sceneHref(videoFile: string): string {
  return `#/scene/${encodeURIComponent(videoFile)}`;
}

export function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/videos.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setDataset)
      .catch((e) => setError(String(e)));
  }, []);

  const byFile = useMemo(() => {
    const map = new Map<string, VideoRecord>();
    for (const v of dataset?.videos ?? []) map.set(v.videoFile, v);
    return map;
  }, [dataset]);

  let body: JSX.Element;
  if (error) {
    body = <p className="not-found">Failed to load the dataset ({error}).</p>;
  } else if (!dataset) {
    body = <p className="not-found">Loading…</p>;
  } else if (route.view === "video" || route.view === "scene") {
    const video = byFile.get(route.videoFile);
    if (!video) {
      body = <p className="not-found">Video not found.</p>;
    } else if (route.view === "video") {
      body = <VideoView video={video} />;
    } else {
      body = (
        <Suspense fallback={<p className="not-found">Loading 3D viewer…</p>}>
          <SceneView video={video} />
        </Suspense>
      );
    }
  } else {
    body = <Gallery videos={dataset.videos} />;
  }

  return (
    <>
      <Header />
      <div className="page-content">
        <div className="wrapper">{body}</div>
      </div>
    </>
  );
}
