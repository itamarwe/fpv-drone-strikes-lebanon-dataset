(() => {
  const CATALOG_URL = "https://d2fioemadmrru3.cloudfront.net/data/videos.json";
  let scenesPromise;

  function viewerUrl(scenePath) {
    const encodedPath = scenePath.split("/").map(encodeURIComponent).join("/");
    return `/scenes/${encodedPath}/viewer/index.html`;
  }

  function sceneFromVideo(video) {
    const scenePath = typeof video.scenePath === "string" ? video.scenePath.trim().replace(/^\/+|\/+$/g, "") : "";
    if (!scenePath || scenePath.split("/").some((part) => !part || part === "." || part === "..")) return null;
    const parts = scenePath.split("/");
    return {
      scene_id: parts.at(-1),
      path: scenePath,
      viewer_url: viewerUrl(scenePath),
      exists: true,
      video_file: video.videoFile || "",
      description: video.description || "",
      date: video.date || "",
      town: video.town || "",
      segment_ids: parts.at(-1).split("_").filter((part) => /^seg\d+$/i.test(part)),
    };
  }

  window.loadPublishedScenes = function loadPublishedScenes() {
    if (!scenesPromise) {
      scenesPromise = fetch(`${CATALOG_URL}?cache=${Date.now()}`, { cache: "no-store" })
        .then((response) => {
          if (!response.ok) throw new Error(`Published scene catalog HTTP ${response.status}`);
          return response.json();
        })
        .then((payload) => {
          const byPath = new Map();
          for (const video of payload.videos || []) {
            const scene = sceneFromVideo(video);
            if (scene) byPath.set(scene.path, scene);
          }
          return [...byPath.values()];
        });
    }
    return scenesPromise;
  };
})();
