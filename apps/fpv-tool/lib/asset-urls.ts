const DEFAULT_CLOUDFRONT_ROOT = "https://d2fioemadmrru3.cloudfront.net";

export { DEFAULT_CLOUDFRONT_ROOT };

export const assetRoot = (process.env.FPV_ASSET_ROOT ?? "").replace(/\/$/, "");

export function joinPublicUrl(root: string, path: string): string {
  const clean = path.replace(/^\//, "");
  if (!root) return `/${clean}`;
  return `${root.replace(/\/$/, "")}/${clean}`;
}

export function withTrailingSlash(url: string): string {
  return url.endsWith("/") ? url : `${url}/`;
}

export function sceneDataBase(relPath: string, appBase = ""): string {
  if (assetRoot) {
    return withTrailingSlash(joinPublicUrl(assetRoot, `scenes/${relPath}/viewer`));
  }
  const local = joinPublicUrl(appBase, `scenes/${relPath}/viewer`);
  return withTrailingSlash(local);
}
