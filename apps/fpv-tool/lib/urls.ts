// Browser-safe URL helpers (no Node-only imports) so client components can use
// them without pulling server modules into the bundle.

export const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? process.env.FPV_APP_BASE ?? "").replace(
  /\/$/,
  "",
);

export function withBasePath(urlPath: string): string {
  if (!basePath) return urlPath;
  return `${basePath}${urlPath.startsWith("/") ? urlPath : `/${urlPath}`}`;
}

// Optional CDN base for the generated gallery thumbnails. When set (e.g.
// https://d2fioemadmrru3.cloudfront.net/thumbnails), the gallery loads
// <slug>/<width>.webp from there instead of the local /thumbnails folder.
export const thumbBase = (process.env.NEXT_PUBLIC_THUMB_BASE ?? "").replace(/\/$/, "");
