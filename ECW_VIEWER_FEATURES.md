# ECW Viewer — Feature Implementation Brief

Context: there's a local ECW tile viewer served by `tools/ecw_tile_server.py`,
run as:

```bash
python3 tools/ecw_tile_server.py --ecw ~/Downloads/tzafon_3_26.ecw --port 8787
```

A heights raster is available at `~/Downloads/height_evoPilot.tif` (GeoTIFF DEM).

Goal: add the features below to the tile server + its HTML/JS viewer. Start by
reading `tools/ecw_tile_server.py` and the HTML it serves to learn the existing
tile scheme (endpoint shape, z/x/y or pixel-window params, tile size, CRS,
image bounds) before writing anything. Match the existing structure; don't
rewrite unless necessary.

---

## 1. Stable viewport (do this first — everything else depends on it)

The page itself must never scroll or zoom; only the map does.

- On the map container: `touch-action: none;` and `overscroll-behavior: none;`.
- `<meta name="viewport" content="width=device-width, initial-scale=1,
  maximum-scale=1, user-scalable=no">`.
- `preventDefault()` on `wheel`, `touchmove`, and `gesturestart`/`gesturechange`
  (Safari) handlers, attached with `{ passive: false }`.
- Body: `position: fixed; inset: 0; overflow: hidden;` so iOS rubber-banding
  can't move the app.
- Keep all transforms on a single inner "world" element so pan/zoom compose
  cleanly (`transform: translate(x,y) scale(s)`), and set
  `will-change: transform`.

## 2. Zoom — mouse wheel + mobile pinch

- **Wheel:** zoom centered on the cursor. Convert cursor screen point → world
  coords, apply scale delta, re-solve translation so that world point stays
  under the cursor. Clamp scale to `[minScale, maxScale]`. Use a smooth factor
  like `scale *= Math.exp(-e.deltaY * 0.0015)`.
- **Pinch:** track two pointers via Pointer Events (`pointerdown/move/up` +
  `setPointerCapture`). Zoom around the midpoint of the two touches using the
  ratio of current/initial finger distance. Same "keep the focal point fixed"
  math as wheel.
- Prefer Pointer Events over Touch Events so one code path covers mouse + touch.

## 3. Keyboard panning

- Arrow keys (and WASD) pan the map by a fixed step; hold = continuous motion
  via a `requestAnimationFrame` loop keyed off a held-keys `Set`.
- `+` / `-` (and `=`) zoom in/out around viewport center.
- Only when the map/canvas is focused; give the container `tabindex="0"` and a
  visible focus style. Don't hijack keys when a text input is focused.

## 4. Momentum / inertial scrolling

- Track pointer velocity over the last few move events (px/ms, exponentially
  smoothed).
- On release, if speed > threshold, start an rAF loop applying velocity to the
  translation and decaying it each frame (`v *= 0.92` ≈ nice glide; tune).
- Stop when speed < ~0.05 px/ms or on any new pointer/keydown.
- Clamp panning to image bounds with a soft rubber-band, or hard clamp — pick
  hard clamp if simpler.

## 5. Anti-aliasing / image quality

- Cause is usually nearest-neighbor upscaling + ignoring devicePixelRatio.
- Render tiles into a `<canvas>` sized to `cssPx * devicePixelRatio`, with the
  CSS size set separately, and `ctx.scale(dpr, dpr)`.
- `ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';`
- For `<img>`-based tiles use `image-rendering: auto` (not `pixelated`).
- Server side: when the ECW reader resamples, request **bilinear/cubic**, not
  nearest. Serve tiles at native resolution for the current zoom level so the
  browser isn't stretching a low-res tile. Confirm tile pixel size matches DPR
  (serve @2x tiles on retina if feasible).

## 6. Tile caching / prefetch

- **Client:** LRU cache of decoded tiles (Map keyed by `z/x/y`, cap ~200,
  evict oldest). After rendering the visible set, prefetch the surrounding ring
  (one tile beyond each edge) plus the next zoom level's covering tiles, at low
  priority (`fetchpriority="low"` or `requestIdleCallback`). Cancel in-flight
  requests for tiles that scroll out of view (`AbortController`).
- **Server:** add HTTP caching headers (`Cache-Control: public, max-age=…`,
  `ETag`) so the browser HTTP cache helps too. Optionally an in-process
  memoized tile cache (functools.lru_cache / dict) keyed by tile params so
  repeat reads skip ECW decode.

## 7. Heights + 3D tilt (from `height_evoPilot.tif`)

Recommended approach: **drape + tilt (2.5D)**, WebGL via three.js.

- Server: add a DEM endpoint. Read the GeoTIFF with `rasterio` (or GDAL),
  reproject/align it to the ECW's CRS + bounds, and serve either:
  - small normalized 16-bit/float heightmap PNG/tiles for the current view, or
  - a downsampled elevation grid as JSON/binary for the whole extent.
  Include min/max elevation + the geo-transform so the client can scale Z.
- Client: three.js `PlaneGeometry` subdivided into a grid; displace vertices by
  sampling the heightmap (vertex shader or CPU). Texture = the ECW tiles draped
  on top. `OrbitControls` (or custom) for pitch/tilt + rotate; clamp pitch so
  you can't flip under the terrain.
- Add a **2D ⇄ 3D toggle** and a vertical-exaggeration slider (terrain relief is
  subtle at true scale). Keep the existing 2D pan/zoom working in flat mode.
- Ship tilt-only first if the DEM plumbing is slow; wire real elevation after.

## Suggested order

1. Stable viewport (#1) — foundation.
2. Zoom (#2) + keyboard (#3) on one transform model.
3. Momentum (#4).
4. Anti-aliasing (#5) + tile caching/prefetch (#6) — perf polish.
5. Heights + 3D (#7) — biggest, do last.

## Acceptance checks

- Page never scrolls/zooms; only the map responds to gestures (desktop + iOS).
- Pinch and wheel zoom stay centered on cursor/fingers.
- Arrows/WASD pan; +/- zoom; keys ignored while typing in inputs.
- Flick-pan glides and settles smoothly.
- Imagery is crisp on retina (no jaggies/shimmer on pan).
- Panning reuses cached tiles; surrounding tiles are already loaded.
- 3D toggle drapes imagery over terrain; tilt/rotate works; exaggeration slider
  visibly changes relief.
