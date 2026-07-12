# VGGT FPV Scene Reconstruction Pipeline

Date: 2026-07-02

This document summarizes the current repo workflow for turning an annotated FPV
video into an inspectable 3D scene with camera path, speed, height above ground,
top-view plots, and frame/reprojection comparisons.

## Short Summary

The high-level flow is:

1. Start from an annotated edited FPV video.
2. Convert annotation markers into flight intervals.
3. Select one or more flight intervals in the annotator.
4. Extract sampled frames from those intervals, using a central clean crop by
   default.
5. Run VGGT on the selected frame sequence.
6. Parse the VGGT output into a point cloud and relative camera path.
7. Render camera-view diagnostics from each recovered pose.
8. Fit a ground plane and attach a grid.
9. Apply a scale factor in meters per VGGT unit.
10. Visualize the scene, camera path, frame/render/overlay images, speed, AGL
    height, and top-view path in the Three.js viewer.

Your summary is correct with two caveats:

- VGGT produces relative geometry. Metric speed and height require a global
  scale factor. The current default is `117.6 m/unit`, but it is a calibration
  value, not a property VGGT recovers by itself.
- The ground grid is fitted automatically to likely ground points. It is useful
  for AGL-style measurements, but it is still an estimated plane, not surveyed
  terrain.

## Repo Storage Structure

The local tool server now persists user-facing artifacts in the repo:

```text
annotations/
  <video_stem>_annotations.json

scenes/
  <video_stem>/
    <scene_id>/
      scene_manifest.json
      annotation_snapshot.json
      metadata.json
      frames.csv
      frames/
        f_000001.jpg
        ...
      vggt_scene.glb
      point_cloud.npz
      relative_path.csv
      camera_views/
        f_000001_vggt_render.jpg
        f_000001_overlay.jpg
        ...
      scene_state.json
      viewer/
        index.html
        scene_meta.json
        points_positions.bin
        points_colors.bin
        camera_view_assets/
        ...
```

`scene_manifest.json` records the source video, selected segments, default
scale, creation time, and `model_config`: the exact preprocessing settings,
VGGT backend/model settings, upload mode, frame cap, point budget, sky-mask
setting, focal length, and camera-view render settings. `scene_state.json`
stores interactive corrections such as a manually measured scale. The viewer
directory is self-contained enough to serve the point cloud and frame
comparison assets.

The default server output root is the repo root, so generated 3D scenes land in
`scenes/`. Heavy video downloads remain cached outside the repo under
`/tmp/fpv-model-benchmark/videos`.

## Scene Browser

The local server now exposes a scene dropdown:

- `GET /api/scenes` scans the repo `scenes/` folder and returns scene
  summaries.
- `GET /api/video-status` returns annotation/scene status for the video
  dropdown.
- `GET /api/jobs` and `GET /api/jobs/<job_id>` expose reconstruction jobs.
  Job records are persisted under `scenes/.jobs/`, so reopening the UI can
  reattach to a still-running backend job and show its logs.
- `POST /api/annotations` saves annotations into the repo `annotations/`
  folder.
- `/scenes/` serves a lightweight scene picker; selecting a scene navigates into
  the generated viewer rather than embedding it in an iframe.
- `Add Folder` on `/scenes/` imports an existing generated scene folder into the
  repo `scenes/<video_stem>/<scene_id>/` structure.
- The annotator has a `Create 3D Scene` button that opens a dialog for selecting
  flight segments, previewing them, choosing crop/FPS/scale parameters, and
  starting reconstruction.
- The annotator, scene browser, and generated scene viewer are all served by the
  same local backend and expose top navigation between `Annotate` and `Scenes`.
- Generated scene viewers also include a scene dropdown in their header, so a
  saved scene can be loaded without returning to the standalone browser page.

This avoids hand-maintaining a scene list. Any reconstructed scene with a
`viewer/index.html` appears in the dropdown.

The browser UI is not the job runner. Once a reconstruction starts, the backend
thread keeps running if the browser tab is closed. If the backend server itself
is restarted before a job finishes, the persisted job is marked `stale` on the
next startup; it is visible in the job manager, but it is not resumed
automatically.

## Step-by-Step Flow

### 1. Annotation to Flight Intervals

The annotator reads JSON files under `annotations/`. A flight interval starts at
`flight_start` or `new_flight_start` and ends at the next non-flight marker, for
example `pause_start`, `replay_start`, `other`, or the end of the video.

The last flight interval for a video is treated as the attack segment and is
marked differently in the scene metadata and viewer.

Tool:

- `tools/apps/annotator/index.html`

### 2. Segment Selection and Frame Extraction

In the annotator, `Create 3D Scene` opens a dialog where you select one or more
detected flight intervals. If multiple intervals are selected, the system
concatenates their extracted frame sequence for VGGT. This should be used only
when the segments likely share scene overlap; otherwise they should be
reconstructed as separate scenes. The dialog can preview the selected intervals
in the source video so transitions can be checked before running VGGT.

Frame extraction uses FFmpeg. The default crop preset is `central_clean`,
restored from the manually selected clean crop used in the first high-quality
VGGT overlay:

```text
crop=x=iw*(120/848), y=ih*(190/478), w=iw*(660/848), h=ih*(280/478)
```

On the common `848x478` source videos, that is exactly `x=120, y=190, w=660,
h=280`. It removes the side blur/rotor regions and the top HUD/propeller band
while preserving the flight-relevant lower central view. Extracted frames are
then scaled to the requested width. When an FPS is explicitly selected, all
frames emitted by FFmpeg's `fps` filter are sent forward; there is no additional
target-frame cap.

The backend also encodes the extracted frame sequence as `vggt_input.mp4`.
For public VGGT Space uploads this MP4 is a transport wrapper, not the timing
source: high-FPS extracted frames are packed into a 2 fps MP4, then VGGT samples
that upload video at 2 fps so every extracted frame is still used. Real timing
for speed/height remains in `frames.csv` via `video_time_s`, `segment_time_s`,
and `sequence_time_s`.

Default UI parameters:

- sample FPS: `3`
- estimated image count: selected flight duration multiplied by sample FPS; this
  is informational when FPS is explicitly selected, not a cap
- crop: `central_clean`
- width: `660`
- focal length for reprojection diagnostics: `812 px`
- default scale: `117.6 m/unit`

When `central_clean` is selected, the annotator draws the crop box over the
displayed video so the user can see exactly which central region will be sent to
VGGT.

Tool:

- `tools/server/fpv_tool_server.py`

### 3. VGGT Reconstruction

The backend calls the existing pipeline:

```bash
python tools/pipeline/flight_path_pipeline.py \
  --out-dir <repo-root> \
  run-vggt \
  --recon-subdir scenes \
  --video-id <scene_id>
```

The VGGT run uses the selected extracted frames. The output `.glb` and related
assets are stored in the scene directory.

Reconstruction jobs refresh the VGGT output by default. This avoids pairing a
newly extracted crop/FPS frame set with a stale `.glb` from an older run of the
same scene id.

Tool:

- `tools/pipeline/flight_path_pipeline.py`

### 4. Extract Point Cloud and Relative Camera Path

After VGGT finishes, the backend runs:

```bash
python tools/pipeline/flight_path_pipeline.py \
  --out-dir <repo-root> \
  extract-vggt \
  --recon-subdir scenes \
  --video-id <scene_id> \
  --refresh
```

This produces:

- `point_cloud.npz`
- `relative_path.csv`

The camera path is relative and unitless. For this VGGT export, camera
orientation is recovered from the visual camera frustum geometry in the GLB,
not from a unique per-camera node transform.

Tool:

- `tools/pipeline/flight_path_pipeline.py`

### 5. Camera-View Reprojection Diagnostics

The backend can render the VGGT point cloud from recovered camera poses:

```bash
python tools/pipeline/render_vggt_reprojection_samples.py \
  <scene_dir> \
  --out-dir <scene_dir>/camera_views \
  --samples 1,2,3,... \
  --flip-y \
  --focal-px 812 \
  --view full \
  --splat 1
```

For the Adaissah Sholef sample, the visually preferred diagnostic used full
frame rendering with `f=812`, `--flip-y`, `--view full`, and `--splat 1`.
Those settings are now the default for the server-generated camera-view assets.

Tool:

- `tools/pipeline/render_vggt_reprojection_samples.py`

### 6. Ground Grid and AGL Height

The viewer generator estimates a ground plane from likely ground-colored point
cloud samples, refines it with a plane fit, and builds a grid on that plane.

The grid is fixed in VGGT units. When the scale changes, the visual scene does
not resize; instead the displayed grid-square size in meters changes. This is
the desired behavior for inspecting whether the chosen scale is plausible.

Height above ground is computed as the signed distance from each camera center
to the fitted ground plane:

```text
height_m = distance_to_ground_plane_vggt_units * scale_m_per_unit
```

The viewer clamps negative heights to zero for display.

Tool:

- `tools/pipeline/create_vggt_threejs_viewer.py`

### 7. Metric Scale

The viewer starts from a default scale, currently `117.6 m/unit`, and lets the
user correct it:

1. Type a scale directly in the scale input.
2. Click `Measure`.
3. Pick two points in the point cloud.
4. Enter the real-world length in meters.
5. Apply the scale.

The corrected scale is saved to `scene_state.json` through:

```text
POST /api/scenes/<scene_id>/state
```

The generated viewer has an explicit `Save` button for this state. Scale edits
are also stored locally while interacting, but the save button is the deliberate
backend write for the scene's current scale.

Important: metric speed, distance, and height are only as good as this scale and
the fitted ground plane. VGGT alone does not determine meters.

Related scale research/tools:

- `docs/automatic_scale_pipeline.md`
- `tools/pipeline/auto_scale_hypotheses.py`
- `tools/pipeline/vggt_metric_depth_scale_votes.py`
- `tools/pipeline/fit_vggt_object_box.py`
- `tools/pipeline/segment_object_with_sam.py`

### 8. Speed and Distance

The viewer computes per-step speed from adjacent recovered camera centers:

```text
speed_m_s = distance(camera_i, camera_i-1)_vggt_units * scale_m_per_unit / dt_s
```

For multi-segment reconstructions, speed is broken at segment boundaries so a
cut between two selected intervals does not create a fake speed spike. The
viewer displays raw speed and a smoothed speed curve.

Distance to target is currently distance to the last camera pose in the selected
sequence:

```text
distance_to_target_m = distance(camera_i, final_camera)_vggt_units * scale_m_per_unit
```

### 9. Viewer Outputs

The generated Three.js viewer shows:

- VGGT point cloud
- camera path
- current camera marker and frustum
- fitted ground grid
- actual / VGGT render / overlay frame panel
- speed vs. time
- height above ground vs. distance to target
- top-view path
- scale correction controls

Tool:

- `tools/pipeline/create_vggt_threejs_viewer.py`

## Local Server Commands

Start the local tool server:

```bash
/tmp/fpv-depth-venv/bin/python tools/server/fpv_tool_server.py \
  --port 8766 \
  --python /tmp/fpv-depth-venv/bin/python \
  --vggt-python /tmp/fpv-depth-venv/bin/python
```

Open:

- annotator and reconstruction launcher: `http://127.0.0.1:8766/`
- scene dropdown browser: `http://127.0.0.1:8766/scenes/`

## Current Limitations

- The flight intervals still come from manual annotations unless the transition
  detector pipeline is run separately.
- Multi-segment selection should be used only when the selected intervals share
  scene overlap.
- The metric scale is not automatic yet. The viewer supports correction, and
  the automatic scale-voting work is documented separately.
- Ground is a fitted plane, not a DEM/DSM-aligned terrain surface.
- Reprojection quality depends on the assumed focal length and the VGGT export's
  recovered camera basis.
