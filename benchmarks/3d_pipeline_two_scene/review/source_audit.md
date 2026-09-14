# Pinned reconstruction output and renderer audit

Audit date: 2026-09-08. The repositories were checked out at the exact commits
in `benchmark.json`; findings below describe those pins, not a guessed future
interface.

## Surflo

- Repository: `https://github.com/Anttwo/Surflo.git`
- Commit: `bf14c6375a92911c45795710cd00bf2af17e9a13`
- Inspected: `README.md`, `scripts/infer.py`, `configs/infer.yaml`,
  `configs/plain/default.yaml`, `configs/guided/default.yaml`,
  `configs/mesh/default.yaml`, `configs/texture/default.yaml`,
  `surflo/data/image_folder.py`, `surflo/utils/io.py`.
- Input preprocessing uses Surflo/VGGT `no_stretch` at target size 518 and can
  rotate portrait inputs. The selected input paths are written to
  `_infer_summary.json`.
- Plain mode writes oriented `initial.ply` and `final.ply` point clouds. These
  are geometry diagnostics; normal-derived vertex colours are not RGB texture.
- Guided mode writes `point_cloud_normals.ply`, attempts
  `point_cloud_rgb.ply`, extracts `mesh.ply`, and can write either
  vertex-coloured `mesh_textured.ply` or UV-textured `mesh_textured.glb`.
- Guided meshing constructs refined cameras internally. The inspected inference
  driver does not write those cameras or a set of matched RGB/depth/normal
  renders. A generic mesh turntable is therefore valid only as an exploratory
  structure/appearance review, not as a source-frame photometric test.

## QuerySplat

- Repository: `https://github.com/inspatio/querysplat.git`
- Commit: `3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae`
- Inspected: `README.md`, `scripts/infer.py`, `scripts/options.py`,
  `scripts/output_utils.py`, `scripts/models/querysplat.py`,
  `scripts/rendering/gs.py`, `scripts/utils/data.py`.
- Custom images are centre-cropped and resized to 512 × 512. The input count
  overrides the config's four-view default.
- The script predicts input cameras and depth with VGGT-Omega, reconstructs
  Gaussians, optionally fits them to the input RGBs with TTO, then calls
  `render_gaussians` on the same predicted input cameras.
- Native outputs include preprocessed input frames, input-camera renders,
  Gaussian PLY, coloured Gaussian-centre PLY, timing, optional camera JSON/NPZ,
  depth/confidence files and a VGGT-Omega depth point cloud. The benchmark
  command requests the optional camera/depth/point-cloud products.
- Gaussian PLY export applies the requested opacity threshold; its default is
  0.05, while the native renderer uses the full in-memory tensor. Offline
  render comparisons need an additional opacity-0 PLY and must retain the
  threshold list in `inference_timing.json`.
- The renderer accepts in-memory Gaussian tensors and explicit `cam_view` and
  `[fx, fy, cx, cy]` intrinsics. The inspected pin does not contain a loader plus
  standalone command for rendering its saved Gaussian PLY at new cameras.
- Native input renders are useful in-sample diagnostics. TTO input renders are
  explicitly photometrically fitted. Neither is held-out evidence.

## Review implication

The first honest equal-input comparison can include native geometry, Surflo
mesh/texture turntables, and QuerySplat input-camera fidelity. A camera-matched
evaluation proxy can use a separate VGGT-Omega camera-head pass, align its
Train16 cameras to the frozen QuerySplat camera frame, and render the frozen
Gaussian PLY. That route is bounded and auditable, but weaker than independent
feature localization: evaluation RGB participates in the pose network and
joint attention also changes the Train16 estimates used for alignment. Report
the alignment residuals and label pose uncertainty. Until the adapter passes
those checks, held-out status is `unavailable`, not zero and not inferred from
similarly numbered files.

## Scal3R and prepared baselines

- Scal3R repository: `https://github.com/zju3dv/Scal3R.git`
- Commit: `dc557e7be5ad821ed44b8ad37700311136061406`
- Inspected: `README.md`, `scal3r/utils/result_utils.py`, and
  `scripts/visualize/viser_viewer.py` plus its utilities.
- Inference writes `mat.txt` c2w poses, EasyVolCap-style intrinsics/extrinsics,
  optional depth EXRs, masks, block point clouds and a downsampled RGB
  `points/whole.ply`. It does not export a textured surface or native held-out
  appearance renders.
- The pinned tree includes a Viser point-cloud/camera-path viewer even though an
  older README TODO still says a simple viewer remains to be released. The code
  at the pinned revision is the authority for this audit.
- The local published VGGT bundle exposes raw position and colour buffers. It
  has no camera-linked RGB render set in the prepared package, so it is a
  current-product geometry baseline until exact camera/render metadata is
  exported.
- GLOMAP is a sparse camera/SfM scaffold. COLMAP model conversion can produce a
  PLY for structural review; it does not make a dense appearance reconstruction.
