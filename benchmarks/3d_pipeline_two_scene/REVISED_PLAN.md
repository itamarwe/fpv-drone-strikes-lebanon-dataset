# Revised benchmark and presentation plan

Revision: 2026-09-08, updated with confirmed scene measurements and reconstruction-quality goals.
Status: execution halted on 2026-09-08; the paid pod has been deleted. See [EXECUTION_STATUS.md](EXECUTION_STATUS.md). The protocol below is historical.

This plan supersedes the experimental design in README.md and benchmark.json.
The prepared archive remains a useful input and script package, but its runners
and evaluators need the corrections below before they implement this protocol.
No new model execution or paid resources are required to review this plan.

## Decisions we want to make

1. Can we produce correctly scaled reconstructions using the existing measured
   elements, and can automatic scale recover those known scales?
2. Can a different reconstruction reduce local shape distortion after a single
   global scale is applied?
3. Can Surflo, QuerySplat and other viable methods produce better, more visually
   appealing 3D reconstructions: coherent surfaces, preserved detail, fewer
   artifacts, believable appearance and stable structure across viewpoints?

Browser delivery is a secondary engineering consideration. Reconstruction
quality is evaluated on the full-quality output before browser conversion,
compression, simplification or rendering performance can affect the assessment.

Choose a winner for each decision. A combined production pipeline is a subsequent
integration experiment. Two scenes support a case study and a pilot adoption
decision, not a broad claim of superiority on FPV footage.

## What the review changed

- A GLOMAP metre-per-unit coefficient cannot be compared directly with a VGGT
  metre-per-unit coefficient. Each reconstruction has its own coordinate scale.
  The current fitter and wrapper perform this invalid comparison; retire that
  error field before scoring. Compare measured physical distances instead.
- Multiplying all coordinates by a scalar changes absolute scale but leaves
  shape distortion intact. MoGe-3 scale fitting and geometric improvement must
  therefore have separate tests.
- A Sim(3) fit removes global scale error. Use it only for shape/trajectory
  agreement. For metric geometry, apply the predicted scale and allow rigid
  alignment only. Published VGGT is a reference, not pose ground truth.
- The previous metric-depth report recorded order-of-magnitude scale misses.
  Check coordinate conventions, focal/crop handling, depth definition, and
  calibration provenance before interpreting a new miss as model failure.
- The staged images include duplicates: Scene A has 124 unique images among
  125 frames; Scene B has 139 among 149. The current common16/heldout split has
  no exact-hash overlap, but near duplicates and temporal leakage need checking.
- Frames held out from common16 are included in full. A full-sequence model
  cannot be scored as unseen-view reconstruction on those same frames.
- Surflo and QuerySplat have their own camera/depth estimation paths. First
  compare their native complete systems. A fixed-camera representation-only
  comparison needs explicit adapters and is a later controlled experiment.
- QuerySplat's inference loader accepts arbitrary input counts and overrides
  the four-view config default. Sixteen views are a valid planned test, with
  memory and image preprocessing still to be verified on the GPU.
- The current image evaluator pairs files by sorted order and resizes rendered
  images. This is insufficient for trustworthy scoring across different crops
  and fields of view. Require explicit frame IDs and calibrated pixel mappings.

## Phase 0: establish trustworthy inputs and evaluation

Keep the two user-selected scenes and the published image preprocessing as the
initial source. Retain original timestamps, segment boundaries, source hashes,
and every method's resize/crop/intrinsics transformation.

Freeze these inputs before viewing method results:

- **Train16:** 16 well-spaced views covering both segments, selected without
  inspecting method outputs. The existing common16 is a starting candidate.
- **Evaluation views:** approximately 8-12 unique views per scene, including
  one short contiguous held-out clip where coverage permits. Exclude exact and
  near duplicates of training views and use temporal buffer frames for that clip.
- **Dense training:** all eligible frames except evaluation views and their
  temporal buffers. This gives dense methods a valid unseen-view test.
- **Full-input showcase:** all frames; use only for the eventual best display
  asset and in-sample QA, with its input count explicitly recorded.

Preserve a map back to every original frame when deduplicating. Score continuity
within segments using elapsed time; report edit/segment transitions separately.
An edited video gap is not automatically a tracking failure.

Use the measurements already saved for these exact scenes. They were found in
the prepared copies of published scene_meta.json, under `calibration`, and are
recorded in [calibration_references.json](calibration_references.json).

| Scene | Measured element length | Length in original VGGT units | Saved m/VGGT unit | Measurement timestamp |
|---|---:|---:|---:|---|
| Scene A | 3 m | 0.019527748975898915 | 153.62753811013192 | 2026-07-02 16:35:25 UTC |
| Scene B | 7 m | 0.06831557840906922 | 102.4656478509844 | 2026-07-04 11:04:33 UTC |

Both records have `source: measured`; the user has confirmed they are measured
elements in the selected scenes. Additional measurements are optional supporting
evidence, not a prerequisite to this benchmark.

Use each selected scene's published calibration record as the authority. The
saved metadata contains the scalar lengths and timestamps but no element labels
or endpoint coordinates. The inspected viewer's save payload stores lengths and
a frame index, but omits the picked endpoints. Recover the measurement location
from an available original state/annotation if possible; otherwise annotate the
same measured element in source frames before transferring calibration to new
geometry. Do not infer the element's identity from length alone.

Use each measurement in two separate experimental modes:

- **Anchor-calibrated reconstruction:** apply the known length to every new
  reconstruction independently. This is the main mode for comparing overall
  metric geometry and reconstruction quality in consistent physical units.
- **Automatic scale test:** withhold the length from MoGe-3 scale fitting, then
  evaluate the predicted physical length against the saved 3 m or 7 m reference.

The same anchor is allowed in both modes, but fitting and evaluation must be
clearly labelled per run. Its exact agreement after anchor calibration is not
independent proof that the rest of the shape is undistorted. Use multiview
consistency, visible structure and any available additional measurements for
that question, without delaying the benchmark to request more measurements.

## Phase 1: inexpensive evidence about automatic scale

Run MoGe-3 on Train16 for both scenes, keeping the existing geometry fixed.
Estimate one robust global scale using reliable multiview observations; retain
the raw votes, rejected votes, spatial coverage, and per-frame estimates.

For the unchanged published VGGT geometry, compare the automatically predicted
scale directly with 153.62753811013192 (Scene A) or 102.4656478509844 (Scene B).
This coefficient comparison is valid because prediction and reference use the
same coordinate units. Also show the equivalent predicted measured-element
length: predicted scale multiplied by the saved measured_vggt_units, versus
3 m or 7 m. Those labels are withheld from the automatic estimator.

For a new reconstruction, measure the same element in that reconstruction's
own coordinates. Compare its predicted physical length with the known length;
never compare its raw scale coefficient directly with the VGGT coefficient.

Use the existing reconstruction's intrinsics where recoverable, and test the
model-estimated focal alternative as a small diagnostic ablation. When exact
intrinsics are unavailable, label the approximation and report its sensitivity.
Distinguish camera-z depth from radial range, normalized from pixel intrinsics,
and original-image from cropped-image coordinates.

Do not optimize scene-specific correction factors against the evaluation
distances. If scaling still fails substantially, retain manual scale and move
forward with geometry and visualization; automatic scale is independently
adoptable or rejectable.

## Phase 2: compare geometry with a useful baseline and ablations

Run the following on the same dense-training frames in both scenes:

| Experiment | Question | Priority |
|---|---|---|
| Fresh current VGGT configuration | What can the existing pipeline do on identical inputs? | Required control |
| Scal3R-ZJU | Does test-time reconstruction improve shape, coverage, and consistency? | Primary candidate |
| GLOMAP with explicit feature/matching configuration | Does classical global SfM give a better geometric scaffold? | Primary candidate |
| Each viable geometry + the same MoGe-3 scale procedure | Does scale transfer consistently across backends? | Reuse Phase 1 depth outputs |
| LightGlue with a released compatible extractor | Can matching rescue weak GLOMAP coverage? | Conditional diagnostic |
| SSMB + LightGlue | Does the requested matcher improve reconstruction? | Deferred until release/access |

Keep the already published VGGT asset as a separate “current product” comparison.
Use fresh matched-input VGGT for controlled method claims. If its exact original
configuration cannot be recovered, name and record the replacement baseline.

Initially retain the pinned standalone GLOMAP version for reproducibility;
its upstream now recommends COLMAP's global mapper. Record that limitation and
avoid changing the mapper during this comparison. Verify matching across the
edited segment transition and choose the largest valid connected reconstruction,
while reporting unregistered frames and additional components.

Run long-sequence candidates on dense input first. Train16 geometry runs are
optional input-count ablations, not the only opportunity for these methods to
succeed. Do not build a full backend × scale × representation grid immediately.

Score absolute distances in metres, and separately score local shape after
one global calibration. Useful local measures include independently known
length ratios and distances across depths. Plane/line residuals, reprojection
residuals, and cross-view consistency are supplementary evidence: flat or
self-consistent output can still be wrong. Avoid local warps in scoring.

Do not implement dense-depth-regularized bundle adjustment in the first round.
It is a separate research task justified only if MoGe depth and base SfM both
show useful evidence. Global post-hoc scaling cannot demonstrate that benefit.

## Phase 3: compare full-quality 3D reconstructions

Start this phase early, after the inputs and render protocol are frozen. This
is a primary reconstruction experiment with its own quality outcomes, and can
succeed even if automatic scale prediction fails. Use the saved element-based
calibration whenever a method needs physical scale.

On identical Train16 images, compare native Surflo plain/guided and QuerySplat
feed-forward/TTO, retaining their predicted cameras and preprocessing metadata.
QuerySplat's exported VGGT-Omega point cloud provides a particularly useful
within-system baseline for whether Gaussian rendering helps its own geometry.
Surflo plain is a surface/structure diagnostic; its normal-coloured cloud should
not compete in an RGB photometric ranking against textured outputs.

There are two comparisons:

1. **Reconstruction comparison:** published and matched-input baseline geometry
   versus candidate native systems, including surface, appearance and camera
   quality, with input counts and costs disclosed.
2. **Representation comparison:** where supported, render a point cloud and new
   representation derived from the same geometry/cameras. Otherwise describe
   the result as a system improvement, not a representation-only improvement.

Assess two complementary views of each full-quality reconstruction:

- **Geometry:** untextured/neutral-shaded meshes or depth/normal views, appropriate
  to the output. Inspect surface continuity, duplicated or floating structures,
  edge sharpness, loss of thin detail, holes, local warping, and spatial coherence.
  Do not force a Gaussian output through an unvalidated mesh extraction just
  to obtain a common format; inspect its depth and cross-view structure instead.
- **Appearance:** RGB/textured/shaded output. Inspect texture sharpness, colour
  consistency, blur, ghosting, plausibility and visual appeal during camera
  motion. Assess whether appearance improvements are supported by stable 3D
  structure. A pleasing single frame is insufficient evidence on its own.

Use native maximum-quality exports for this comparison, with consistent camera
views, resolution and lighting where applicable. Keep opaque shaded surfaces,
normal-coloured clouds and photorealistic outputs in suitable comparison modes.
Render quality should not depend on a particular browser implementation.

Begin with the same 16-view input budget, then give promising methods a documented
quality pass using more eligible views and/or refinement where supported. Repeat
the chosen settings on both scenes. Report equal-input and best-achieved results
separately so a higher-quality result can be appreciated with its compute/input cost.
Preserve the raw mesh, splats, cloud, camera, depth and texture artifacts exported
by each method. Browser assets are downstream derivatives of these originals.

For evaluation views, recover intrinsics and poses independently of test-view
RGB fitting. Prefer localization against frozen training geometry. If a fixed
published camera rig is used as an approximate reference, align it using only
training correspondences and report that source of uncertainty. No per-test-view
camera refinement to maximize the appearance score. Missing reliable poses mean
the corresponding photometric result is unavailable, not a fabricated score.

Use explicit image IDs, common visible field of view, valid-pixel masks, and
identical display resolution/background. Report PSNR/SSIM/LPIPS with image
coverage and masked fraction. Show both a full-frame result and prespecified
static-region crops so method-specific holes cannot disappear through masking.

Evaluate near observed viewpoints and during limited lateral camera movement.
Large orbits into never-observed surfaces are labelled exploratory and are not
the main reconstruction-quality test. Browser performance is a later delivery
test on a common device, viewport, network condition and loading-cache state.

## Success criteria and experiment budget

These are proposed practical decision thresholds, to freeze before execution,
not claims about results or statistical significance:

| Decision | Candidate adoption criterion | Supporting evidence |
|---|---|---|
| Anchor calibration | Use the saved 3 m/7 m length to determine physical scale independently for each reconstruction | Corresponding element, endpoint mapping and per-method scale; anchor fit is a calibration check |
| Automatic metric scale | At most 20% relative error on each scene's known element, withheld from automatic fitting | Predicted length versus 3 m/7 m; one reference result per scene, not a fabricated multi-distance median |
| Reduced distortion | Clear improvement in prespecified geometric/structural checks after anchor calibration, without material coverage loss | Multiview consistency and visible shapes; quantify distance-error reduction where independent extra measurements exist |
| Better 3D reconstruction | Clear preference in full-quality structural and appearance comparisons; more coherent surfaces/detail with fewer artifacts and stable views | Separate geometry and appearance ratings, native exports, held-out comparisons and recorded limitations |
| Browser delivery (secondary) | Aim for 30 FPS and first usable view within 5 seconds under declared reference conditions | Optimization target after quality selection; a browser limitation does not negate a reconstruction-quality win |

Photometric deltas support the visual decision; a single metric is insufficient.
Report by scene and by temporal block. Frames are correlated observations, so
do not treat hundreds of frames as hundreds of independent scenes. Repeat the
baseline and provisional winners with three seeds where stochasticity matters;
report exact-run determinism where it does not. Keep failed/OOM runs in the table.

Use a single-pod smoke test to validate dependencies, exports, intrinsics and
cost before the full queue. Establish an hourly price and bounded run schedule
when credit is available; actual GPU capacity follows measured memory demand.
Run one frozen setting per method first. Spend tuning effort on the promising
candidate, then rerun the same setting on both scenes. Log setup separately
from inference, including peak memory, runtime and dollar cost.

## Presenting successful results

Present the full-quality reconstruction improvement through a small case-study
package: synchronized high-quality renders/videos, annotated geometry and
appearance comparisons, downloadable native reconstruction artifacts, and an
interactive comparison page. Extend the existing viewer's frame slider,
Actual/Render/Overlay controls and measurement UI for the interactive part.

The opening view shows the existing reconstruction and the selected improvement
side by side, with linked cameras, time, field of view and zoom. A source-image
panel anchors both to the actual footage. The initial camera path and detail
locations are chosen before seeing results. Labels identify the method, input
count, whether the view was used in reconstruction, and metric-scale status.

Each scene has three preset comparisons:

- **Overall scene:** a short synchronized path showing spatial coherence.
- **Detail:** fixed crops and neutral-shaded/depth views highlighting both
  surface/edge improvement and appearance improvement.
- **Measurement:** the actual saved 3 m/7 m element, with calibration mode and
  predicted values labelled; include independent checks where available.

Provide toggles for RGB, depth/normals, sparse/mesh geometry, and confidence or
coverage where the method exports it. Measurements should snap to evaluated
geometry; an attractive Gaussian surface alone is not certified measurement
geometry. Include one clearly labelled difficult view or remaining failure.

For every confirmed positive result, produce a compact evidence set:

1. a 15-30 second high-quality synchronized before/after video for sharing;
2. high-resolution annotated geometry and appearance images with matched views;
3. an interactive link opening the exact comparison state and native artifact links;
4. a small scorecard with the absolute values, change, input count and cost; and
5. a short explanation connecting the visible gain to a supported claim.

Possible story templates, filled only after measurement:

- “The measured 3 m/7 m element was predicted as X m before manual calibration.”
- “After calibration with the same measured element, these surfaces became more
  coherent and these duplicated/warped structures were reduced.”
- “The reconstruction preserves sharper detail and cleaner appearance across
  these held-out viewpoints.”

Use a numeric local-distance improvement claim only when an independent distance
supports it. Browser responsiveness can be a separate delivery achievement.

Lead the public case study with confirmed improvements. Keep an adjacent link
to the complete two-scene matrix, including failures, coverage and unresolved
cases. If a method helps only one scene, say so. Use “not tested” or “measurement
unavailable” for empty values. Do not build fake numeric results into mockups.

After selecting winners, make a full-input showcase reconstruction and label it
separately from the held-out benchmark. Preserve the scoring assets unchanged.

## Work required before the next GPU run

- Revise the machine manifest and bundle splits to this protocol.
- Use calibration_references.json, correct the cross-coordinate scale comparison,
  and recover/annotate the corresponding measured elements for new geometry.
- Implement exact crop/intrinsics mapping, explicit render-frame association,
  masks and reliable held-out camera handling; add actual LPIPS execution.
- Add fresh matched-input baseline commands and timestamp/segment-aware QA.
- Verify runner environment/build contracts, checkpoint revisions, and export
  completeness on a small GPU smoke test; the existing local unit tests do not
  establish end-to-end GPU readiness.
- Define the common result manifest (scene, split, method, revisions, input
  hashes, cameras, transform, unit convention, representation files, metrics,
  runtime/memory/cost, status and failure reason) used by the comparison page.
- Build the comparison page against empty/fixture states, then populate it with
  real results. The existing transfer archive predates these changes.
- Add full-quality artifact exports and geometry/appearance review views before
  making browser-optimized derivatives; record quality and delivery separately.

## Evidence reviewed

- Local metric-depth history: ../../docs/scale_runpod_results_2026-07-11.md.
- Local bundle metadata/hashes and existing scale, trajectory and image scorers.
- [Scal3R-ZJU](https://github.com/zju3dv/Scal3R): released test-time reconstruction.
- [MoGe](https://github.com/microsoft/MoGe): monocular geometry/depth estimation.
- [GLOMAP](https://github.com/colmap/glomap): global SfM and migration notice.
- [Surflo](https://github.com/Anttwo/Surflo): native plain/guided surface reconstruction.
- [QuerySplat](https://github.com/inspatio/querysplat): VGGT-Omega geometry,
  camera/depth exports and optional test-time optimization.
