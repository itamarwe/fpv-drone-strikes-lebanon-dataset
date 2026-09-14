# Two-scene execution — 2026-09-08, continued 2026-09-09

Status: GPU execution stopped on 2026-09-08 at 12:02 UTC; benchmark incomplete.
Local continuation on 2026-09-09 is recorded in the final section of this file.

RunPod confirmed pod `4sf4shthsamb3f` stopped at 12:02:48 UTC and subsequently
deleted. A follow-up lookup returned 404 (pod not found). Its ephemeral disks
were removed; remote-only files are not recoverable from that pod. Previously
downloaded local artifacts remain unchanged, and execution logs were retrieved
to `results/execution_logs/` before shutdown. There is no remaining pod charge
from this instance. RunPod's pod billing ledger records $2.127364 for the
instance across the 10:00, 11:00, and 12:00 UTC buckets. This includes the
provider's recorded disk allocation and supersedes the earlier $2.09
runtime-only estimate.

Earlier run records and generated reviews are partial artifacts, not a final
validated recommendation. Nothing was published or committed.

Starting available balance:
approximately $19.99. One A100-SXM4 80 GB pod was provisioned at a quoted
$1.59/hour; this is not a final invoice or an all-in storage estimate.

- Pod: `4sf4shthsamb3f`, `fpv-two-scene-benchmark-20260908`.
- Created: 2026-09-08 10:43:54 UTC.
- Requested cloud stop deadline: 2026-09-08 14:45 UTC.
- Requested cloud termination deadline: 2026-09-08 16:00 UTC.
- Parent orchestrator must retrieve artifacts and terminate the pod before these deadlines.
- SSH verified; GPU is A100 80 GB, driver 580.126.16, Python 3.12.3.
- Image lacks conda and an nvcc executable on PATH; isolated environments are being prepared.

## Frozen input protocol

| Scene | Common inputs | Dense training inputs | Held-out views | All frames (in-sample only) |
|---|---:|---:|---:|---:|
| Scene A | 16 | 82 | 14 | 125 |
| Scene B | 16 | 101 | 14 | 149 |

Held-out candidates are segment-local temporal midpoints screened against common
inputs using exact hashes and a conservative grayscale-thumbnail difference test.
Dense inputs exclude those views, same-segment one-frame buffers, images passing
the near-duplicate screen against held-out views, and exact training duplicates.
The image-content test is a heuristic, not proof of independent viewpoints.
Exact indices and exclusions are preserved in each `scene_bundle.json`.

Held-out image scores remain unavailable until independent evaluation cameras,
crop/intrinsics transformations, masks, and rendered-image mappings are verified.
Training-view renders must be labeled in-sample.

## Original execution order — superseded

The sequence below records the stopped plan for provenance only.

1. Prepare isolated method environments and verify checkpoint downloads.
2. Run common16 native methods and MoGe-3 scale estimation; inspect failures and outputs.
3. Evaluate measured-element scale agreement on the fixed published reconstruction;
   do not compare raw scale coefficients across unrelated reconstruction gauges.
4. Select informative denser/refinement runs within the resource deadline.
5. Retrieve native assets, diagnostics, logs, provenance, and an honest result report.

SSMB + LightGlue remains a release-dependent gap; it must not be silently replaced
by a different frontend and reported as SSMB.

Three bounded faster-model agents handle runner readiness, metric evaluation,
and quality reporting. The parent owns paid resources, integration, scheduling,
and acceptance of results. No commits, uploads to the public site,
or changes to published scene assets were made.

## Verified closure — 2026-09-08

- `runpodctl pod list --all` returned no pods.
- `runpodctl network-volume list` returned no network volumes.
- Account spend is $0/hour; the available balance is $17.860455.
- The frozen local bundle passes `check_3d_pipeline_benchmark.py` with no errors.
- All four locally confirmed review claims reference existing evidence files.
- The Scene B QuerySplat TTO wrapper completed remotely, but its outputs were
  not retrieved. The attempt is recorded as unrecovered/partial rather than
  running or completed evidence.

## Continuation — 2026-09-09 (local, no paid resources)

Verified state at the start of the continuation:

- RunPod: no pods, no network volumes, $0/hour spend, available balance
  $17.860455 (GraphQL `myself.clientBalance`), unchanged since closure.
- The execution gate `runpod.paid_execution_allowed` is absent from
  `benchmark.json`, so the bundled runner refuses install/download/run.
- Local unit tests: 24 passed. `check_3d_pipeline_benchmark.py` reports no
  errors on the frozen bundle.
- The previous `review/inventory.json` pointed at a results root that no
  longer exists (`fpv-3d-pipeline-benchmark-dataset/...`). It was regenerated
  against this directory; the summary is unchanged (26 expected runs, 17 with
  outputs, 1 partial, 8 not run). `review/site/` was rebuilt with the
  incomplete-closure banner and the same four confirmed claims.

Work completed today:

- `results/SURFLO_COMMON16_FINDINGS.md` written from the four retrieved Surflo
  runs. Equal-input result is negative: 11-16k guided points, a fragmented
  wrapped mesh, texture refinement ran with zero iterations, and cameras are
  not exported. Scene A neutral and Scene B native/neutral turntables were
  rendered remotely but never retrieved.
- `results/QUERYSPLAT_COMMON16_FINDINGS.md` written from the three retrieved
  QuerySplat runs plus the wrapper log of the lost Scene B TTO run. Scene B
  feed-forward is the strongest equal-input appearance result (18.55 dB /
  0.692 SSIM on input views); Scene A collapses in its late low-texture
  segment; TTO adds +2.04 dB in-sample on Scene A. No held-out scores exist.

Evidence matrix after the continuation (common16 unless stated):

| Method | Scene A | Scene B | Findings document |
|---|---|---|---|
| MoGe-3 automatic scale (published VGGT geometry) | complete, fail (61% low) | complete, fail (69% low) | METRIC_SCALE_FINDINGS.md |
| MoGe-3 + GLOMAP | complete, 13/16 cameras | complete, 15/16 cameras | GLOMAP_COMMON16_FINDINGS.md |
| Scal3R-ZJU | complete, path drift late seg04 | complete, 5.95% Sim(3) RMSE | SCAL3R_COMMON16_FINDINGS.md |
| VGGT-Omega matched control (via QuerySplat export) | complete | complete | VGGT_OMEGA_MATCHED_CONTROL_FINDINGS.md |
| Surflo plain / guided | complete, negative | complete, negative | SURFLO_COMMON16_FINDINGS.md |
| QuerySplat feed-forward | complete | complete | QUERYSPLAT_COMMON16_FINDINGS.md |
| QuerySplat TTO | complete | unrecovered | QUERYSPLAT_COMMON16_FINDINGS.md |
| Held-out (heldout profile) photometric scores | not run | not run | adapter exists, never executed |
| dense_train Scal3R / GLOMAP / VGGT control / Surflo | not run | not run | — |
| SSMB + LightGlue | blocked upstream | blocked upstream | — |

Decisions supported so far: automatic metric scale is a completed negative
result; manual 3 m / 7 m anchors stay authoritative. Geometry and appearance
decisions remain open because every appearance number is in-sample and no
dense-input run exists.

### Proposed GPU session 2 (requires explicit approval and reopening the gate)

Goal: convert the open decisions into held-out and dense-input evidence with a
bounded spend. Everything below reuses the frozen bundle and existing scripts.

1. Provision one A100-SXM4 80 GB pod (last quote $1.59/hour; the previous
   1 h 19 min instance billed $2.13 including disk). Set a cloud stop deadline
   of 3 hours and a termination deadline of 3.5 hours at creation time.
2. Setup and checkpoint download: about 35-45 minutes based on the 2026-09-08
   timeline (pod created 10:43 UTC, first method run 11:16 UTC).
3. Runs, in priority order, retrieving each result directory immediately after
   it completes rather than at the end:
   - Scene B QuerySplat TTO rerun (about 4 minutes).
   - Held-out proxy evaluation for QuerySplat feed-forward and TTO on both
     scenes with LPIPS on CUDA (about 15 minutes total).
   - Scal3R `dense_train` on both scenes (82 and 101 views; minutes each).
   - GLOMAP `dense_train` on both scenes with CPU SIFT exhaustive matching
     (about 10-15 minutes each; the longest item).
   - VGGT-Omega `dense_train` control export on both scenes, if memory allows.
   - Surflo guided `dense_train` with texture refinement enabled, one scene
     first, second only if the first is informative.
4. Expected cost: about $4 for a 2.5-hour session; hard cap $6, which leaves
   more than $11 of balance. Abort the session if setup has not completed
   within 60 minutes.

Nothing in this continuation was committed, uploaded, or published.

## GPU session 2 — 2026-09-09 (in progress)

- User approved the full session at about 07:40 UTC with a $6 hard cap.
- Pod `vw53h2ac3c67bl`, `fpv-two-scene-benchmark-s2-20260909`, A100-SXM4 80 GB
  SECURE, $1.59/hour, 40 GB container disk, 150 GB volume, created
  2026-09-09 07:45:09 UTC. Cloud stop-after 10:45:08 UTC and terminate-after
  11:15:08 UTC were set at creation. Starting balance $17.860455.
- Progress and closure are appended below as they happen.

### Session 2 closure — 2026-09-09 10:07 UTC

- Pod `vw53h2ac3c67bl` was stopped at about 10:05 UTC and deleted at about
  10:07 UTC after a file-by-file size check confirmed all 2,146 remote result
  files were retrieved. `runpodctl pod list --all` and the network-volume list
  are empty; spend is $0/hour.
- Balance moved from $17.860455 to $14.282628: session cost $3.577827 for
  about 2 h 22 min, within the $6 cap. Setup (base plus four environments and
  all checkpoints) took 07:58-08:51 UTC; every one of the 26 queued steps and
  the three follow-up jobs exited 0. Remote logs are in
  `results/execution_logs_session2/`.
- Completed on this pod: QuerySplat common16 feed-forward and TTO reruns on
  both scenes (recovering the lost Scene B TTO), held-out proxy evaluation with
  LPIPS for all four, `dense_train` Scal3R, MoGe-3 + GLOMAP, VGGT-Omega
  control (via QuerySplat feed-forward export) and QuerySplat feed-forward on
  both scenes with held-out proxy scores, and Surflo guided `dense_train` with
  0 and 500 texture iterations on both scenes.
- Not done: Surflo dense render review (no Blender on this pod); QuerySplat
  dense TTO; three-seed repeats; the `full` profiles; SSMB (still unreleased).
- Findings written: `results/QUERYSPLAT_COMMON16_FINDINGS.md` (held-out
  section), `results/DENSE_TRAIN_FINDINGS.md`, and a dense section in
  `results/SURFLO_COMMON16_FINDINGS.md`.

Evidence matrix after session 2:

| Method | common16 A / B | dense_train A / B | Held-out appearance |
|---|---|---|---|
| MoGe-3 automatic scale | fail / fail | fail (via GLOMAP) / fail | n/a |
| MoGe-3 + GLOMAP | 13/16, 4.2% / 15/16, 4.5% | 79/82, 3.6% / 101/101, 4.7% | n/a |
| Scal3R-ZJU | 16/16, 37.9% / 16/16, 5.95% | 82/82, 19.2% / 101/101, 23.8% | n/a |
| VGGT-Omega matched control | 6.1% / 2.5% | 4.3% / 1.9% | n/a |
| QuerySplat feed-forward | complete / complete | complete / complete | A 12.9 -> 13.6 dB; B 18.1 -> 18.8 dB |
| QuerySplat TTO | complete / complete (recovered) | not run | A 12.8 dB; B 18.7 dB |
| Surflo plain / guided | negative / negative | guided negative (0 and 500 iters) | unavailable (no cameras) |
| SSMB + LightGlue | blocked | blocked | n/a |

Percentages are Sim(3) path RMSE as a fraction of the published path's RMS
radius. Decisions: automatic metric scale rejected; GLOMAP is the preferred
geometry scaffold at dense input; QuerySplat feed-forward on dense input is
the Scene B presentation candidate; Scene A has no acceptable appearance
result; Surflo and Scal3R are not recommended at these settings.

The repository copy of `benchmark.json` still has the execution gate closed.
Nothing was committed, uploaded, or published.

### Scene viewer exports — 2026-09-09

`tools/pipeline/build_benchmark_scene_viewers.py` exports 14 viewer scenes
(published VGGT, VGGT-Omega/QuerySplat depth cloud, Scal3R and GLOMAP sparse
for common16 and dense_train on both scenes) into `scenes/benchmark_3d_two_scene/`,
aligned to the published frame, with a comparison page at
`review/scene_comparison.html` served by `tools/local_scene_viewer_server.py`.
The viewer now defaults its camera panel to the actual frame when a scene has
no overlay renders. Surflo meshes are not exported (no cameras to align).
