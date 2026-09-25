# Building-constellation geolocation on the Chamaa aerial photo

**Test date:** 25 September 2026
**Method under test:** the Sainte-Maxime building-constellation camera search
([write-up](../../../docs/constellation_geolocation_method.md))
**Verdict: the method did not succeed.** In both search areas every candidate was
at least 300 m from the truth, and none was within the 50 m "correct" threshold.
Diagnostics show the reason is the **objective function**, not the search budget:
the method's cost scores the *true* camera pose **worse** than a wrong pose 480 m
away.

## Summary

| | 2 × 2 km | 4 × 4 km |
|---|---:|---:|
| Search box offset from truth (box centre − truth, E/N) | −477 m / −427 m | +29 m / −485 m |
| Truth distance to nearest box edge | 523 m | 1,515 m |
| Reference buildings (OSM, in box) | 316 | 859 |
| Image building detections (SAM 3, after filtering) | 51 | 51 |
| Distinct candidates returned | 3 | 7 |
| **Top candidate's error** | **482 m** | **455 m** |
| Closest candidate at any rank | 326 m (rank 3) | 314 m (rank 2) |
| **Rank of the correct location (≤ 50 m)** | **not found** | **not found** |
| Rank of any candidate within 100 m | not found | not found |
| Score margin, 2nd best − best (training cost, lower is better) | 1.285 | 0.169 |
| Best candidate's held-out cost | 7.26 | 7.32 |
| Shuffled-control held-out cost (median of 100) | 7.20 | 7.40 |
| Runtime (search, excluding one-off SAM and OSM) | 23.9 s | 32.0 s |
| Objective evaluations | 81,011 | 81,014 |

The held-out rows are the telling ones. The winning poses explain the reserved
buildings **no better than randomly shuffled detections do** (7.26 vs 7.20; 7.32
vs 7.40). At Sainte-Maxime the held-out cost was 1.18 against a shuffled median
of 5.47. There, the fit carried real information; here it does not.

### How confidence changes with area

There was no real confidence to lose. With a wrong winner and held-out costs
equal to the shuffled null in both boxes, the margins measure competition between
wrong answers rather than certainty about a right one. The trend is still clear,
though: going from 2 × 2 to 4 × 4 km, the number of distinct candidates rose from
3 to 7 and the winner's margin collapsed from 1.29 to 0.17, as the extra 543 map
buildings supplied more look-alike fits. A larger box gives the method more ways
to be wrong with equal conviction.

## Top candidates (primary model, `joint`)

Score = training assignment cost (the method's selection criterion; lower is better).
Inlier buildings = detections given a map identity by the overall one-to-one
assignment. Latitude/longitude are the **estimated footprint centre** (where the
image-centre ray meets the ground), which is what the ground truth describes.

**2 × 2 km** (only 3 distinct candidates: the 14 refined hypotheses collapsed onto 3 locations)

| Rank | Lat | Lon | Score | Held-out | Inlier buildings | Error | Heading |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.150050 | 35.198497 | 4.227 | 7.262 | 31 | 482 m | 75° |
| 2 | 33.151080 | 35.197942 | 5.512 | 8.055 | 25 | 553 m | 69° |
| 3 | 33.151500 | 35.200853 | 5.870 | 7.782 | 20 | 326 m | 260° |

**4 × 4 km**

| Rank | Lat | Lon | Score | Held-out | Inlier buildings | Error | Heading |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.149264 | 35.198804 | 4.425 | 7.317 | 27 | 455 m | 73° |
| 2 | 33.148446 | 35.206646 | 4.594 | 7.437 | 29 | 314 m | 260° |
| 3 | 33.151983 | 35.219341 | 5.698 | 7.740 | 21 | 1,485 m | 60° |
| 4 | 33.146948 | 35.206292 | 5.708 | 6.954 | 26 | 396 m | 267° |
| 5 | 33.148752 | 35.207930 | 5.753 | 7.270 | 24 | 414 m | 251° |

**Truth:** 33.149742 N, 35.203649 E (UTM 36N 705527.8 E, 3670049.5 N); the true heading is 223°.

### Secondary: the write-up's facade-height variant (`joint_offset`)

The write-up ran three model variants. `fixed` needs a prior focal-length estimate,
which this photo does not have. `joint_offset` adds a shared downward correction
α × building height, meant to move each roof centre toward the centre of the
visible building. **I ran it after seeing the primary fail**, so treat it as
exploratory. It also failed:

| | 2 × 2 km | 4 × 4 km |
|---|---:|---:|
| Top candidate's error | 466 m | 509 m |
| Closest candidate at any rank | 414 m | 460 m |
| Rank of correct / within 100 m | not found / not found | not found / not found |
| Margin | 0.322 | 0.239 |
| Held-out cost vs shuffled median | 7.31 vs 7.39 | 7.81 vs 7.26 |
| Runtime | 28.3 s | 42.2 s |

Full top-5 tables are in `2x2km/joint_offset/evaluation.json` and
`4x4km/joint_offset/evaluation.json`.

## Why it failed: search or model?

A failed search could mean the optimiser never reached the right place, or that
the cost does not prefer the right place even when it gets there. They call for
opposite fixes, so I tested both directly. These two diagnostics **use the ground
truth and are not part of the blind result.**

| Pose evaluated | Footprint error | Training cost | Held-out cost | Inlier buildings |
|---|---:|---:|---:|---:|
| **True camera pose**, recovered from the alignment | ≈ 0 m | **4.73** | 7.48 | 27 |
| True pose + best facade correction (α = 0.40) | ≈ 0 m | 4.47 | 7.30 | 28 |
| Best pose with footprint held within 60 m of truth, orientation free | 13 m | 4.15 | 7.16 | 27 |
| **Blind winner, 2 × 2 km** | 482 m | **4.23** | 7.26 | 31 |

The true pose came from the alignment itself. The homography maps a grid of 117
photo pixels to ground points (z from the DEM), and a PnP solve recovers the
camera (3.6 px RMS): focal 1,594 px, heading 223°, pitch −30.6°, 274 m above
the viewed ground. My reparametrisation reproduces that camera to within 2 m.

**At the true pose the cost is higher than at the wrong winner.** This is a model
failure. More iterations, more seeds or a heading prior could not fix it, because
the optimiser already prefers the wrong answer. (All six global runs did hit the
240-iteration cap without converging, but that is not what decided the outcome.)
The within-60 m oracle recovered the right orientation (heading 227°, pitch
−30.7°), which confirms that the search can find the right basin. It just does
not score it best.

Three mechanisms, all visible in the figures:

1. **A density attractor.** The blind winner maps the photo's scattered villas
   onto buildings inside a dense fenced compound north-west of the truth
   ([figure](figures/2x2km_image_vs_constellation.jpg)). With dozens of buildings
   packed together, a map point falls within tolerance of almost any detection. The
   cost checks that each detection is near some map point; it has no term for how
   dense the map is locally, and none for whether the spacing between matched
   buildings agrees. The compound is genuinely in the photo, at the top-right
   edge; the winner spreads it across the whole frame. All four blind winners (two
   areas × two model variants) look east, at headings 73–76°, against a true 223°.
   This is the same density trap the earlier quad-hashing constellation attempt on
   this photo hit (2.1 km off): raw agreement favours the densest place.
2. **Parallax between mask centroids and roof centres.** At the true pose the
   projected OSM roof centres land on the correct houses, but consistently
   10–20 px above-left of the SAM mask centroids (median 13 px;
   [figure](figures/diagnostic_true_pose_projection.jpg)). From 274 m up at −31°,
   a mask covers roof *plus* visible facade, so its centroid sits between roof and
   base. The cost's tolerance is σ = max(4 px, 0.16 × width), about 5–8 px here,
   so every correct pair pays 2–3σ. At Sainte-Maxime the houses were small and
   distant from a ground-level camera, and this offset was negligible. The facade
   correction removes only part of it (4.73 → 4.47).
3. **Unmatched clutter on both sides.** About 8 detections have no map
   counterpart (a blue-roofed shed, a stone yard, trees), and about 5 OSM
   buildings are not visible (one is now a cleared lot). Each unmatched detection
   costs the maximum 9, which flattens the difference between right and wrong poses.

The missing heading prior is **not** the cause. It makes the search space about
6.5× larger than Sainte-Maxime's, but the true pose already loses on cost.

## Comparison with Sainte-Maxime

| | Sainte-Maxime (write-up) | Chamaa (this test) |
|---|---|---|
| Outcome | **3.15 m** from the reference pin | **Failed**: ≥ 314 m, correct not found |
| Held-out cost vs shuffled median | 1.18 vs 5.47 | 7.26 vs 7.20 (no better than noise) |
| Camera | Phone, ground level (terrain + 2 m), near-horizontal | Aerial, 274 m up, pitch −31° |
| Heading prior | Yes, yaw −20..35° | None, 0–360° (blind) |
| Search area | 2.5 × 3.5 km rectangle | 2 × 2 km and 4 × 4 km, offset |
| Reference buildings | 2,159 IGN 3D roofs, **with measured roof elevations** | 316 / 859 OSM 2D footprints, height = DEM + 7 m (assumed) |
| Detections | 69 SAM instances (48 train / 21 reserve) | 51 (36 / 15) |
| Blind? | No: the pin was known in the conversation (the write-up says so) | Yes: the truth never enters the search |
| Runtime | ~58 s joint-model subtotal | 24–42 s per area and mode |

The write-up itself flagged two conditions this test hit directly. It listed as
**not established** "success with plain 2D OSM footprints and no roof heights",
and it asked for "a genuinely blind test". This is both, and on this scene the
method did not transfer. The geometry is also very different: a steep oblique
aerial view makes the roof-versus-mask-centroid parallax large, where a
ground-level phone view kept it small.

## What was reproduced, and what had to change

**Unchanged from the original code** (`original_sainte_maxime/joint_building_constellation.py`, `refine_building_constellations.py`):
the rotation and projection conventions; SAM 3 detection with prompts building/house/roof,
threshold 0.3 and mask-IoU dedup 0.72, run with the original script; mask centroids as
observations, with bbox width as the size cue; the near-duplicate rule; the 70/30 train/reserve
split with seed 7; footprint centroids, minAreaRect half-axes and the 45–2,500 m² filter; the
coarse nearest-neighbour cost; the one-to-one assignment with dummy "unmatched" slots, its σ
and its width term; DE with 3 seeds × popsize 16 × 240 iterations; Powell polishing of the
best 12; local DE refinement of the best 2; training-only selection; reserved detections
barred from reusing training map identities; and the 100 x-shuffled controls.

**Changed, because the scene required it** (decided before any evaluation):

1. **Camera height.** The original fixes the camera at terrain + 2 m (a phone on the ground).
   This photo is aerial, so height above the viewed ground is a searched parameter, 40–1,200 m.
2. **Parametrisation.** The search state is the ground point hit by the image-centre ray,
   bounded to the search box, rather than the camera position. The camera is then placed back
   along that ray. This makes "search area" mean where the photo looks, and the estimate
   compares directly with the footprint-centre truth.
3. **Heading.** No prior: yaw is the full 0–360°; pitch −80..−12°, roll ±10°, focal
   400–3,000 px. The original used yaw −20..35° from knowing the view faced north.
4. **Map heights.** IGN supplied roof elevations. Here roof centre = 10 m DEM +
   `building:levels` × 3 m when tagged, otherwise 7 m.
5. **Image filters**, rescaled to this photo from the detection statistics alone: score ≥ 0.35
   (unchanged), mask area 100–9,000 px, bbox width 8–160 px and height < 200 px, 5 px edge
   margin, and the bottom black band excluded. The original used a 180–570 px horizontal band
   suited to a ground-level photo.
6. **Only the `joint` model** as primary (the write-up's designated primary), and `joint_offset`
   as secondary. `fixed` needs a prior focal estimate that does not exist here, and there was
   no cross-seeding between modes.
7. **No terrain-silhouette check.** The photo shows no sky, so there is no skyline to render.

**Reference source.** OSM coverage of this village is unusually good: nearly every house in
the photo has a footprint ([coverage check](figures/diagnostic_osm_coverage_on_photo.jpg); OSM
outlines projected into the photo through the ground-truth alignment, so it is a diagnostic
that never entered the search). So this failure is not explained by missing map data.

## Ground truth and blindness

* The truth is the footprint centre from the earlier LoFTR alignment
  (`data/alignment_results.json`: 225 inliers at 180°, control tile abstained, ~10 m
  conservative uncertainty). `make_truth_and_areas.py` reproduces the alignment's two recorded
  numbers (image centre within 1 m, supplied coordinate exactly) before accepting it.
* The boxes are the truth plus a random offset (seed 20260925), and the offsets are recorded in
  `data/truth.json`. The search reads only `data/search_areas.json`, which holds the box
  bounds. Only `evaluate.py` and the two `diagnose_*.py` scripts read `data/truth.json`.
* "Correct" was defined as ≤ 50 m before results were examined.

## Figures

| | 2 × 2 km | 4 × 4 km |
|---|---|---|
| Area map with all candidates | [area](figures/2x2km_area_candidates.jpg) | [area](figures/4x4km_area_candidates.jpg) |
| Top candidate's matched buildings on the orthophoto | [ortho](figures/2x2km_top_matched_on_ortho.jpg) | [ortho](figures/4x4km_top_matched_on_ortho.jpg) |
| Image beside the matched constellation | [side by side](figures/2x2km_image_vs_constellation.jpg) | [side by side](figures/4x4km_image_vs_constellation.jpg) |
| Same for `joint_offset` | `figures/2x2km_joint_offset_*.jpg` | `figures/4x4km_joint_offset_*.jpg` |

Diagnostics (use the truth): [map points at the true pose vs detections](figures/diagnostic_true_pose_projection.jpg),
[OSM coverage projected into the photo](figures/diagnostic_osm_coverage_on_photo.jpg).

![Image vs matched constellation, 2 × 2 km](figures/2x2km_image_vs_constellation.jpg)

![Candidates over the 4 × 4 km area](figures/4x4km_area_candidates.jpg)

![At the true pose, OSM roof centres sit up-left of the SAM centroids](figures/diagnostic_true_pose_projection.jpg)

## What would have to change

These are hypotheses, **not tested here**:

* **A density-aware score.** Normalise agreement by local map density (a Poisson-style null
  with a multiple-testing penalty, as this repo's ortho-matching work recommends), so a dense
  compound cannot win by offering a neighbour to everything.
* **Relative-geometry consistency.** Penalise matched sets whose inter-building spacing and
  ordering disagree with the photo. The current cost scores each pair independently.
* **A mask model that fits the view.** Predict where the visible roof-plus-facade centroid falls
  for each building, or widen σ for steep oblique views. The facade correction captured only
  part of the offset.
* **Evaluate on a second, unrelated blind scene** before attributing this to Chamaa alone.

## Reproduce

```bash
cd geolocation/chamaa_constellation
python3 make_truth_and_areas.py            # truth + offset boxes (seeded)
python3 fetch_reference.py                 # OSM buildings for the boxes (cached in data/)
# SAM 3 detections are committed in data/sam_photo1/segments.json. To regenerate them, run the
# original script from the branch where it lives (it imports tools/ortho_matching/extract_scene_geometry.py):
#   python3 tools/geolocation/segment_cross_modal_anchors.py --ground <this dir>/data/chamaa_photo1.png \
#     --ortho <this dir>/data/orthophoto_1km_040mpp.jpg --out-dir <this dir>/data/sam_photo1 --only ground \
#     --semantics building --building-prompts building,house,roof --threshold 0.3 --max-side 1280
python3 test_chamaa_constellation.py       # 10 geometry tests
python3 chamaa_constellation.py --area 2x2km
python3 chamaa_constellation.py --area 4x4km
python3 chamaa_constellation.py --area 2x2km --mode joint_offset
python3 chamaa_constellation.py --area 4x4km --mode joint_offset
python3 evaluate.py && python3 evaluate.py --mode joint_offset
python3 diagnose_oracle.py && python3 diagnose_true_pose.py   # use the truth; not blind
```

Dependencies: numpy, scipy, opencv, pyproj, rasterio. Regenerating detections needs SAM 3
(`facebook/sam3`) and the original branch's segmentation wrapper; the committed detections make
that optional. Ortho cuts need the
`tzafon_3_26.ecw` layer and the `ecw2tiff` converter (`ortho_io.py`); the DEM is
`height_evoPilot.tif` (10 m, UTM 36N). The last three are local, not in the repository.
