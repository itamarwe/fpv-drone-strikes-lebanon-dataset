#!/usr/bin/env python3
"""Can off-the-shelf image embeddings match a rooftop across modalities (oblique photo vs
nadir ortho), and pick it out from the other houses in the area?

DIAGNOSTIC - uses the ground truth to pair photo roofs with map buildings.

Pairs: every filtered SAM 3 roof detection (the 51 the search uses) is warped to the ortho
with the LoFTR photo->ortho homography; it is paired with the OSM footprint it overlaps most
(>= 30% of the footprint covered, one-to-one). Gallery: ortho crops (tzafon_3_26, 0.4 m/px)
of all OSM buildings in the 4x4 km box (859), and the 2x2 km subset (316).

Settings (what the matcher is told about the camera):
  A blind          raw photo crop vs north-up ortho crops, best of 8 rotations (45 deg steps)
  B heading known  raw photo crop vs ortho crop rotated so the camera's forward points up and
                   squashed vertically by sin(30 deg) (the view's foreshortening)
  C pose known     photo crop warped to top-down (north-up) vs north-up ortho crop - the case
                   where a constellation candidate supplies the pose
Crop sizes: roof (1.5x the building's size, >= 20 m) and context (3.5x, >= 60 m).
Models: DINOv2-B, DINOv3-L (web), DINOv3-L (SAT-493M satellite), SigLIP2-B. Cosine similarity
of the pooled embedding. Metric: rank of the true building among all gallery buildings.
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

import chamaa_constellation as cc
from make_truth_and_areas import TILE_GSD, TILE_ORIGIN
from ortho_io import cut_ortho, read_rgb

OUT = cc.RESULTS / "roof_embeddings"
SCRATCH = Path("/private/tmp/claude-501/-Users-itamarwe-Documents-code-fpv-drone-strikes-lebanon-dataset/"
               "8ce82b6a-5cc1-4c7e-b4f8-950ccaa22836/scratchpad")
ORTHO_GSD = 0.4
HEADING, PITCH = 223.0, -30.6          # truth pose (diagnose_true_pose.py) - setting B only
SIZES = {"roof": (1.5, 20.0), "context": (3.5, 60.0)}
N = 224
MODELS = {
    "dinov2-b": "facebook/dinov2-base",
    "dinov3-l-web": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-l-sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "siglip2-b": "google/siglip2-base-patch16-224",
}


# ---------------------------------------------------------------- geometry
def homography():
    al = json.loads((cc.DATA / "alignment_results.json").read_text())
    H = np.array(al["loftr_query_to_northwest_tile_homography"], float)
    # photo px -> UTM
    T = np.array([[TILE_GSD, 0, TILE_ORIGIN[0]], [0, -TILE_GSD, TILE_ORIGIN[1]], [0, 0, 1]])
    return T @ H


def to_utm(Hu, pts):
    p = np.c_[pts, np.ones(len(pts))] @ Hu.T
    return p[:, :2] / p[:, 2:]


def pairs_from_truth(pr, Hu):
    """photo detection index -> map index, by footprint overlap of the warped SAM mask."""
    polys = [Polygon(m["polygon_utm"]).buffer(0) for m in pr.map]
    cand = []
    for i, det in enumerate(pr.observed):
        mask = cv2.imread(str(cc.DATA / "sam_photo1/ground_masks" / f"{det['id']}.png"), 0)
        cs, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)[:, 0].astype(float)
        if len(c) < 3:
            continue
        wp = Polygon(to_utm(Hu, c)).buffer(0)
        for j, p in enumerate(polys):
            if p.intersects(wp):
                cov = p.intersection(wp).area / p.area
                if cov >= .3:
                    cand.append((cov, i, j))
    out, used_i, used_j = {}, set(), set()
    for cov, i, j in sorted(cand, reverse=True):
        if i not in used_i and j not in used_j:
            out[i] = (j, cov); used_i.add(i); used_j.add(j)
    return out


def building_frame(m):
    v = np.array(m["polygon_utm"]); c = Polygon(v).centroid
    ext = float(np.max(np.ptp(v, axis=0)))
    return np.array([c.x, c.y]), ext


# ---------------------------------------------------------------- crops
class Ortho:
    def __init__(self, bounds):
        pad = 80.0
        self.b = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
        tile = cut_ortho(self.b, SCRATCH / "ortho_4x4_tzafon_3_26_040.tif", gsd=ORTHO_GSD)
        self.img = np.ascontiguousarray(read_rgb(tile.path))     # cv2 copies non-contiguous arrays per call

    def crop(self, centre, side_m, rot_deg=0.0, squash=1.0):
        """N x N crop centred on a UTM point: the ortho rotated CCW by rot_deg (so azimuth rot_deg
        points up), covering side_m across and side_m / squash in depth (oblique foreshortening)."""
        x = (centre[0] - self.b[0]) / ORTHO_GSD; y = (self.b[3] - centre[1]) / ORTHO_GSD
        s = side_m / ORTHO_GSD
        R = np.vstack([cv2.getRotationMatrix2D((x, y), rot_deg, 1.0), [0, 0, 1]])
        S = np.array([[N / s, 0, 0], [0, N * squash / s, 0], [0, 0, 1]])
        T = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]]); Tb = np.array([[1, 0, N / 2], [0, 1, N / 2], [0, 0, 1]])
        M = (Tb @ S @ T @ R)[:2]      # rotate about the centre, move it to the origin, scale, recentre
        return cv2.warpAffine(self.img, M, (N, N), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))


def photo_crop_raw(photo, det, scale, min_px=40):
    x0, y0, x1, y1 = det["bbox_xyxy"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(min_px, scale * max(x1 - x0, y1 - y0))
    M = np.array([[N / side, 0, N / 2 - cx * N / side], [0, N / side, N / 2 - cy * N / side]])
    return cv2.warpAffine(photo, M, (N, N), flags=cv2.INTER_AREA)


def photo_crop_rectified(photo, Hu, centre, side_m):
    """Top-down, north-up crop of the photo around a UTM point (photo warped via the homography)."""
    g = side_m / N
    A = np.array([[g, 0, centre[0] - side_m / 2], [0, -g, centre[1] + side_m / 2], [0, 0, 1]])  # out px -> UTM
    M = np.linalg.inv(Hu) @ A                                                                     # out px -> photo px
    return cv2.warpPerspective(photo, M, (N, N), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


# ---------------------------------------------------------------- embeddings
def load_model(name, device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MODELS[name])
    model = AutoModel.from_pretrained(MODELS[name]).to(device).eval()
    return proc, model


@torch.no_grad()
def embed(name, proc, model, images, device, bs=64):
    out = []
    for k in range(0, len(images), bs):
        batch = list(images[k:k + bs])
        inp = proc(images=batch, return_tensors="pt", do_resize=True, size={"height": N, "width": N},
                   do_center_crop=False).to(device)
        if name.startswith("siglip"):
            f = model.get_image_features(pixel_values=inp["pixel_values"])
            f = f.pooler_output if hasattr(f, "pooler_output") else f
        else:
            o = model(pixel_values=inp["pixel_values"])
            first = 5 if name.startswith("dinov3") else 1              # DINOv3: CLS + 4 registers
            f = torch.cat([o.pooler_output, o.last_hidden_state[:, first:].mean(1)], 1)   # CLS + mean patch
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


# ---------------------------------------------------------------- main
def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    log("loading detections, OSM buildings, DEM")
    pr = cc.Problem(areas["4x4km"])
    b2 = areas["2x2km"]["bounds_utm"]
    Hu = homography()
    photo = cv2.cvtColor(cv2.imread(str(cc.DATA / "chamaa_photo1.png")), cv2.COLOR_BGR2RGB)
    assert photo.shape[:2] == (cc.IMAGE_H, cc.IMAGE_W), photo.shape
    log("pairing photo roofs with OSM footprints (truth homography)")
    pairs = pairs_from_truth(pr, Hu)
    log(f"{len(pr.observed)} photo roofs, {len(pairs)} paired with an OSM building; gallery {len(pr.map)} (4x4 km)")
    frames = [building_frame(m) for m in pr.map]
    in2 = np.array([b2[0] <= c[0] <= b2[2] and b2[1] <= c[1] <= b2[3] for c, _ in frames])
    log("reading the 4x4 km ortho")
    ortho = Ortho(pr.bounds_utm)
    log(f"ortho {ortho.img.shape}")
    squash = math.sin(math.radians(-PITCH))

    crops = {}   # (setting, size) -> dict(query=[...], gallery=[[...rotations]])
    qi = sorted(pairs)
    for size, (k, lo) in SIZES.items():
        log(f"cutting {size} crops")
        gal_side = [max(lo, k * ext) for _, ext in frames]
        # query side in metres: from the paired building would leak the truth; use the detection's own warped extent
        q_side = []
        for i in qi:
            x0, y0, x1, y1 = pr.observed[i]["bbox_xyxy"]
            w = to_utm(Hu, np.array([[x0, y1], [x1, y1]])); q_side.append(max(lo, k * float(np.linalg.norm(w[1] - w[0]))))
        q_centre = [to_utm(Hu, np.array([pr.observed[i]["centroid_xy"]]))[0] for i in qi]
        crops[("A", size)] = dict(query=[photo_crop_raw(photo, pr.observed[i], k) for i in qi],
                                  gallery=[[ortho.crop(c, s, r) for r in range(0, 360, 45)]
                                           for (c, _), s in zip(frames, gal_side)])
        crops[("B", size)] = dict(query=crops[("A", size)]["query"],
                                  gallery=[[ortho.crop(c, s, HEADING, squash)] for (c, _), s in zip(frames, gal_side)])
        crops[("C", size)] = dict(query=[photo_crop_rectified(photo, Hu, c, s) for c, s in zip(q_centre, q_side)],
                                  gallery=[[g[0]] for g in crops[("A", size)]["gallery"]])
    # sanity sheet: the true pairs in every setting (check orientation conventions by eye)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, i in enumerate(qi[:10]):
        j = pairs[i][0]; cells = []
        for s in ("A", "B", "C"):
            cells += [crops[(s, "context")]["query"][n], crops[(s, "context")]["gallery"][j][0]]
        rows.append(np.hstack([np.pad(c, ((2, 2), (2, 2), (0, 0))) for c in cells]))
    cv2.imwrite(str(OUT / "true_pairs_sanity.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    log(f"crops done; sanity sheet -> {OUT / 'true_pairs_sanity.jpg'}")
    if "--crops-only" in sys.argv:
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    truth_j = np.array([pairs[i][0] for i in qi])
    results = []
    models = [m for m in MODELS if not any(a.startswith("--only=") for a in sys.argv)
              or m in next(a for a in sys.argv if a.startswith("--only="))[7:].split(",")]
    sims_keep = {}
    for mname in models:
        log(f"loading {mname}")
        proc, model = load_model(mname, device)
        for (setting, size), c in crops.items():
            log(f"{mname}: embedding setting {setting} {size} ({len(c['query'])} queries, "
                f"{len(c['gallery']) * len(c['gallery'][0])} gallery crops)")
            q = embed(mname, proc, model, c["query"], device)
            nrot = len(c["gallery"][0])
            flat = [im for g in c["gallery"] for im in g]
            G = embed(mname, proc, model, flat, device).reshape(len(c["gallery"]), nrot, -1)
            S = np.einsum("qd,grd->qgr", q, G).max(2)                 # best rotation
            for box, sel in (("4x4km", np.ones(len(frames), bool)), ("2x2km", in2)):
                idx = np.where(sel)[0]; pos = {j: k for k, j in enumerate(idx)}
                ranks = []
                for n, j in enumerate(truth_j):
                    if j not in pos:
                        continue
                    s = S[n, idx]; ranks.append(int((s > s[pos[j]]).sum()) + 1)
                ranks = np.array(ranks); G_n = len(idx)
                results.append(dict(model=mname, setting=setting, size=size, box=box, gallery=G_n,
                                    queries=len(ranks), median_rank=float(np.median(ranks)),
                                    median_percentile=float(np.median(ranks / G_n) * 100),
                                    top1=int((ranks == 1).sum()), top10=int((ranks <= 10).sum()),
                                    top5pct=int((ranks <= .05 * G_n).sum()), ranks=ranks.tolist()))
                r = results[-1]
                print(f"{mname:<14}{setting}  {size:<8}{box:<6} gallery {G_n:>4} | median rank {r['median_rank']:6.0f} "
                      f"({r['median_percentile']:4.1f}% ; chance 50%) | top1 {r['top1']:>2}/{len(ranks)} "
                      f"top10 {r['top10']:>2} top5% {r['top5pct']:>2}", flush=True)
            sims_keep[(mname, setting, size)] = S
        del model; torch.mps.empty_cache() if device == "mps" else None
    (OUT / "results.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - pairs come from the ground-truth homography",
        pairs={int(i): dict(map_index=int(pairs[i][0]), osm_id=pr.map[pairs[i][0]]["osm_id"], coverage=pairs[i][1])
               for i in qi}, results=results), indent=2, default=float))
    np.savez_compressed(SCRATCH / "roof_embedding_sims.npz",
                        **{"|".join(k): v for k, v in sims_keep.items()}, truth_j=truth_j, in2=in2)

    # retrieval sheet for the best configuration (4x4 km): query | truth (rank) | top 5
    best = min((r for r in results if r["box"] == "4x4km"), key=lambda r: r["median_rank"])
    S = sims_keep[(best["model"], best["setting"], best["size"])]
    c = crops[(best["setting"], best["size"])]; rows = []
    for n in range(min(8, len(qi))):
        order = np.argsort(-S[n]); j = truth_j[n]; rank = int((S[n] > S[n, j]).sum()) + 1
        cells = [c["query"][n], c["gallery"][j][0]] + [c["gallery"][k][0] for k in order[:5]]
        cells = [np.pad(x, ((3, 3), (3, 3), (0, 0)), constant_values=255 if (x is cells[1]) else 0) for x in cells]
        row = np.hstack(cells)
        cv2.putText(row, f"truth rank {rank}/{len(frames)}", (236, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        rows.append(row)
    cv2.imwrite(str(OUT / f"retrieval_best_{best['model']}_{best['setting']}_{best['size']}.jpg"),
                cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"best (4x4 km): {best['model']} setting {best['setting']} {best['size']} median rank {best['median_rank']}")


if __name__ == "__main__":
    main()
