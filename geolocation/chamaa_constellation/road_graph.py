#!/usr/bin/env python3
"""Road networks as junctions with branch tangents, in either modality.

Following the road-network geolocalization literature (Li et al.,
arXiv:1906.12174; the projective-invariant contour feature, Remote Sensing
13(3):490), the matching primitive is not the road curve but the *junction plus
its incident tangents*. Under a projective transform, lengths and angles are not
invariant, so curve geometry gives nothing stable to match on; a junction is a
distinguished point, and the lines through it stay lines.

Two properties are used later and are worth naming here:
  * lines map to lines, so the intersection of two corresponding lines is a
    corresponding point - which manufactures extra point correspondences out of
    tangents alone;
  * the cyclic order of branches around a junction survives an
    orientation-preserving projective map, which collapses the branch-assignment
    search from all permutations to a handful of rotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Junction:
    xy: np.ndarray               # position in the source frame
    dirs: np.ndarray             # (k,2) unit tangents, sorted by angle
    degree: int


def _neighbours(pts: set, p):
    y, x = p
    return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in pts]


def junctions_from_mask(mask: np.ndarray, walk_px: int = 12,
                        merge_px: int = 6, min_branch_px: int = 6) -> list[Junction]:
    """Skeletonise a road mask and return junctions with their branch tangents."""
    sk = skeletonize(mask > 0)
    pts = {(int(y), int(x)) for y, x in zip(*np.nonzero(sk))}
    if not pts:
        return []
    deg = {p: len(_neighbours(pts, p)) for p in pts}
    raw = [p for p, d in deg.items() if d > 2]

    clusters: list[list] = []
    for p in raw:
        for c in clusters:
            if any(abs(p[0] - q[0]) <= merge_px and abs(p[1] - q[1]) <= merge_px for q in c):
                c.append(p)
                break
        else:
            clusters.append([p])

    out: list[Junction] = []
    for c in clusters:
        cy = float(np.mean([p[0] for p in c]))
        cx = float(np.mean([p[1] for p in c]))
        core = {p for p in pts
                if abs(p[0] - cy) <= merge_px and abs(p[1] - cx) <= merge_px}
        # Branch seeds are skeleton pixels just OUTSIDE the junction core that
        # touch it. Seeding from the core's centre instead finds nothing: the
        # centre's neighbours are themselves core pixels, so every walk is
        # rejected before it starts.
        seeds = set()
        for q in core:
            for nb in _neighbours(pts, q):
                if nb not in core:
                    seeds.add(nb)
        dirs = []
        for seed in seeds:
            prev, cur, steps = None, seed, 0
            visited = {seed}
            while steps < walk_px:
                nxt = [q for q in _neighbours(pts, cur)
                       if q != prev and q not in core and q not in visited]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                visited.add(cur)
                steps += 1
            if steps < min_branch_px:
                continue
            v = np.array([cur[1] - cx, cur[0] - cy], float)
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            dirs.append(v / n)
        if len(dirs) < 2:
            continue
        keep = []
        for d in dirs:
            if not any(float(np.dot(d, k)) > 0.94 for k in keep):
                keep.append(d)
        if len(keep) < 2:
            continue
        keep = sorted(keep, key=lambda d: np.arctan2(d[1], d[0]))
        out.append(Junction(xy=np.array([cx, cy]), dirs=np.array(keep), degree=len(keep)))
    return out


def rasterise_world(points_xy: np.ndarray, gsd: float, pad: float = 5.0,
                    close_px: int = 5):
    xmin, ymin = points_xy.min(0) - pad
    xmax, ymax = points_xy.max(0) + pad
    W = int((xmax - xmin) / gsd) + 1
    H = int((ymax - ymin) / gsd) + 1
    g = np.zeros((H, W), np.uint8)
    g[((ymax - points_xy[:, 1]) / gsd).astype(int),
      ((points_xy[:, 0] - xmin) / gsd).astype(int)] = 1
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return g, (xmin, ymin, xmax, ymax)


def line_through(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Homogeneous line through point p with direction d."""
    q = p + d
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def line_intersection(l1: np.ndarray, l2: np.ndarray, min_w: float = 1e-6):
    p = np.cross(l1, l2)
    if abs(p[2]) < min_w:
        return None                      # near-parallel: intersection at infinity
    return np.array([p[0] / p[2], p[1] / p[2]])


def tuple_correspondence(jA, jB, ja, jb, rotA: int, rotB: int,
                         max_coord: float = 1e5):
    """Four corresponding points from two junctions and their tangents.

    The junctions give two correspondences directly. The other two come from
    intersecting branch lines across the pair: lines map to lines, so the
    meeting point of two corresponding lines is itself a corresponding point.
    That is what turns tangent directions - which are not themselves projective
    invariants - into usable point correspondences.
    """
    dA = np.roll(jA.dirs, 0, axis=0)
    dB = np.roll(jB.dirs, 0, axis=0)
    da = np.roll(ja.dirs, rotA, axis=0)
    db = np.roll(jb.dirs, rotB, axis=0)
    k = min(len(dA), len(da), 2)
    m = min(len(dB), len(db), 2)
    if k < 2 or m < 2:
        return None
    src = [jA.xy, jB.xy]
    dst = [ja.xy, jb.xy]
    for i in range(2):
        s = line_intersection(line_through(jA.xy, dA[i]), line_through(jB.xy, dB[i]))
        d = line_intersection(line_through(ja.xy, da[i]), line_through(jb.xy, db[i]))
        if s is None or d is None:
            return None
        if np.max(np.abs(s)) > max_coord or np.max(np.abs(d)) > max_coord:
            return None
        src.append(s)
        dst.append(d)
    return np.array(src, np.float32), np.array(dst, np.float32)
