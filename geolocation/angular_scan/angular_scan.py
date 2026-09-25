#!/usr/bin/env python3
"""Blind building-constellation camera search for near-horizontal views (ground phones, FPV approach).

Generalises the Chamaa decomposed search (geolocation/chamaa_constellation/blind_grid_search.py),
whose bird's-eye rectification breaks when the camera looks near the horizon over hilly terrain.

Scan (exhaustive, truth never read):
  for every camera position on a grid over the search box (and every height above ground, or
  terrain + 2 m for a hand-held phone):
    * every map building within range -> direction (azimuth, elevation) from the camera, using the
      DEM for ground heights; buildings below the terrain horizon along their azimuth are dropped
    * the buildings are drawn as tolerance discs on an azimuth x elevation raster
  for every (pitch, roll, focal) on a small grid:
    * every detection -> direction relative to a camera with heading 0
    * turning the heading is a pure azimuth shift (the rotation is about the vertical), so ALL
      headings are scored at once by gathering raster values at shifted columns
    * score z = (hits - expected) / sqrt(expected + 1); expected = n_detections x the raster
      coverage inside the camera's field of view at that heading (sampled at image grid points),
      so a pose gets no credit for pointing at a dense area
  keep the best peaks (distinct positions).
Verify: each peak is refined in the full camera model and scored by chance-corrected significance
S = -log10 P(Binomial(n, q) >= k) (k one-to-one matches, q = image fraction covered by tolerance
discs), exactly the houses term of geolocation/chamaa_constellation/significance.py.
"""
import json
import math
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import binom

T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def rotation(yaw, pitch, roll):
    """Rows: camera right, down, forward in world E, N, Up (same as the Sainte-Maxime/Chamaa code)."""
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.]); down = np.cross(forward, right)
    return np.array([right * np.cos(roll) + down * np.sin(roll), down * np.cos(roll) - right * np.sin(roll), forward])


# ------------------------------------------------------------------ scene
@dataclass
class Scene:
    name: str
    W: int
    H: int
    cx: float
    cy: float
    uv: np.ndarray              # (n, 2) undistorted (pinhole) detection centres
    wh: np.ndarray              # (n, 2) detection box sizes (px, undistorted scale)
    bxy: np.ndarray             # (m, 2) building centres (scene CRS, metres)
    bz: np.ndarray              # (m,) target point height (ground + fraction of building height)
    bext: np.ndarray            # (m,) building horizontal extent (m)
    dem_x: np.ndarray           # DEM grid (ascending x, ascending y), z[y, x]
    dem_y: np.ndarray
    dem_z: np.ndarray
    box: tuple                  # camera search box (e0, n0, e1, n1)
    heights: list               # AGL heights to scan; "ground" scenes use [2.0]
    pitches: list
    rolls: list
    focals: list
    step_m: float = 25.0
    r_min: float = 40.0
    r_max: float = 5000.0
    bin_deg: float = 0.1
    tol_px: float = 10.0        # detection-side tolerance radius (px)
    map_sigma_m: float = 12.0   # map-side tolerance (m), covers the position grid
    valid: np.ndarray = None    # optional (H, W) mask of usable image pixels (HUD, black borders)
    extra: dict = field(default_factory=dict)

    def ground(self, x, y):
        fx = (np.asarray(x) - self.dem_x[0]) / (self.dem_x[1] - self.dem_x[0])
        fy = (np.asarray(y) - self.dem_y[0]) / (self.dem_y[1] - self.dem_y[0])
        return map_coordinates(self.dem_z, [np.atleast_1d(fy), np.atleast_1d(fx)], order=1, mode="nearest")


S_: Scene = None   # worker global


def _init(scene):
    global S_
    S_ = scene


# ------------------------------------------------------------------ geometry helpers
def visible_directions(sc, cam):
    """Azimuth/elevation (deg), range and angular radius of the buildings visible from cam (E, N, Z)."""
    d = sc.bxy - cam[:2]; r = np.hypot(d[:, 0], d[:, 1])
    near = (r > sc.r_min) & (r < sc.r_max)
    idx = np.where(near)[0]; d = d[near]; r = r[near]
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    el = np.degrees(np.arctan2(sc.bz[idx] - cam[2], r))
    # terrain horizon: running max elevation angle along each azimuth bin
    NA, dr = 720, max(15.0, float(sc.dem_x[1] - sc.dem_x[0]))
    rs = np.arange(dr, sc.r_max, dr); A = np.radians((np.arange(NA) + .5) * 360 / NA)
    X = cam[0] + np.sin(A)[:, None] * rs[None]; Y = cam[1] + np.cos(A)[:, None] * rs[None]
    Z = sc.ground(X.ravel(), Y.ravel()).reshape(X.shape)
    E = np.degrees(np.arctan2(Z - cam[2], rs[None]))
    horizon = np.maximum.accumulate(E, axis=1)              # max terrain angle up to each range
    ai = (az / 360 * NA).astype(int) % NA
    ri = np.clip(((r - 30) / dr).astype(int) - 1, 0, len(rs) - 1)   # terrain strictly in front of it
    vis = (el > horizon[ai, ri]) | (r - 30 < dr)
    return idx[vis], az[vis], el[vis], r[vis]


def detection_dirs(sc, uv, f, pitch, roll):
    """Directions (az_rel, el) in degrees of pixel rays for a heading-0 camera."""
    R = rotation(0.0, pitch, roll)
    rays = np.c_[(uv[:, 0] - sc.cx) / f, (uv[:, 1] - sc.cy) / f, np.ones(len(uv))] @ R
    az = np.degrees(np.arctan2(rays[:, 0], rays[:, 1]))
    el = np.degrees(np.arctan2(rays[:, 2], np.hypot(rays[:, 0], rays[:, 1])))
    return az, el


def sample_pixels(sc, n=48):
    xs = np.linspace(.04, .96, n) * sc.W; ys = np.linspace(.04, .96, max(3, int(n * sc.H / sc.W))) * sc.H
    P = np.array([[x, y] for y in ys for x in xs])
    if sc.valid is not None:
        P = P[sc.valid[P[:, 1].astype(int), P[:, 0].astype(int)]]
    return P


# ------------------------------------------------------------------ scan
EL_LO, EL_HI = -45.0, 25.0          # elevation band of the raster (deg)
_FFT_CACHE = {}
LOCAL_WIN_PX = 150.0      # local-density window for the chance model (image pixels)
SIZE_CLASSES = [(lo, lo * 1.5) for lo in (4, 6, 9, 13.5, 20, 30, 45, 67.5, 101, 152)]   # detection width classes (px)


def _det_fft(sc, f, pitch, roll, shape, cls):
    """FFT of the raster of one detection size class (cls None = all) for a heading-0 camera; cached."""
    key = (f, pitch, roll, cls)
    if key not in _FFT_CACHE:
        from scipy import fft
        nel, naz = shape
        pts = sample_pixels(sc) if cls == "samples" else (sc.uv if cls is None else sc.uv[_class_members(sc, cls)])
        az, el = detection_dirs(sc, pts, f, pitch, roll)
        D = np.zeros(shape, np.float32)
        ei = np.clip(np.round((el - EL_LO) / sc.bin_deg).astype(int), 0, nel - 1)
        ai = np.round(az / sc.bin_deg).astype(int) % naz
        np.add.at(D, (ei, ai), 1.0)
        _FFT_CACHE[key] = (np.conj(fft.rfft2(D)), len(pts))
    return _FFT_CACHE[key]


def _class_members(sc, cls):
    lo, hi = SIZE_CLASSES[cls]
    return (sc.wh[:, 0] >= lo) & (sc.wh[:, 0] < hi)


def scan_position(args):
    """All headings x pitch offsets at once: corr[de, da] = sum_i M[e_i + de, a_i + da] via FFT.
    With sc.extra['size_gate'] each detection size class is correlated with a raster of the buildings
    whose apparent width is within 2x of the class, and its expected count with that raster's coverage."""
    from scipy import fft
    pos_i, cam_xy, h = args
    sc = S_
    cam = np.array([cam_xy[0], cam_xy[1], float(sc.ground(*cam_xy)[0]) + h])
    idx, az, el, r = visible_directions(sc, cam)
    if len(idx) < 10:
        return []
    naz = int(round(360 / sc.bin_deg)); nel = int(round((EL_HI - EL_LO) / sc.bin_deg))
    wmax = int(round(sc.extra.get("pitch_window", 3.0) / sc.bin_deg))
    rows = np.r_[np.arange(0, wmax + 1), np.arange(nel - wmax, nel)]              # de in [-wmax, wmax]
    gate = sc.extra.get("size_gate", False)
    classes = [c for c in range(len(SIZE_CLASSES)) if _class_members(sc, c).any()] if gate else [None]
    out = []
    for f in sc.focals:
        rad = np.degrees(np.sqrt((sc.tol_px / f) ** 2 + (sc.map_sigma_m / r) ** 2))
        wpx = f * sc.bext[idx] / r
        base = (wpx > sc.extra.get("min_building_px", 3)) & (wpx < 400) & (el > EL_LO + 1) & (el < EL_HI - 1)
        FMs = []
        for c in classes:
            sel = base if c is None else base & (np.abs(np.log(wpx / math.sqrt(SIZE_CLASSES[c][0] * SIZE_CLASSES[c][1])))
                                                 <= math.log(2.0))
            M = np.zeros((nel, naz), np.uint8)
            for a, e, rr in zip(az[sel], el[sel], rad[sel]):
                ry = max(1, int(round(rr / sc.bin_deg))); rx = max(1, int(round(ry / max(math.cos(math.radians(e)), .2))))
                cx_, cy_ = int(round(a / sc.bin_deg)), int(round((e - EL_LO) / sc.bin_deg))
                for off in (0, naz, -naz):
                    if 0 <= cx_ + off + rx and cx_ + off - rx < naz:
                        cv2.ellipse(M, (cx_ + off, cy_), (rx, ry), 0, 0, 360, 1, -1)
            Mf = M.astype(np.float32)
            # local density: mean coverage in a window around each point (LOCAL_WIN_PX image pixels wide)
            k = max(3, int(round(np.degrees(LOCAL_WIN_PX / f) / sc.bin_deg)) | 1)
            Mb = cv2.blur(np.pad(Mf, ((0, 0), (k, k)), mode="wrap"), (k, k), borderType=cv2.BORDER_CONSTANT)[:, k:-k]
            FMs.append((fft.rfft2(Mf), fft.rfft2(Mb)))
        for p in sc.pitches:
            for ro in sc.rolls:
                hit = 0; exp = 0
                for c, (FM, FB) in zip(classes, FMs):
                    FD, nd = _det_fft(sc, f, p, ro, (nel, naz), c)
                    hit = hit + fft.irfft2(FM * FD, s=(nel, naz))[rows]
                    exp = exp + fft.irfft2(FB * FD, s=(nel, naz))[rows]
                z = (hit - exp) / np.sqrt(np.maximum(exp, 0) + 1)
                zc = z.max(0); kk = np.argsort(-zc)[:4]                     # best few headings per combo
                for ka in kk:
                    ke = int(np.argmax(z[:, ka]))
                    de = rows[ke] if rows[ke] <= wmax else rows[ke] - nel
                    out.append((float(z[ke, ka]), pos_i, float(cam[0]), float(cam[1]), float(h), float(ka * sc.bin_deg),
                                float(p + de * sc.bin_deg), float(ro), float(f), int(round(hit[ke, ka])), float(exp[ke, ka])))
    out.sort(key=lambda t: -t[0])

    def top_distinct(cands, n=3):
        keep = []
        for o in cands:                # distinct headings at this position
            if all(abs((o[5] - q[5] + 180) % 360 - 180) > 10 for q in keep):
                keep.append(o)
            if len(keep) >= n:
                break
        return keep
    keep = top_distinct(out)
    prior = sc.extra.get("heading_range")       # also keep the best inside a heading prior (flag 1)
    if prior:
        lo, hi = prior
        keep += [o for o in top_distinct([o for o in out if (o[5] - lo) % 360 <= (hi - lo) % 360]) if o not in keep]
    return keep


def scan(sc, workers=6, chunk=4):
    xs = np.arange(sc.box[0] + sc.step_m / 2, sc.box[2], sc.step_m)
    ys = np.arange(sc.box[1] + sc.step_m / 2, sc.box[3], sc.step_m)
    jobs = [(i, (x, y), h) for i, (x, y, h) in enumerate((x, y, h) for x in xs for y in ys for h in sc.heights)]
    log(f"scan: {len(xs) * len(ys)} positions x {len(sc.heights)} heights, {len(sc.focals)} focals x "
        f"{len(sc.pitches)} pitches x {len(sc.rolls)} rolls x {int(360 / sc.bin_deg)} headings; {len(sc.uv)} detections, "
        f"{len(sc.bxy)} map buildings")
    res = []; done = 0
    with Pool(workers, initializer=_init, initargs=(sc,)) as pool:
        for part in pool.imap_unordered(scan_position, jobs, chunksize=chunk):
            res += part; done += 1
            if done % max(1, len(jobs) // 20) == 0:
                best = max(res, key=lambda t: t[0]) if res else None
                log(f"  {done}/{len(jobs)} positions; best z so far {best[0]:.2f}" if best else f"  {done}/{len(jobs)}")
    keys = ["z", "pos", "e", "n", "h", "heading", "pitch", "roll", "focal", "hits", "expected"]
    return [dict(zip(keys, t)) for t in sorted(res, key=lambda t: -t[0])]


def in_range(h, rng):
    return rng is None or (h - rng[0]) % 360 <= (rng[1] - rng[0]) % 360


def distinct(peaks, sep_m=150, n=20, heading_range=None):
    out = []
    for d in peaks:
        if not in_range(d["heading"], heading_range):
            continue
        if all(math.dist((d["e"], d["n"]), (q["e"], q["n"])) > sep_m or abs((d["heading"] - q["heading"] + 180) % 360 - 180) > 20
               for q in out):
            out.append(d)
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ verify (full camera model + significance S)
class Verifier:
    SCALE = 4
    DISC_K = 2.0              # disc radius in sigmas (2 sigma covers 86% of a 2-D Gaussian)
    MAP_SIGMA_M = 3.0         # map-side position error (m)

    def __init__(self, sc):
        self.sc = sc
        if not sc.extra.get("size_gate", False):
            self.SIZE_RATIO = 1e9                    # detections are clusters, not single houses: no size test
        med = np.median(sc.wh, axis=0)
        self.sig_obs = float(np.maximum(4, med * [.16, .30]).mean())
        self.rw, self.rh = sc.W // self.SCALE, sc.H // self.SCALE
        self.valid_small = (cv2.resize(sc.valid.astype(np.uint8), (self.rw, self.rh), interpolation=cv2.INTER_NEAREST) > 0
                            if sc.valid is not None else np.ones((self.rh, self.rw), bool))

    def camera(self, p):
        e, n, h, yaw, pitch, roll, logf = p
        return np.array([e, n, float(self.sc.ground(e, n)[0]) + h]), rotation(yaw, pitch, roll), math.exp(logf)

    def project(self, p):
        cam, R, f = self.camera(p)
        idx, az, el, r = visible_directions(self.sc, cam)
        P = np.c_[self.sc.bxy[idx], self.sc.bz[idx]] - cam
        v = P @ R.T; z = v[:, 2]; ok = z > 1
        uv = f * v[ok, :2] / z[ok, None] + [self.sc.cx, self.sc.cy]
        w = f * self.sc.bext[idx][ok] / r[ok]
        ins = ((uv[:, 0] >= 0) & (uv[:, 0] < self.sc.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.sc.H)
               & (w > self.sc.extra.get("min_building_px", 4)) & (w < 400))
        uv, w, ids, rr = uv[ins], w[ins], idx[ok][ins], r[ok][ins]
        if self.sc.valid is not None and len(uv):
            m = self.sc.valid[uv[:, 1].astype(int), uv[:, 0].astype(int)]
            uv, w, ids, rr = uv[m], w[m], ids[m], rr[m]
        sig_map = f * self.MAP_SIGMA_M / np.maximum(rr, 1)
        return uv, w, ids, self.DISC_K * np.sqrt(self.sig_obs ** 2 + sig_map ** 2)

    SIZE_RATIO = 2.0          # a detection may only match a building whose projected width is within 2x

    def score(self, p, detail=False):
        """Significance of the one-to-one, size-compatible matches. Each detection i has its own chance
        probability q_i = image fraction covered by discs of size-compatible buildings; the number of
        matches is Poisson-binomial under the null; S = -log10 P(K >= k)."""
        uv, w, ids, rad = self.project(p)
        det_w = self.sc.wh[:, 0]; n = len(det_w)
        if len(uv) == 0:
            return (0., 0, 0, 0., []) if detail else 0.
        lr = np.abs(np.log(det_w[:, None] / w[None]))
        D = np.linalg.norm(self.sc.uv[:, None] - uv[None], axis=2)
        ok = (D <= rad[None]) & (lr <= math.log(self.SIZE_RATIO))
        C = np.where(ok, D, 1e6); ri, ci = linear_sum_assignment(C)
        good = C[ri, ci] < 1e6; k = int(good.sum())
        # per-detection coverage, via detection size classes (factor 1.25 bins)
        cls = np.round(np.log(det_w) / math.log(1.25)).astype(int)
        q = np.zeros(n)
        win = max(3, int(LOCAL_WIN_PX / self.SCALE) | 1)
        vs = self.valid_small.astype(np.float32); vb = cv2.blur(vs, (win, win))
        di = np.clip((self.sc.uv / self.SCALE).astype(int), 0, [self.rw - 1, self.rh - 1])
        for c in np.unique(cls):
            wc = 1.25 ** c
            sel = np.abs(np.log(wc / w)) <= math.log(self.SIZE_RATIO)
            m = np.zeros((self.rh, self.rw), np.uint8)
            for (u, v), rr in zip(uv[sel], rad[sel]):
                cv2.circle(m, (int(u / self.SCALE), int(v / self.SCALE)), max(1, int(rr / self.SCALE)), 1, -1)
            local = cv2.blur(m.astype(np.float32) * vs, (win, win)) / np.maximum(vb, 1e-6)   # coverage near each point
            idx = np.where(cls == c)[0]
            q[idx] = local[di[idx, 1], di[idx, 0]]
        q = np.clip(q, 1e-6, 1 - 1e-9)
        S = poisson_binomial_sf(k, q) if k > 0 else 0.
        if detail:
            return S, k, len(uv), float(q.mean()), [(int(a), int(ids[b])) for a, b in zip(ri[good], ci[good])]
        return S

    def refine(self, p0, radius=(30, 30, None, 3, 2, 2, .15)):
        p0 = np.array(p0, float)
        rad = np.array([radius[0], radius[1], radius[2] if radius[2] is not None else max(1.0, .25 * p0[2]),
                        *radius[3:]], float)
        if self.sc.extra.get("focal_fixed"):
            rad[6] = 0.0
        if self.sc.extra.get("handheld"):          # hand-held camera: height above terrain is known
            rad[2] = 0.0
        lo, hi = p0 - rad, p0 + rad; lo[2] = max(lo[2], 1.0)

        def obj(x):
            return -self.score(np.clip(x, lo, hi))
        best = p0; fbest = obj(p0)
        for scale in (1.0, .3):                     # coarse-to-fine random restarts around the peak
            rng = np.random.default_rng(int(scale * 100))
            starts = [best] + [np.clip(best + rng.normal(0, 1, 7) * rad * scale, lo, hi) for _ in range(3)]
            for s in starts:
                r = minimize(obj, s, method="Powell", options=dict(xtol=1e-3, ftol=1e-3, maxfev=400))
                if r.fun < fbest:
                    best, fbest = np.clip(r.x, lo, hi), r.fun
        return best, -fbest


def poisson_binomial_sf(k, q):
    """-log10 P(K >= k) for K = sum of independent Bernoulli(q_i) (exact DP, log-safe enough for n < 300)."""
    pmf = np.zeros(len(q) + 1); pmf[0] = 1.0
    for qi in q:
        pmf[1:] = pmf[1:] * (1 - qi) + pmf[:-1] * qi; pmf[0] *= (1 - qi)
    tail = float(pmf[k:].sum())
    return float(-math.log10(max(tail, 1e-300)))


def peak_params(d):
    return [d["e"], d["n"], d["h"], d["heading"], d["pitch"], d["roll"], math.log(d["focal"])]


def centre_ground_point(sc, p, max_r=6000):
    """Where the image-centre ray meets the terrain (for comparing with a centre-point truth)."""
    cam = np.array([p[0], p[1], float(sc.ground(p[0], p[1])[0]) + p[2]])
    fwd = rotation(p[3], p[4], p[5])[2]
    for t in np.arange(5, max_r, 5):
        q = cam + t * fwd
        if q[2] <= float(sc.ground(q[0], q[1])[0]):
            return q[:2]
    return None


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
