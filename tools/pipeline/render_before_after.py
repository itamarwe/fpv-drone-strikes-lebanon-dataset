#!/usr/bin/env python3
"""Offline before/after renders of reconstruction point clouds for sharing.

Software point-splat renderer (numpy z-buffer) with a perspective camera. Each
panel: one point cloud (RGB or tinted), optional polylines (camera paths), label
and caption. Panels are composed side by side with a divider.
"""
from __future__ import annotations

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def quat_to_R(q):  # three.js order (x, y, z, w)
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)  # down
    R = np.stack([r, d, f])  # world -> camera rows
    return R, eye


def project(P, R, eye, fpx, W, H):
    C = (R @ (P - eye).T).T
    z = C[:, 2]; ok = z > 1e-6
    u = W / 2 + fpx * C[:, 0] / np.where(ok, z, 1); v = H / 2 + fpx * C[:, 1] / np.where(ok, z, 1)
    return u, v, z, ok


def splat(img, zbuf, P, col, R, eye, fpx, size=2):
    W, H = img.shape[1], img.shape[0]
    u, v, z, ok = project(P, R, eye, fpx, W, H)
    ok &= (u >= 0) & (u < W - size) & (v >= 0) & (v < H - size)
    u, v, z, col = u[ok].astype(int), v[ok].astype(int), z[ok], col[ok]
    order = np.argsort(-z)  # far first
    u, v, z, col = u[order], v[order], z[order], col[order]
    for dy in range(size):
        for dx in range(size):
            uu, vv = u + dx, v + dy
            better = z < zbuf[vv, uu]
            zbuf[vv[better], uu[better]] = z[better]
            img[vv[better], uu[better]] = col[better]


def draw_polyline(im, P, R, eye, fpx, colour, width=3):
    W, H = im.size
    u, v, z, ok = project(np.asarray(P, float), R, eye, fpx, W, H)
    d = ImageDraw.Draw(im); pts = [(float(a), float(b)) for a, b, o in zip(u, v, ok) if o]
    if len(pts) > 1: d.line(pts, fill=colour, width=width, joint="curve")


def render_panel(W, H, layers, lines, R, eye, fpx, bg=(12, 13, 15), size=2):
    img = np.full((H, W, 3), bg, np.uint8); zbuf = np.full((H, W), np.inf)
    for P, col in layers: splat(img, zbuf, P, col, R, eye, fpx, size)
    im = Image.fromarray(img)
    for P, colour, width in lines: draw_polyline(im, P, R, eye, fpx, colour, width)
    return im


def font(sz):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf", "/Library/Fonts/Arial.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()


def compose(panels, labels, captions, title, out, legend=None):
    W, H = panels[0].size; gap = 12; header = 110; footer = 150
    canvas = Image.new("RGB", (W * len(panels) + gap * (len(panels) - 1), H + header + footer), (12, 13, 15))
    d = ImageDraw.Draw(canvas)
    d.text((28, 22), title, fill=(235, 238, 242), font=font(40))
    for i, (p, lab, cap) in enumerate(zip(panels, labels, captions)):
        x = i * (W + gap); canvas.paste(p, (x, header))
        d.rectangle([x, header, x + W - 1, header + 64], fill=(12, 13, 15, 0))
        tag = "BEFORE" if i == 0 else "AFTER"; col = (255, 77, 109) if i == 0 else (54, 228, 255)
        d.rounded_rectangle([x + 20, header + 18, x + 20 + 190, header + 18 + 56], radius=10, fill=col)
        d.text((x + 44, header + 24), tag, fill=(12, 13, 15), font=font(38))
        d.text((x + 230, header + 30), lab, fill=(235, 238, 242), font=font(30))
        y = header + H + 18
        for line in cap.split("\n"):
            d.text((x + 24, y), line, fill=(200, 208, 216), font=font(26)); y += 34
    if legend:
        d.text((28, 66), legend, fill=(139, 149, 161), font=font(24))
    canvas.save(out, quality=95); print("wrote", out, canvas.size)
    return canvas
