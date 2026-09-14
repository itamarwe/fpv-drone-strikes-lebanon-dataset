#!/usr/bin/env python3
"""Copy keypoints, descriptors, matches and two-view geometries from a pycolmap (4.x) database into a
COLMAP 3.11 database with the same images (matched by name), so GLOMAP 1.1 can map faiss-computed matches.
usage: transplant_matches.py <src_pycolmap.db> <dst_colmap311.db>"""
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
s_ids = {name: iid for iid, name in src.execute("SELECT image_id, name FROM images")}
d_ids = {name: iid for iid, name in dst.execute("SELECT image_id, name FROM images")}
common = sorted(set(s_ids) & set(d_ids)); print("images src", len(s_ids), "dst", len(d_ids), "common", len(common))
def pair_id(a, b): return a * 2147483647 + b if a < b else b * 2147483647 + a
def unpair(p): return p // 2147483647, p % 2147483647
s2d = {s_ids[n]: d_ids[n] for n in common}
for tbl in ("keypoints", "descriptors", "matches", "two_view_geometries"):
    dst.execute(f"DELETE FROM {tbl}")
n_k = n_d = n_m = n_g = 0
for iid, rows, cols, data in src.execute("SELECT image_id, rows, cols, data FROM keypoints"):
    if iid in s2d: dst.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?,?,?,?)", (s2d[iid], rows, cols, data)); n_k += 1
for iid, rows, cols, data in src.execute("SELECT image_id, rows, cols, data FROM descriptors"):
    if iid in s2d: dst.execute("INSERT INTO descriptors(image_id, rows, cols, data) VALUES (?,?,?,?)", (s2d[iid], rows, cols, data)); n_d += 1
for pid, rows, cols, data in src.execute("SELECT pair_id, rows, cols, data FROM matches"):
    a, b = unpair(pid)
    if a in s2d and b in s2d:
        na, nb = s2d[a], s2d[b]
        if na > nb:  # keep pair ordering consistent with the blob's column order
            import numpy as np
            arr = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)[:, ::-1].copy() if rows else data
            data = arr.tobytes() if rows else data
        dst.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES (?,?,?,?)", (pair_id(na, nb), rows, cols, data)); n_m += 1
dcols = [r[1] for r in dst.execute("PRAGMA table_info(two_view_geometries)")]
scols = [r[1] for r in src.execute("PRAGMA table_info(two_view_geometries)")]
use = [c for c in dcols if c in scols]
print("two_view_geometries columns dst", dcols, "src", scols)
for row in src.execute(f"SELECT {', '.join(use)} FROM two_view_geometries"):
    rec = dict(zip(use, row)); a, b = unpair(rec["pair_id"])
    if a in s2d and b in s2d:
        na, nb = s2d[a], s2d[b]
        if na > nb and rec.get("rows"):
            import numpy as np
            rec["data"] = np.frombuffer(rec["data"], dtype=np.uint32).reshape(rec["rows"], rec["cols"])[:, ::-1].copy().tobytes()
        rec["pair_id"] = pair_id(na, nb)
        dst.execute(f"INSERT INTO two_view_geometries({', '.join(use)}) VALUES ({', '.join('?' * len(use))})", [rec[c] for c in use]); n_g += 1
dst.commit(); print("copied keypoints", n_k, "descriptors", n_d, "matches", n_m, "geometries", n_g)
