#!/usr/bin/env python3
"""Local (Mac CPU) lens self-calibration: the same COLMAP -> GLOMAP pipeline as starred_calibrate_remote.sh.

  COLMAP 3.11.1 SIFT (single OPENCV_FISHEYE camera, CPU) -> exhaustive matching -> GLOMAP global mapper
  (all intrinsics free) -> TXT model in scenes/<vid>/calib/glomap_fisheye/{cameras,images,analyzer}.txt
  plus scenes/<vid>/calib/status.json in the pod's format.

Toolchain (micromamba, no conda needed), TWO envs because conda-forge osx-arm64 has glomap 1.2 only (libboost 1.88) and
colmap 3.11.1 needs libboost 1.86:
  export MAMBA_ROOT_PREFIX=<root>
  micromamba create -n colmap311 -c conda-forge --override-channels "colmap=3.11.1=*cpu*"   # features + matching (same as the pod)
  micromamba create -n sfm4 -c conda-forge --override-channels "colmap=4.*=cpu*" "glomap=1.2.*" libfaiss  # glomap + model_converter
glomap 1.2 cannot open a colmap 3.11 database as-is (needs rig/frame tables, new images/pose_priors schema), so
to_glomap12_db() rewrites the 3.11 database in place. (colmap 4.0.4 matching also works but its FLANN segfaults and brute force
is ~10x slower, ~39 min vs ~4 min for 125 frames; it is not used.) Defaults below point at the scratchpad root;
override with CALIB_MAMBA_ROOT / CALIB_MICROMAMBA.

usage: calibrate_local.py <scene_dir_or_video_id> [...]  [--images-subdir published/images] [--threads 8]
       [--force] [--work DIR] [--out-subdir calib]   (scene args may be bare ids under scenes/all_undistort/)
A scene already holding calib/glomap_fisheye/cameras.txt + a succeeded status.json is skipped unless --force.
--out-subdir lets you write somewhere else (e.g. a parity test) without touching the real calib/ dir."""
import argparse, datetime, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/private/tmp/claude-501/-Users-itamarwe-Documents-code-fpv-drone-strikes-lebanon-dataset/ef4551ac-c779-4415-9942-6c4ba4df500e/scratchpad")
MAMBA_ROOT = os.environ.get("CALIB_MAMBA_ROOT", str(SCRATCH / "mamba"))
MICROMAMBA = os.environ.get("CALIB_MICROMAMBA", str(SCRATCH / "mm/bin/micromamba"))


def log(msg: str) -> None:
    print(f"[calib {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def tool(env: str, name: str, from_path: bool) -> list[str]:
    if from_path:
        return [name]
    return [MICROMAMBA, "run", "-n", env, name]


def run(cmd: list[str], logf, env_extra=None) -> int:
    env = {**os.environ, "MAMBA_ROOT_PREFIX": MAMBA_ROOT, "QT_QPA_PLATFORM": "offscreen", **(env_extra or {})}
    logf.write(f"\n$ {' '.join(cmd)}\n"); logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = time.time()
    for line in p.stdout:
        logf.write(line)
        if time.time() - last > 30:  # heartbeat so a long step is visibly alive
            log("  ... " + line.strip()[:110]); last = time.time()
    logf.flush()
    return p.wait()


def write_status(d: Path, status: str, reg: int, n: int, note: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"status": status, "registered": reg, "images": n, "note": note,
                                               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}))


def to_glomap12_db(path: Path) -> None:
    """Rewrite a colmap 3.11 database into the schema glomap 1.2 reads (user_version 3900): rig/frame tables
    (one rig per camera, one frame per image), pose_priors keyed by image_id, images without the prior_q*/prior_t* columns."""
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS rigs (rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, ref_sensor_id INTEGER NOT NULL, ref_sensor_type INTEGER NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON rigs(ref_sensor_id, ref_sensor_type);
    CREATE TABLE IF NOT EXISTS rig_sensors (rig_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL, sensor_from_rig BLOB,
      FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
    CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON rig_sensors(sensor_id, sensor_type);
    CREATE TABLE IF NOT EXISTS frames (frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, rig_id INTEGER NOT NULL, FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
    CREATE TABLE IF NOT EXISTS frame_data (frame_id INTEGER NOT NULL, data_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL,
      FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
    CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON frame_data(data_id, sensor_type);
    DROP TABLE IF EXISTS pose_priors;
    CREATE TABLE pose_priors (image_id INTEGER PRIMARY KEY NOT NULL, position BLOB, coordinate_system INTEGER NOT NULL, position_covariance BLOB,
      FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
    """)
    if not c.execute("SELECT 1 FROM rigs").fetchone():
        for (cam,) in c.execute("SELECT camera_id FROM cameras").fetchall():
            c.execute("INSERT INTO rigs(ref_sensor_id, ref_sensor_type) VALUES (?, 0)", (cam,))
            rid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            for iid, in c.execute("SELECT image_id FROM images WHERE camera_id = ? ORDER BY image_id", (cam,)).fetchall():
                c.execute("INSERT INTO frames(rig_id) VALUES (?)", (rid,))
                fid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
                c.execute("INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, 0)", (fid, iid, cam))
    cols = [r[1] for r in c.execute("PRAGMA table_info(images)")]
    if cols != ["image_id", "name", "camera_id"]:
        c.executescript("""
        CREATE TABLE images_new (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
          CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647), FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
        INSERT INTO images_new SELECT image_id, name, camera_id FROM images;
        PRAGMA foreign_keys=OFF; DROP TABLE images; ALTER TABLE images_new RENAME TO images; CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);""")
    if "type" in [r[1] for r in c.execute("PRAGMA table_info(descriptors)")]:
        c.executescript("""
        CREATE TABLE descriptors_new (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
          FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
        INSERT INTO descriptors_new SELECT image_id, rows, cols, data FROM descriptors; DROP TABLE descriptors; ALTER TABLE descriptors_new RENAME TO descriptors;""")
    c.execute("PRAGMA user_version = 3900"); c.commit(); c.close()


def calibrate(scene: Path, a) -> bool:
    vid = scene.name
    img = scene / a.images_subdir
    final = scene / a.out_subdir
    if not img.is_dir():
        log(f"{vid}: no images dir {img}"); return False
    cam, st = final / "glomap_fisheye/cameras.txt", final / "status.json"
    if not a.force and cam.exists() and cam.stat().st_size and st.exists() and '"succeeded"' in st.read_text():
        log(f"{vid}: already calibrated"); return True
    n = len(list(img.glob("*.jpg")))
    work = Path(a.work) / vid
    shutil.rmtree(work, ignore_errors=True); out = work / "glomap_fisheye"; out.mkdir(parents=True)
    db = work / "db.db"; t0 = time.time()
    colmap = tool("colmap311", "colmap", a.bin_from_path); colmap4 = tool("sfm4", "colmap", a.bin_from_path); glomap = tool("sfm4", "glomap", a.bin_from_path)
    logf = open(work / "log.txt", "w")
    th = str(a.threads)

    def fail(note: str) -> bool:
        logf.close(); final.mkdir(parents=True, exist_ok=True); shutil.copy(work / "log.txt", final / "log.txt")
        write_status(final, "failed", 0, n, note); log(f"{vid}: FAILED - {note}"); return False

    log(f"{vid}: {n} images; feature extraction")
    write_status(final, "running", 0, n, "features")
    if run(colmap + ["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.single_camera", "1",
                     "--ImageReader.camera_model", "OPENCV_FISHEYE", "--SiftExtraction.use_gpu", "0",
                     "--SiftExtraction.max_num_features", "16384", "--SiftExtraction.num_threads", th], logf):
        return fail("feature_extractor failed")
    # COLMAP >= 3.10 fisheye pairs need a focal prior on the camera row; drop images with (almost) no keypoints.
    c = sqlite3.connect(db); c.execute("UPDATE cameras SET prior_focal_length=1")
    weak = [r[0] for r in c.execute("SELECT i.image_id FROM images i LEFT JOIN keypoints k ON k.image_id = i.image_id WHERE k.rows IS NULL OR k.rows < 64")]
    for iid in weak:
        for tbl in ("keypoints", "descriptors", "images"):
            c.execute(f"DELETE FROM {tbl} WHERE image_id = ?", (iid,))
    c.commit(); c.close(); log(f"{vid}: dropped {len(weak)} images with < 64 keypoints; matching")
    write_status(final, "running", 0, n, "matching")
    if run(colmap + ["exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", th], logf):
        return fail("exhaustive_matcher failed (the pod falls back to pycolmap+transplant on a FLANN segfault; not implemented locally)")
    log(f"{vid}: matching done in {time.time() - t0:.0f}s; glomap")
    write_status(final, "running", 0, n, "glomap")
    to_glomap12_db(db)
    sparse = work / "glomap_sparse"; sparse.mkdir()
    if run(glomap + ["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sparse),
                     "--GlobalPositioning.use_gpu", "0", "--BundleAdjustment.use_gpu", "0"], logf):
        return fail("glomap mapper failed")
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        return fail("glomap produced no model")
    model = models[0]
    # GLOMAP 1.2 writes a COLMAP-4-era binary model; convert with the 3.11 tool as the pod does.
    run(colmap4 + ["model_converter", "--input_path", str(model), "--output_path", str(out), "--output_type", "TXT"], logf)
    with open(out / "analyzer.txt", "w") as af:
        r = subprocess.run(colmap4 + ["model_analyzer", "--path", str(model)], stdout=af, stderr=subprocess.STDOUT, text=True,
                           env={**os.environ, "MAMBA_ROOT_PREFIX": MAMBA_ROOT, "QT_QPA_PLATFORM": "offscreen"})
    if not (out / "cameras.txt").exists() or not (out / "images.txt").exists():
        return fail("model_converter produced no TXT model")
    reg = sum(1 for l in (out / "images.txt").read_text().splitlines() if l and not l.startswith("#")) // 2
    dt = int(time.time() - t0)
    (final / "glomap_fisheye").mkdir(parents=True, exist_ok=True)
    for f in ("cameras.txt", "images.txt", "analyzer.txt"):
        shutil.copy(out / f, final / "glomap_fisheye" / f)
    logf.close(); shutil.copy(work / "log.txt", final / "log.txt")
    write_status(final, "succeeded", reg, n, f"glomap OPENCV_FISHEYE (local mac, colmap 3.11.1 features/matching + glomap 1.2), {dt}s")
    log(f"{vid}: registered {reg} / {n} images in {dt}s")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--images-subdir", default="published/images")
    ap.add_argument("--out-subdir", default="calib")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--work", default=str(SCRATCH / "calib_work"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bin-from-path", action="store_true")
    a = ap.parse_args()
    ok = True
    for s in a.scenes:
        p = Path(s)
        if not p.is_dir():
            p = ROOT / "scenes/all_undistort" / s
        ok &= calibrate(p.resolve(), a)
    log("all done")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
