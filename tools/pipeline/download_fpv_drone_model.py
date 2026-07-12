#!/usr/bin/env python3
"""Download the CC-BY FPV kamikaze drone GLB from Sketchfab into the scene viewer assets."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

SKETCHFAB_UID = "87f5bbb5b08641b782bc084ddd4082a7"
SKETCHFAB_URL = f"https://sketchfab.com/3d-models/fpv-kamikaze-drone-low-poly-{SKETCHFAB_UID}"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "apps" / "scene-viewer" / "assets" / "fpv_kamikaze_drone.glb"


def fetch_download_url(token: str) -> str:
    request = Request(
        f"https://api.sketchfab.com/v3/models/{SKETCHFAB_UID}/download",
        headers={"Authorization": f"Token {token}"},
    )
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode()
    import json

    data = json.loads(payload)
    glb = data.get("glb") or {}
    url = glb.get("url")
    if not url:
        gltf = data.get("gltf") or {}
        url = gltf.get("url")
    if not url:
        raise RuntimeError(f"No downloadable archive in Sketchfab response: {payload[:300]}")
    return str(url)


def download_archive(url: str, dest: Path) -> None:
    request = Request(url)
    with urlopen(request, timeout=300) as response:
        data = response.read()
    dest.write_bytes(data)


def extract_glb(archive: Path, out_glb: Path) -> None:
    if archive.suffix.lower() == ".glb":
        shutil.copyfile(archive, out_glb)
        return
    with zipfile.ZipFile(archive) as zf:
        glb_names = [name for name in zf.namelist() if name.lower().endswith(".glb")]
        if glb_names:
            out_glb.write_bytes(zf.read(glb_names[0]))
            return
        gltf_names = [name for name in zf.namelist() if name.lower().endswith(".gltf")]
        if not gltf_names:
            raise RuntimeError("Archive has no .glb or .gltf file")
        raise RuntimeError(
            "Sketchfab returned a glTF zip without .glb. Re-run after converting scene.gltf to .glb, "
            "or download manually from Sketchfab and place it at "
            f"{out_glb}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN", ""))
    args = parser.parse_args()

    if not args.token:
        print(
            "Set SKETCHFAB_API_TOKEN to a Sketchfab API token, then re-run.\n"
            f"Model: {SKETCHFAB_URL}\n"
            "License: CC Attribution (Kvasovich / Sketchfab).",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".download")
    url = fetch_download_url(args.token.strip())
    download_archive(url, tmp)
    extract_glb(tmp, args.out)
    if tmp != args.out and tmp.exists():
        tmp.unlink()
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
