"""Public asset URL helpers for local dev vs CloudFront-hosted scene data."""

from __future__ import annotations

import os

DEFAULT_CLOUDFRONT_ROOT = "https://d2fioemadmrru3.cloudfront.net"


def asset_root_from_env() -> str:
    return os.environ.get("FPV_ASSET_ROOT", "").strip().rstrip("/")


def join_public_url(root: str, path: str) -> str:
    clean = path.lstrip("/")
    if not root:
        return f"/{clean}"
    return f"{root.rstrip('/')}/{clean}"


def with_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def scene_data_base(rel_path: str, asset_root: str = "") -> str:
    return with_trailing_slash(join_public_url(asset_root, f"scenes/{rel_path}/viewer"))


def scene_viewer_page_url(rel_path: str, app_base: str = "") -> str:
    return join_public_url(app_base, f"scenes/{rel_path}/viewer/index.html")
