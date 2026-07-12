# Tools

| Directory | Purpose |
| --- | --- |
| `catalog/` | Normalize and audit catalog/annotation identity; verify public assets |
| `pipeline/` | Reusable annotation, reconstruction, calibration, and analysis pipeline |
| `scene_viewer/` | Browser-based 3D scene viewer and measurement tool |
| `e2e/` | Playwright end-to-end browser tests for complete user workflows |

The annotator entry point and local server remain at `annotator.html` and
`fpv_tool_server.py`. Publishing entry points remain at `publish_web.sh` and the
root-level `.mjs` scripts so existing operational commands keep working.

ECW decoding and terrain-viewer code belong in their own repository and should
not be added here.
