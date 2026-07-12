# Tools

| Directory | Purpose |
| --- | --- |
| `apps/` | Browser tools: annotator, scene browser, and scene viewer/measurement UI |
| `catalog/` | Normalize and audit catalog/annotation identity; verify public assets |
| `media/` | High-quality source replacement queue and media maintenance |
| `pipeline/` | Reusable annotation, reconstruction, calibration, and analysis pipeline |
| `publishing/` | Generate thumbnails/manifests and publish runtime data |
| `server/` | Local dependency-light API/static server for the browser tools |
| `e2e/` | Playwright end-to-end browser tests for complete user workflows |

Use the root-level npm scripts for catalog and publishing operations. Start the
browser tools with `python tools/server/fpv_tool_server.py`; old `/tools/*` app
URLs remain compatible through server aliases.

ECW decoding and terrain-viewer code belong in their own repository and should
not be added here.
