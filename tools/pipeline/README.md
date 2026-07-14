# Reconstruction pipeline

These scripts cover automatic annotation, frame preparation, VGGT/Omega scene
reconstruction, calibration, rendering, and evaluation. Run them from the
repository root so relative data paths resolve consistently.

The primary orchestration entry points are:

- `auto_annotate_batch.py` for batch annotation generation
- `flight_path_pipeline.py` for reusable flight-path reconstruction
- `reconstruct_scenes.py` and `run_vggt_batch_from_annotations.py` for scenes
- `auto_scale_hypotheses.py` for metric-scale hypotheses

## Resumable Omega Runs

Use the wrapper for a long Omega batch. It starts the local tool server with
the same output root as the batch, checkpoints one result row per resolved
scene, and clears only Omega's transient `demo_outputs/input_images_*` upload
directories between sequential jobs. It never removes scene outputs, job
records, checkpoints, or model weights.

```bash
python tools/pipeline/reconstruct_scenes.py --preset clean --skip-existing --stop-pod \
  2026-05-26_anti_drone_platform_biranit
```

Pass `--out-dir /path/to/output-root` when the local server and generated
scenes are intentionally outside this checkout. The command stores its ledger
at `<out-dir>/scenes/.batch_<preset>.json`; rerunning it updates rows by scene
id, so successful retries replace old error rows instead of accumulating stale
totals. Use `--reset-results` only with `run_vggt_batch_from_annotations.py`
when deliberately starting a new ledger.
