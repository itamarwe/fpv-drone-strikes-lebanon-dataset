# Reconstruction pipeline

These scripts cover automatic annotation, frame preparation, VGGT/Omega scene
reconstruction, calibration, rendering, and evaluation. Run them from the
repository root so relative data paths resolve consistently.

The primary orchestration entry points are:

- `auto_annotate_batch.py` for batch annotation generation
- `flight_path_pipeline.py` for reusable flight-path reconstruction
- `reconstruct_scenes.py` and `run_vggt_batch_from_annotations.py` for scenes
- `auto_scale_hypotheses.py` for metric-scale hypotheses
