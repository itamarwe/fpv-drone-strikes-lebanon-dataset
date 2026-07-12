# Scene viewer assets

## FPV kamikaze drone (optional GLB)

- Source: [FPV kamikaze drone Low-poly by Kvasovich on Sketchfab](https://sketchfab.com/3d-models/fpv-kamikaze-drone-low-poly-87f5bbb5b08641b782bc084ddd4082a7)
- License: [CC Attribution](https://creativecommons.org/licenses/by/4.0/)
- File: `fpv_kamikaze_drone.glb` (not committed by default)

Download into this folder:

```bash
export SKETCHFAB_API_TOKEN=your_token
python tools/pipeline/download_fpv_drone_model.py
```

If the GLB is missing, the viewer falls back to a built-in procedural ~15 inch FPV model.
