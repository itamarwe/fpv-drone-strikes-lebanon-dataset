# Synthetic flyover: methods vs Blender truth

| Run | Profile | Cameras | Scale m/unit | Path RMSE % | Cam err med m | Orient med deg | Local scale range | Surface med / p90 m | Within 3 m | Points | Depth ratio full / central | Claimed scale / true |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| VGGT-Omega direct (demo pipeline) | full | 120/120 | 239.71 | 2.0 | 5.6 | 3.1 | 0.78 – 1.18 | 2.1 / 6.7 | 67% | 2,000,000 | 1.38 / 1.08 | n/a |
| VGGT-Omega direct (demo pipeline) | pinhole | 120/120 | 204.08 | 0.2 | 0.5 | 1.7 | 0.97 – 1.01 | 1.2 / 4.2 | 85% | 2,000,000 | 1.00 / 1.00 | n/a |

Scale is the Sim(3) fit of predicted camera centres to the exact Blender poses. Path RMSE is the residual as a fraction of the 330 m path. Surface distance is nearest-neighbour from the aligned cloud to a 3M-point sample of the Blender surface after rigid refinement. Depth ratios are predicted depth (times the fitted scale) over true camera-z, VGGT-Omega direct runs only. Claimed scale is MoGe-3's automatic metres-per-unit against the true value.
