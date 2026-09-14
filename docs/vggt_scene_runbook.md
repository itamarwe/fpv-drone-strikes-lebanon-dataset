# VGGT Scene Reconstruction Runbook

This note is the reusable path for rebuilding VGGT-Omega FPV scenes, producing
local scene artifacts, and backing up the gated model assets.

## Secrets and Local State

Do not paste Hugging Face or AWS tokens into chat or commit them to the repo.

The VGGT-Omega checkpoint is gated. The local machine should have a read token
with access to `facebook/VGGT-Omega` in one of these places:

```bash
export HF_TOKEN=...
# or
mkdir -p ~/.cache/huggingface
chmod 700 ~/.cache/huggingface
printf '%s' "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

RunPod access uses `runpodctl`. AWS access is separate; for model-backup upload,
authenticate first:

```bash
aws login --profile admin
aws sts get-caller-identity --profile admin
```

## Zawtar 250-Frame / 10M Scene

The Zawtar scene path is:

```bash
SCENE_DIR="scenes/2026-05-26_humvee_zawtar_al_sharqiyah_riverbed_mmirleb_17068/2026-05-26_humvee_zawtar_al_sharqiyah_riverbed_mmirleb_17068_fullflight_250f_10m_8p1_36p9"
```

Prepare or refresh the exact 250 input frames from `8.1s` to `36.9s`:

```bash
python3 tools/reconstruct_zawtar_vggt_250_5m.py --skip-vggt --skip-artifacts
```

Create a fresh RunPod GPU pod when old pods are host-capacity blocked. H100 SXM
has been enough for the 250-frame run:

```bash
POD_ID="$(
  runpodctl pod create \
    --name fpv-vggt-zawtar-250f-10m-h100 \
    --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
    --gpu-id "NVIDIA H100 80GB HBM3" \
    --gpu-count 1 \
    --cloud-type SECURE \
    --container-disk-in-gb 80 \
    --volume-in-gb 120 \
    --volume-mount-path /workspace \
    --ports "7860/http,22/tcp" \
    -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)"
```

Install VGGT-Omega and forward the local HF token to the pod:

```bash
python3 tools/setup_runpod_vggt_omega_server.py "$POD_ID" --wait-url-s 0
```

Run the reconstruction directly on the pod. This bypasses the Gradio upload
queue/proxy layer and uses the same `gradio_demo()` code against files copied to
`/workspace`.

```bash
python3 tools/run_vggt_omega_direct_on_runpod.py \
  "$POD_ID" \
  "$SCENE_DIR" \
  --remote-name zawtar_250_10m \
  --max-points-k 10000 \
  --artifact-max-points 10000000 \
  --default-scale-m-per-unit 76.0 \
  --title "Zawtar VGGT 250 frames 10M full flight"
```

Stop the pod as soon as artifacts are copied back:

```bash
runpodctl pod stop "$POD_ID" -o json
```

Expected local outputs:

```text
$SCENE_DIR/vggt_scene.glb
$SCENE_DIR/vggt_remote_summary.json
$SCENE_DIR/point_cloud.npz
$SCENE_DIR/relative_path.npy
$SCENE_DIR/relative_path.csv
$SCENE_DIR/viewer/points_positions.bin
$SCENE_DIR/viewer/points_colors.bin
$SCENE_DIR/viewer/scene_meta.json
```

For the 2026-07-11 run, verification was:

```text
frames: 250
VGGT preprocessed shape: (250, 3, 368, 720)
GLB bytes: 160288688
point_count: 10000000
default_scale_m_per_unit: 76.0
```

## Generic Scene Pattern

For another scene, produce a scene directory with:

```text
frames/*.jpg
frames.csv
metadata.json
```

Then run:

```bash
python3 tools/run_vggt_omega_direct_on_runpod.py \
  "$POD_ID" \
  "$SCENE_DIR" \
  --remote-name "$(basename "$SCENE_DIR")" \
  --max-points-k 10000 \
  --artifact-max-points 10000000 \
  --default-scale-m-per-unit <scale_m_per_vggt_unit> \
  --title "<scene title>"
```

Use the measured or georegistered scale when known. If scale is unknown, keep it
explicit and temporary in `metadata.json`; do not mix temporary `save_button`
values with measured-scale benchmark cases.

## Private Backup of VGGT-Omega Assets

The reproducibility-critical gated assets are:

- `facebook/VGGT-Omega:vggt_omega_1b_512.pt`
- `facebook/vggt-omega` Space asset `skyseg.onnx`

Download them locally under ignored `model_backups/` and write checksums:

```bash
python3 tools/backup_vggt_omega_assets.py --skip-s3
```

Do not back these assets up to the public/CloudFront dataset bucket. Use a
separate private bucket with Block Public Access enabled.

Example first-time private bucket setup:

```bash
BUCKET="<private-model-backup-bucket>"
REGION="us-east-1"

aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --profile admin

aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
  --profile admin

aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' \
  --profile admin

aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled \
  --profile admin
```

Upload the backup after `aws login --profile admin` succeeds:

```bash
python3 tools/backup_vggt_omega_assets.py \
  --s3-uri "s3://$BUCKET/huggingface/vggt-omega/" \
  --aws-profile admin \
  --aws-region "$REGION"
```

The upload helper uses `aws s3 sync --sse AES256` and keeps a local
`manifest.json` containing file sizes and SHA-256 hashes.
