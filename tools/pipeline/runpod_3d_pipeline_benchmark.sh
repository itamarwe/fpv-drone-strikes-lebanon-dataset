#!/usr/bin/env bash
# Run inside a RunPod after extracting the prepared benchmark bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${FPV_3D_WORKSPACE:-/workspace/fpv-3d-benchmark}"
REPOS_ROOT="$WORKSPACE_ROOT/repos"
RESULTS_ROOT="$WORKSPACE_ROOT/results"
LOGS_ROOT="$WORKSPACE_ROOT/logs"
SPEC="$BUNDLE_ROOT/benchmark.json"
CONDA_ROOT="${FPV_3D_CONDA_ROOT:-/workspace/miniforge3}"

# Keep model, wheel, torch and conda package caches on the pod workspace disk.
# HF_HOME stays unchanged so the securely installed default HF token is found.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORKSPACE_ROOT/cache/huggingface/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_ROOT/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$WORKSPACE_ROOT/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORKSPACE_ROOT/cache/torch-extensions}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKSPACE_ROOT/cache/conda-pkgs/base}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

SCAL3R_REV="dc557e7be5ad821ed44b8ad37700311136061406"
MOGE_REV="74fbce054ebed49800de42d0ad0e83495065719a"
GLOMAP_REV="99806d0869f802fad218516a2e027793e7ca687d"
SURFLO_REV="bf14c6375a92911c45795710cd00bf2af17e9a13"
QUERYSPLAT_REV="3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae"
MINIFORGE_VERSION="24.11.3-2"
BUILD_JOBS="${FPV_BUILD_JOBS:-6}"
PREPROCESS_WORKERS="${FPV_PREPROCESS_WORKERS:-6}"
export MAX_JOBS="${MAX_JOBS:-$BUILD_JOBS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

log() { printf '[fpv-3d] %s\n' "$*"; }
die() { printf '[fpv-3d] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  runpod_3d_pipeline_benchmark.sh preflight
  runpod_3d_pipeline_benchmark.sh setup-base
  runpod_3d_pipeline_benchmark.sh checkpoint-preflight
  runpod_3d_pipeline_benchmark.sh setup METHOD
  runpod_3d_pipeline_benchmark.sh download METHOD
  runpod_3d_pipeline_benchmark.sh run METHOD SCENE PROFILE

METHOD: scal3r_zju | moge3_glomap | surflo | querysplat | ssmb_lightglue
SCENE: scene_a | scene_b
PROFILE: common16 | dense_train | full

QuerySplat's feed-forward run exports the matched-input VGGT-Omega camera,
depth and point-cloud baseline. SSMB remains blocked pending its public release.
Commands that install, download, or run methods require
runpod.paid_execution_allowed=true in benchmark.json.
EOF
}

require_execution_allowed() {
  [[ -f "$SPEC" ]] || die "missing benchmark spec: $SPEC"
  python3 - "$SPEC" <<'PY'
import json
import sys

spec_path = sys.argv[1]
with open(spec_path) as handle:
    spec = json.load(handle)
runpod = spec.get("runpod", {})
if runpod.get("paid_execution_allowed") is True:
    raise SystemExit(0)
note = runpod.get("note") or "Execution is disabled by benchmark.json."
print(f"[fpv-3d] ERROR: execution policy gate is closed: {note}", file=sys.stderr)
raise SystemExit(1)
PY
}

install_conda() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
    die "automatic Miniforge bootstrap supports Linux x86_64 only"
  local filename="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  local url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${filename}"
  local temp_dir
  temp_dir="$(mktemp -d)"
  log "Installing pinned Miniforge ${MINIFORGE_VERSION} in $CONDA_ROOT"
  curl -fsSL --retry 4 "$url" -o "$temp_dir/$filename"
  curl -fsSL --retry 4 "$url.sha256" -o "$temp_dir/$filename.sha256"
  (cd "$temp_dir" && sha256sum --check "$filename.sha256")
  bash "$temp_dir/$filename" -b -p "$CONDA_ROOT"
  "$CONDA_ROOT/bin/conda" config --system --set auto_activate_base false
  rm -f "$temp_dir/$filename" "$temp_dir/$filename.sha256"
  rmdir "$temp_dir"
}

ensure_conda() {
  if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    export PATH="$CONDA_ROOT/bin:$PATH"
  elif ! command -v conda >/dev/null 2>&1; then
    install_conda
    export PATH="$CONDA_ROOT/bin:$PATH"
  fi
  command -v conda >/dev/null 2>&1 || die "conda bootstrap failed"
}

clone_pinned() {
  local name="$1" url="$2" revision="$3" destination
  destination="$REPOS_ROOT/$name"
  mkdir -p "$REPOS_ROOT"
  [[ -d "$destination/.git" ]] || git clone --filter=blob:none --recursive "$url" "$destination"
  git -C "$destination" fetch --tags origin "$revision"
  git -C "$destination" checkout --detach "$revision"
  git -C "$destination" submodule update --init --recursive
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || \
    die "$name checkout is not at pinned revision $revision"
}

verify_sha256() {
  local expected="$1" file="$2"
  [[ -f "$file" ]] || die "missing checkpoint: $file"
  printf '%s  %s\n' "$expected" "$file" | sha256sum --check --status || \
    die "checksum mismatch: $file"
}

verify_size() {
  local expected="$1" file="$2" actual
  actual="$(stat -c %s "$file")"
  [[ "$actual" == "$expected" ]] || die "size mismatch: $file ($actual != $expected)"
}

scene_profile_dir() {
  local scene="$1" profile="$2" destination
  destination="$BUNDLE_ROOT/scenes/$scene/profiles/$profile/images"
  [[ -d "$destination" ]] || die "missing prepared images: $destination"
  printf '%s\n' "$destination"
}

profile_image_count() {
  local directory="$1" count
  count="$(find -L "$directory" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
  [[ "$count" -gt 0 ]] || die "no images found in $directory"
  printf '%s\n' "$count"
}

glomap_bin() {
  local candidate
  for candidate in "$REPOS_ROOT/glomap/build/glomap/glomap" "$REPOS_ROOT/glomap/build/bin/glomap"; do
    [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  done
  command -v glomap >/dev/null 2>&1 && { command -v glomap; return; }
  die "could not locate the GLOMAP executable"
}

activate_querysplat_cuda() {
  ensure_conda
  local env_prefix cuda_target
  env_prefix="$(conda run -n querysplat python -c 'import sys; print(sys.prefix)')"
  cuda_target="$env_prefix/targets/x86_64-linux"
  [[ -x "$env_prefix/bin/nvcc" ]] || die "QuerySplat nvcc missing: $env_prefix/bin/nvcc"
  [[ -f "$cuda_target/include/cuda_runtime.h" ]] || \
    die "QuerySplat CUDA headers missing: $cuda_target/include/cuda_runtime.h"
  export CUDA_HOME="$env_prefix"
  export CPATH="$cuda_target/include${CPATH:+:$CPATH}"
  export C_INCLUDE_PATH="$cuda_target/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$cuda_target/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export LIBRARY_PATH="$cuda_target/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$cuda_target/lib:$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

preflight() {
  [[ -f "$SPEC" ]] || die "missing benchmark spec: $SPEC"
  python3 "$SCRIPT_DIR/check_3d_pipeline_benchmark.py" --bundle "$BUNDLE_ROOT"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  mkdir -p "$WORKSPACE_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT"
  df -h "$WORKSPACE_ROOT"
  if command -v hf >/dev/null 2>&1; then
    hf auth whoami >/dev/null 2>&1 && log "Hugging Face authentication is available" || \
      log "Hugging Face authentication is absent; VGGT-Omega download will fail"
  else
    log "hf CLI is not installed yet; setup-base installs it"
  fi
}

checkpoint_preflight() {
  command -v hf >/dev/null 2>&1 || die "hf CLI is required; run setup-base first"
  hf download xbillowy/Scal3R scal3r.pt --dry-run --format json >/dev/null
  hf download Ruicheng/moge-3-vitl model.pt --dry-run --format json >/dev/null
  hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --dry-run --format json >/dev/null
  hf download inspatio/querysplat querysplat_vggto_1B_512_8192.safetensors --dry-run --format json >/dev/null
  hf auth whoami >/dev/null 2>&1 || die "authenticated Hugging Face access is required for VGGT-Omega"
  hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --dry-run --format json >/dev/null
  curl -fLIsS --max-time 30 https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt >/dev/null
  log "All released checkpoint endpoints are accessible"
}

setup_base() {
  if [[ "${FPV_3D_SKIP_BASE:-0}" == 1 ]]; then
    ensure_conda
    command -v cmake >/dev/null 2>&1 || die "cmake missing while FPV_3D_SKIP_BASE=1"
    command -v colmap >/dev/null 2>&1 || die "colmap missing while FPV_3D_SKIP_BASE=1"
    command -v hf >/dev/null 2>&1 || die "hf missing while FPV_3D_SKIP_BASE=1"
    log "Reusing completed base setup"
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache curl git jq ninja-build pkg-config wget \
    libboost-graph-dev libboost-program-options-dev libboost-system-dev libcgal-dev \
    libceres-dev libeigen3-dev libflann-dev libfreeimage-dev libglew-dev \
    libgoogle-glog-dev libgtest-dev libmetis-dev libsqlite3-dev libsuitesparse-dev \
    libcurl4-openssl-dev libssl-dev liblz4-dev libjpeg-dev libpng-dev libtiff-dev \
    libegl1-mesa-dev libgl1-mesa-dev qtbase5-dev libqt5opengl5-dev colmap
  ensure_conda
  conda install -n base -y "cmake>=3.28,<3.31" ninja pip
  conda run -n base python -m pip install --upgrade "huggingface_hub>=0.33,<2"
  hash -r
  cmake --version | head -n 1
  hf --version
}

setup_scal3r() {
  export CONDA_PKGS_DIRS="$WORKSPACE_ROOT/cache/conda-pkgs/scal3r"
  setup_base
  clone_pinned Scal3R https://github.com/zju3dv/Scal3R.git "$SCAL3R_REV"
  (cd "$REPOS_ROOT/Scal3R" && CONDA_ENV=scal3r TORCH_CHANNEL=cu128 USE_UV=0 bash scripts/install.sh)
  # OpenCV 5 wheels currently lack the EXR writer used by Scal3R's depth/point
  # exporter. Keep one verified OpenCV provider and a NumPy version accepted by
  # both OpenCV 4.12 and Numba.
  conda run -n scal3r python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
  conda run -n scal3r python -m pip install "numpy>=2,<2.3" "opencv-python==4.12.0.88"
  conda run -n scal3r python -m scal3r.run --help >/dev/null
  local exr_test="$WORKSPACE_ROOT/scal3r-exr-writer-test.exr"
  conda run -n scal3r python - "$exr_test" <<'PY'
import cv2, numpy as np, os, sys
path = sys.argv[1]
if not cv2.imwrite(path, np.zeros((2, 2), dtype=np.float32)):
    raise SystemExit("OpenCV EXR writer smoke test returned false")
roundtrip = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if roundtrip is None or roundtrip.shape != (2, 2):
    raise SystemExit("OpenCV EXR writer smoke test could not read its output")
os.unlink(path)
print(f"OpenCV {cv2.__version__} EXR write/read smoke passed")
PY
}

download_scal3r() {
  local checkpoint_dir="$REPOS_ROOT/Scal3R/data/checkpoints"
  [[ -d "$REPOS_ROOT/Scal3R" ]] || die "run setup scal3r_zju first"
  mkdir -p "$checkpoint_dir"
  hf download xbillowy/Scal3R scal3r.pt --repo-type model --local-dir "$checkpoint_dir"
  curl -fL --retry 4 --continue-at - https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt -o "$checkpoint_dir/dino_salad.ckpt"
  verify_sha256 7e5ff2b1c4a3d7ffb17a9c8c05478aa52118f9ae67f984fe0b221c1176bed21f "$checkpoint_dir/scal3r.pt"
  # SALAD publishes no checksum. Enforce its release byte size and record its digest.
  verify_size 352040378 "$checkpoint_dir/dino_salad.ckpt"
  sha256sum "$checkpoint_dir/scal3r.pt" "$checkpoint_dir/dino_salad.ckpt" > "$checkpoint_dir/SHA256SUMS.downloaded"
}

setup_moge3_glomap() {
  export CONDA_PKGS_DIRS="$WORKSPACE_ROOT/cache/conda-pkgs/moge3-glomap"
  setup_base
  clone_pinned MoGe https://github.com/microsoft/MoGe.git "$MOGE_REV"
  clone_pinned glomap https://github.com/colmap/glomap.git "$GLOMAP_REV"
  ensure_conda
  conda env list | awk '{print $1}' | grep -Fxq moge3 || conda create -n moge3 python=3.11 -y
  conda run -n moge3 python -m pip install --upgrade pip
  conda run -n moge3 python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  conda run -n moge3 python -m pip install -e "$REPOS_ROOT/MoGe"
  cmake -S "$REPOS_ROOT/glomap" -B "$REPOS_ROOT/glomap/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=OFF -DTESTS_ENABLED=OFF
  # RunPod may expose the host's CPU count even when cgroups grant far fewer
  # cores. A small explicit default avoids memory and scheduler contention.
  cmake --build "$REPOS_ROOT/glomap/build" --parallel "$BUILD_JOBS" --target glomap_main
  "$(glomap_bin)" --help >/dev/null
  colmap -h >/dev/null
}

download_moge3_glomap() {
  local checkpoint_dir="$REPOS_ROOT/MoGe/checkpoints/moge-3-vitl"
  [[ -d "$REPOS_ROOT/MoGe" ]] || die "run setup moge3_glomap first"
  mkdir -p "$checkpoint_dir"
  hf download Ruicheng/moge-3-vitl model.pt --local-dir "$checkpoint_dir"
  verify_sha256 9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925 "$checkpoint_dir/model.pt"
  sha256sum "$checkpoint_dir/model.pt" > "$checkpoint_dir/SHA256SUMS.downloaded"
}

setup_surflo() {
  export CONDA_PKGS_DIRS="$WORKSPACE_ROOT/cache/conda-pkgs/surflo"
  setup_base
  clone_pinned Surflo https://github.com/Anttwo/Surflo.git "$SURFLO_REV"
  ensure_conda
  conda env list | awk '{print $1}' | grep -Fxq surflo-cu124 || conda env create -f "$REPOS_ROOT/Surflo/install/environment-cu124.yml"
  (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 python -m pip install -e '.[demo,texture,train]')
  (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 bash install/activate_cuda.sh)
  (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 bash install/build_extensions.sh --all)
  (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 python install/verify_install.py --check-isolation)
}

download_surflo() {
  local checkpoint_dir="$REPOS_ROOT/Surflo/checkpoints"
  [[ -d "$REPOS_ROOT/Surflo" ]] || die "run setup surflo first"
  mkdir -p "$checkpoint_dir"
  hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --local-dir "$checkpoint_dir"
  verify_sha256 55fddb5dcbb5d375ba8036ec959acd215ee8ffd89e173cdabe1199e034052662 "$checkpoint_dir/surflo_v0.pt"
  sha256sum "$checkpoint_dir/surflo_v0.pt" > "$checkpoint_dir/SHA256SUMS.downloaded"
}

setup_querysplat() {
  export CONDA_PKGS_DIRS="$WORKSPACE_ROOT/cache/conda-pkgs/querysplat"
  setup_base
  clone_pinned QuerySplat https://github.com/inspatio/querysplat.git "$QUERYSPLAT_REV"
  ensure_conda
  conda env list | awk '{print $1}' | grep -Fxq querysplat || conda create -n querysplat python=3.12 -y
  conda install -n querysplat -y -c nvidia/label/cuda-12.8.0 \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-driver-dev=12.8 cuda-cccl=12.8
  activate_querysplat_cuda
  conda run -n querysplat python -m pip install --upgrade pip
  conda run -n querysplat python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  (cd "$REPOS_ROOT/QuerySplat" && conda run -n querysplat python -m pip install --no-build-isolation -r requirements.txt)
  conda run -n querysplat python -m pip install -U huggingface_hub
  (cd "$REPOS_ROOT/QuerySplat" && conda run -n querysplat python -m scripts.infer --help >/dev/null)
}

download_querysplat() {
  local checkpoint_dir="$REPOS_ROOT/QuerySplat/checkpoints"
  [[ -d "$REPOS_ROOT/QuerySplat" ]] || die "run setup querysplat first"
  mkdir -p "$checkpoint_dir"
  hf auth whoami >/dev/null 2>&1 || die "authenticated Hugging Face access is required for VGGT-Omega"
  hf download inspatio/querysplat querysplat_vggto_1B_512_8192.safetensors --local-dir "$checkpoint_dir"
  hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir "$checkpoint_dir"
  verify_sha256 ef817f4a7e2d6d73f0089767d848c242192f6aea439e2d9752e0929a36ff2f61 "$checkpoint_dir/querysplat_vggto_1B_512_8192.safetensors"
  verify_sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934 "$checkpoint_dir/vggt_omega_1b_512.pt"
  sha256sum "$checkpoint_dir/querysplat_vggto_1B_512_8192.safetensors" "$checkpoint_dir/vggt_omega_1b_512.pt" > "$checkpoint_dir/SHA256SUMS.downloaded"
}

run_scal3r() {
  local scene="$1" profile="$2" output_dir="$3" input_dir image_count block_size overlap_size
  input_dir="$(scene_profile_dir "$scene" "$profile")"
  image_count="$(profile_image_count "$input_dir")"
  block_size=60
  overlap_size=30
  if [[ "$image_count" -le "$block_size" ]]; then
    block_size="$image_count"
    overlap_size="$((image_count / 2))"
  fi
  [[ -f "$REPOS_ROOT/Scal3R/data/checkpoints/scal3r.pt" ]] || die "Scal3R checkpoint missing"
  (cd "$REPOS_ROOT/Scal3R" && conda run -n scal3r python -m scal3r.run \
    --input_dir "$input_dir" --output_dir "$output_dir" --checkpoint data/checkpoints/scal3r.pt \
    --block_size "$block_size" --overlap_size "$overlap_size" \
    --preprocess_workers "$PREPROCESS_WORKERS" --pgo_workers "$PREPROCESS_WORKERS" \
    --test_use_amp --save_dpt 1 --save_xyz 1 --streaming_state 1 \
    --offload_batches 1 --offload_outputs 1 --cleanup_offload 1)
}

run_moge3_glomap() {
  local scene="$1" profile="$2" output_dir="$3" input_dir database sparse_bin sparse_txt depth_dir scaled_dir model_dir
  input_dir="$(scene_profile_dir "$scene" "$profile")"
  database="$output_dir/database.db"
  sparse_bin="$output_dir/sparse-bin"
  sparse_txt="$output_dir/sparse-txt"
  depth_dir="${MOGE3_DEPTHS_DIR:-$output_dir/moge3-depths}"
  scaled_dir="$output_dir/sparse-metric-txt"
  mkdir -p "$sparse_bin" "$sparse_txt"
  colmap feature_extractor --database_path "$database" --image_path "$input_dir" \
    --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0 \
    --SiftExtraction.max_num_features 16384
  # Exhaustive matching is tractable at <=149 frames, gives a common pair graph,
  # and avoids the vocabulary tree required by sequential loop detection.
  colmap exhaustive_matcher --database_path "$database" --SiftMatching.use_gpu 0
  "$(glomap_bin)" mapper --database_path "$database" --image_path "$input_dir" --output_path "$sparse_bin"
  if [[ -f "$sparse_bin/images.bin" ]]; then
    model_dir="$sparse_bin"
  else
    model_dir="$(find "$sparse_bin" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/images.bin' \; -print | sort | head -n 1)"
  fi
  [[ -n "$model_dir" && -f "$model_dir/images.bin" ]] || die "GLOMAP did not emit a sparse model"
  colmap model_converter --input_path "$model_dir" --output_path "$sparse_txt" --output_type TXT
  if [[ -n "${MOGE3_DEPTHS_DIR:-}" ]]; then
    [[ -d "$depth_dir" ]] || die "MOGE3_DEPTHS_DIR is not a directory: $depth_dir"
    python3 - "$input_dir" "$depth_dir" <<'PY'
import json, sys
from pathlib import Path
images_dir, depths_dir = map(Path, sys.argv[1:])
image_names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
expected_npz = {f"{Path(name).stem}.npz" for name in image_names}
actual_npz = {p.name for p in depths_dir.glob("*.npz")}
missing = sorted(expected_npz - actual_npz)
extra = sorted(actual_npz - expected_npz)
if missing or extra:
    raise SystemExit(f"MoGe depth/input mismatch; missing={missing[:5]} extra={extra[:5]}")
manifest_path = depths_dir / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing MoGe depth manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
manifest_names = sorted(item["image"] for item in manifest.get("images", []))
if manifest_names != image_names:
    raise SystemExit("MoGe depth manifest image names do not match the reconstruction input")
print(f"validated {len(image_names)} reusable MoGe depth maps")
PY
    log "Reusing validated MoGe-3 depths: $depth_dir"
    python3 - "$output_dir/moge3_depth_source.json" "$depth_dir" <<'PY'
import json, sys
from pathlib import Path
destination, source = map(Path, sys.argv[1:])
destination.write_text(json.dumps({"schema_version": 1, "source": str(source.resolve()), "reused": True}, indent=2) + "\n")
PY
  else
    conda run -n moge3 python "$SCRIPT_DIR/moge3_export_depths.py" --images "$input_dir" --output "$depth_dir" \
      --checkpoint "$REPOS_ROOT/MoGe/checkpoints/moge-3-vitl/model.pt"
  fi
  # This scale is native to GLOMAP. Evaluation must use calibrated endpoint
  # distances, never compare this coefficient to VGGT's metres-per-unit value.
  conda run -n moge3 python "$SCRIPT_DIR/fit_moge3_colmap_scale.py" --model "$sparse_txt" --depths "$depth_dir" \
    --output-model "$scaled_dir" --summary "$output_dir/metric_scale.json"
}

run_surflo() {
  local scene="$1" profile="$2" output_dir="$3" input_dir image_count mode="${SURFLO_MODE:-both}"
  [[ "$profile" == common16 || "$profile" == dense_train || "$profile" == full || "$profile" == pinhole ]] || die "Surflo supports common16, dense_train and full profiles"
  [[ "$mode" == plain || "$mode" == guided || "$mode" == both ]] || die "SURFLO_MODE must be plain, guided, or both"
  input_dir="$(scene_profile_dir "$scene" "$profile")"
  image_count="$(profile_image_count "$input_dir")"
  if [[ "$mode" == plain || "$mode" == both ]]; then
    (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 python scripts/infer.py mode=plain plain=default \
      ckpt=checkpoints/surflo_v0.pt source.image_folder="$input_dir" source.n_images="$image_count" \
      num_query_points=100000 output_dir="$output_dir/plain")
  fi
  if [[ "$mode" == guided || "$mode" == both ]]; then
    (cd "$REPOS_ROOT/Surflo" && conda run -n surflo-cu124 python scripts/infer.py mode=guided guided=default mesh=default \
      ckpt=checkpoints/surflo_v0.pt source.image_folder="$input_dir" source.n_images="$image_count" \
      num_query_points=100000 texture.enabled=true texture.n_iterations="${SURFLO_TEXTURE_ITERATIONS:-0}" output_dir="$output_dir/guided_default")
  fi
}

run_querysplat() {
  local scene="$1" profile="$2" output_dir="$3" input_dir repo="$REPOS_ROOT/QuerySplat" mode="${QUERYSPLAT_MODE:-both}"
  [[ "$profile" == common16 || "$profile" == dense_train || "$profile" == full || "$profile" == pinhole ]] || die "QuerySplat supports common16, dense_train and full profiles"
  [[ "$mode" == feed_forward || "$mode" == tto || "$mode" == both ]] || die "QUERYSPLAT_MODE must be feed_forward, tto, or both"
  activate_querysplat_cuda
  input_dir="$(scene_profile_dir "$scene" "$profile")"
  # Feed-forward exports include the matched-input VGGT-Omega baseline.
  if [[ "$mode" == feed_forward || "$mode" == both ]]; then
    (cd "$repo" && conda run -n querysplat python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$input_dir" --output_dir "$output_dir/feed_forward" --save_predicted_input_cameras \
      --save_vggt_input_depths --save_vggt_depth_pointcloud --vggt_depth_pointcloud_target_points 1000000 \
      --gaussian_save_opacity_threshold 0 0.05 \
      --save_gaussian_alpha_distribution --save_gaussian_scale_distribution)
  fi
  if [[ "$mode" == tto || "$mode" == both ]]; then
    (cd "$repo" && conda run -n querysplat python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$input_dir" --output_dir "$output_dir/tto" --use_tto --tto_save_step 5 10 20 \
      --save_predicted_input_cameras --save_vggt_input_depths --save_vggt_depth_pointcloud \
      --vggt_depth_pointcloud_target_points 1000000 --gaussian_save_opacity_threshold 0 0.05 \
      --save_gaussian_alpha_distribution --save_gaussian_scale_distribution)
  fi
}

setup_method() {
  case "$1" in scal3r_zju) setup_scal3r ;; moge3_glomap) setup_moge3_glomap ;; surflo) setup_surflo ;;
    querysplat) setup_querysplat ;; ssmb_lightglue) die "SSMB code/weights have not been released" ;;
    *) die "unknown method: $1" ;; esac
}

download_method() {
  case "$1" in scal3r_zju) download_scal3r ;; moge3_glomap) download_moge3_glomap ;; surflo) download_surflo ;;
    querysplat) download_querysplat ;; ssmb_lightglue) die "SSMB code/weights have not been released" ;;
    *) die "unknown method: $1" ;; esac
}

method_revision() {
  case "$1" in scal3r_zju) printf '%s\n' "$SCAL3R_REV" ;; moge3_glomap) printf 'moge:%s,glomap:%s\n' "$MOGE_REV" "$GLOMAP_REV" ;;
    surflo) printf '%s\n' "$SURFLO_REV" ;; querysplat) printf '%s\n' "$QUERYSPLAT_REV" ;; *) printf 'unknown\n' ;; esac
}

write_run_json() {
  python3 - "$@" <<'PY'
import json, os, platform, socket, subprocess, sys
(destination, method, scene, profile, status, started, ended, elapsed, exit_code, image_count, revision) = sys.argv[1:]
try:
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
except Exception:
    gpu = ""
record = {"schema_version": 1, "method": method, "scene": scene, "profile": profile, "status": status,
          "started_at_utc": started, "ended_at_utc": ended or None, "elapsed_seconds": int(elapsed),
          "exit_code": int(exit_code), "input_image_count": int(image_count), "upstream_revision": revision,
          "hostname": socket.gethostname(), "platform": platform.platform(), "gpu": gpu,
          "surflo_mode": os.environ.get("SURFLO_MODE"), "querysplat_mode": os.environ.get("QUERYSPLAT_MODE"),
          "moge3_depths_dir": os.environ.get("MOGE3_DEPTHS_DIR")}
with open(destination, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
}

run_method_impl() {
  case "$1" in scal3r_zju) run_scal3r "$2" "$3" "$4" ;; moge3_glomap) run_moge3_glomap "$2" "$3" "$4" ;;
    surflo) run_surflo "$2" "$3" "$4" ;; querysplat) run_querysplat "$2" "$3" "$4" ;;
    ssmb_lightglue) die "SSMB code/weights have not been released" ;; *) die "unknown method: $1" ;; esac
}

run_method() {
  local method="$1" scene="$2" profile="$3" input_dir image_count base output_dir attempt started ended elapsed code status revision
  [[ -d "$BUNDLE_ROOT/scenes/$scene" ]] || die "unknown scene: $scene"
  case "$profile" in common16|dense_train|full|pinhole) ;; *) die "unknown profile: $profile" ;; esac
  input_dir="$(scene_profile_dir "$scene" "$profile")"
  image_count="$(profile_image_count "$input_dir")"
  attempt="$(date -u +%Y%m%dT%H%M%SZ)"
  base="$RESULTS_ROOT/$scene/$profile/$method"
  output_dir="$base/attempt-$attempt"
  mkdir -p "$output_dir"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  revision="$(method_revision "$method")"
  write_run_json "$output_dir/run.json" "$method" "$scene" "$profile" running "$started" "" 0 0 "$image_count" "$revision"
  SECONDS=0
  set +e
  (set -e; run_method_impl "$method" "$scene" "$profile" "$output_dir") 2>&1 | tee "$output_dir/run.log"
  code="${PIPESTATUS[0]}"
  set -e
  elapsed="$SECONDS"
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ "$code" == 0 ]] && status=complete || status=failed
  write_run_json "$output_dir/run.json" "$method" "$scene" "$profile" "$status" "$started" "$ended" "$elapsed" "$code" "$image_count" "$revision"
  if [[ "$code" == 0 ]]; then
    ln -sfn "$(basename "$output_dir")" "$base/latest"
    log "Completed $method $scene $profile in ${elapsed}s: $output_dir"
  else
    log "Failed $method $scene $profile after ${elapsed}s: $output_dir"
  fi
  return "$code"
}

run_logged_action() {
  local action="$1" method="$2" logfile stamp code
  shift 2
  mkdir -p "$LOGS_ROOT"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  logfile="$LOGS_ROOT/${action}-${method}-${stamp}.log"
  set +e
  (set -e; "$@") 2>&1 | tee "$logfile"
  code="${PIPESTATUS[0]}"
  set -e
  log "$action log: $logfile"
  return "$code"
}

action="${1:-}"
case "$action" in
  preflight) preflight ;;
  setup-base) require_execution_allowed; run_logged_action setup base setup_base ;;
  checkpoint-preflight) require_execution_allowed; checkpoint_preflight ;;
  setup) [[ $# == 2 ]] || { usage; exit 2; }; require_execution_allowed; run_logged_action setup "$2" setup_method "$2" ;;
  download) [[ $# == 2 ]] || { usage; exit 2; }; require_execution_allowed; run_logged_action download "$2" download_method "$2" ;;
  run) [[ $# == 4 ]] || { usage; exit 2; }; require_execution_allowed; run_method "$2" "$3" "$4" ;;
  *) usage; exit 2 ;;
esac
