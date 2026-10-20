#!/usr/bin/env bash
# Clone the upstream code TIDES builds on (not vendored; see README for licences).
#   - 4DGaussians (Wu et al., CVPR 2024): fits the dynamic Gaussian scene per sequence.
#   - Easi3R (Chen et al., ICCV 2025): camera poses + sparse points for dynamic scenes.
# COLMAP (static scenes) is a system binary: `conda install -c conda-forge colmap` or apt.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone() {  # url dir [commit]
  if [ ! -d "third_party/$2/.git" ]; then
    git clone --recursive "$1" "third_party/$2"
  else
    echo "[skip] third_party/$2 exists"
  fi
  if [ -n "${3:-}" ]; then git -C "third_party/$2" checkout "$3"; fi
}

clone https://github.com/hustvl/4DGaussians.git 4DGaussians
clone https://github.com/Inception3D/Easi3R.git Easi3R

cat <<'MSG'

Next steps (inside your CUDA environment):
  pip install -r third_party/4DGaussians/requirements.txt
  pip install -e third_party/4DGaussians/submodules/depth-diff-gaussian-rasterization
  pip install -e third_party/4DGaussians/submodules/simple-knn
  # Easi3R (only for HS-ERGB / BS-ERGB poses): follow third_party/Easi3R/README.md
MSG
