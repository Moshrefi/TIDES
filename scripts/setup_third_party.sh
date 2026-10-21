#!/usr/bin/env bash
# Clone the upstream code TIDES builds on (not vendored; see README for licences).
#   - 4DGaussians (Wu et al., CVPR 2024): fits the dynamic Gaussian scene per sequence.
#   - Easi3R (Chen et al., ICCV 2025), only with --easi3r: poses for dynamic scenes (HS-ERGB, BS-ERGB).
# Installing 4DGaussians' Python deps and CUDA extensions: scripts/install_4dgs.sh (calls this first).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

UP=third_party/4DGaussians
if [ ! -d "$UP/.git" ]; then
  git clone https://github.com/hustvl/4DGaussians.git "$UP"
fi
# Only the two CUDA submodules (skip the large SIBR viewer).
git -C "$UP" submodule update --init --recursive submodules/depth-diff-gaussian-rasterization  # includes glm || true
if ! git -C "$UP" submodule update --init submodules/simple-knn; then
  echo "[warn] gitlab.inria.fr unreachable; using the GitHub mirror of simple-knn"
  rm -rf "$UP/submodules/simple-knn"
  git clone https://github.com/camenduru/simple-knn.git "$UP/submodules/simple-knn"
fi

if [ "${1:-}" = "--easi3r" ] && [ ! -d third_party/Easi3R/.git ]; then
  git clone --recursive https://github.com/Inception3D/Easi3R.git third_party/Easi3R
fi
echo "third_party ready"
