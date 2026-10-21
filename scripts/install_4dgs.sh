#!/usr/bin/env bash
# One-time install of what scripts/fit_4dgs.sh needs, into the CURRENT conda env.
# Run it on a GPU node with the CUDA toolkit loaded, e.g.:
#   module load cuda/12.1   # same major.minor as `python -c "import torch; print(torch.version.cuda)"`
#   bash scripts/install_4dgs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_third_party.sh
UP=third_party/4DGaussians

TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
[ -n "$TORCH_CUDA" ] || { echo "This PyTorch has no CUDA support; install a CUDA build first (see README)."; exit 1; }
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Load the CUDA toolkit matching PyTorch's CUDA $TORCH_CUDA (e.g. 'module load cuda/$TORCH_CUDA') and rerun."
  exit 1
fi
echo "PyTorch CUDA $TORCH_CUDA, nvcc: $(nvcc --version | tail -1)"

# Upstream requirements.txt pins torch==1.13.1; install the rest without touching torch.
# yapf 0.40.1: newer yapf breaks mmcv 1.x Config loading.
# mmcv 1.6.0 has no wheel and its setup.py needs pkg_resources, which setuptools>=81 removed:
# build it against an older setuptools, without pip's isolated build env.
# opencv < 4.11 and numpy < 2 keep the env compatible with torch builds made for NumPy 1.x.
pip install "setuptools<70" wheel "numpy<2" "opencv-python<4.11"
pip install --no-build-isolation "mmcv==1.6.0" "yapf==0.40.1"
pip install addict lpips plyfile pytorch_msssim open3d "imageio[ffmpeg]" tensorboard "numpy<2"

# Recent compilers need <cstdint> in the 3DGS rasterizer.
H=$UP/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
grep -q "<cstdint>" "$H" || sed -i '1i #include <cstdint>' "$H"

pip install --no-build-isolation "$UP/submodules/depth-diff-gaussian-rasterization" "$UP/submodules/simple-knn"
python -c "import diff_gaussian_rasterization, simple_knn._C, mmcv; print('4DGaussians ready')"
