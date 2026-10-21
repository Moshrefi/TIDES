#!/usr/bin/env bash
# One-time install of what scripts/fit_4dgs.sh needs, into the CURRENT conda env.
# Run it on a GPU node with the CUDA toolkit loaded, e.g.:
#   module load cuda/12.1   # same major.minor as `python -c "import torch; print(torch.version.cuda)"`
#   bash scripts/install_4dgs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_third_party.sh
[ -f third_party/4DGaussians/submodules/depth-diff-gaussian-rasterization/third_party/glm/glm/glm.hpp ] || { echo "glm missing in the rasterizer; check network access to github.com"; exit 1; }
UP=third_party/4DGaussians

TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
[ -n "$TORCH_CUDA" ] || { echo "This PyTorch has no CUDA support; install a CUDA build first (see README)."; exit 1; }
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Load the CUDA toolkit matching PyTorch's CUDA $TORCH_CUDA (e.g. 'module load cuda/$TORCH_CUDA') and rerun."
  exit 1
fi
NVCC_CUDA=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "PyTorch $TORCH_VER (CUDA $TORCH_CUDA), nvcc CUDA $NVCC_CUDA"
if [ "${TORCH_CUDA%%.*}" != "${NVCC_CUDA%%.*}" ]; then
  echo "PyTorch was built for CUDA $TORCH_CUDA but nvcc is CUDA $NVCC_CUDA; the extensions cannot build."
  echo "Load the CUDA toolkit matching PyTorch (see 'module avail cuda'), e.g.:"
  echo "  module load cuda/$TORCH_CUDA"
  echo "then rerun this script. (Or reinstall PyTorch for your nvcc's CUDA version.)"
  exit 1
fi

# Keep this torch build: without the pin, pip pulls the latest torchvision for lpips,
# which drags in the latest torch (a different CUDA build).
PIN=$(mktemp)
echo "torch==$TORCH_VER" > "$PIN"
echo "numpy<2" >> "$PIN"
if ! python -c "import torchvision" 2>/dev/null; then
  TV=$(python -c "v='$TORCH_VER'.split('.'); print(f'0.{int(v[1]) + 15}.{v[2]}')")
  pip install "torchvision==$TV" -c "$PIN" --index-url "https://download.pytorch.org/whl/cu$(echo "$TORCH_CUDA" | tr -d .)"
fi

# Upstream requirements.txt pins torch==1.13.1; install the rest without touching torch.
# yapf 0.40.1: newer yapf breaks mmcv 1.x Config loading.
# mmcv 1.6.0 has no wheel and its setup.py needs pkg_resources, which setuptools>=81 removed:
# build it against an older setuptools, without pip's isolated build env.
# opencv < 4.11 and numpy < 2 keep the env compatible with torch builds made for NumPy 1.x.
pip install -c "$PIN" "setuptools<70" wheel "opencv-python<4.11"
pip install -c "$PIN" --no-build-isolation "mmcv==1.6.0" "yapf==0.40.1"
pip install -c "$PIN" addict lpips plyfile pytorch_msssim open3d "imageio[ffmpeg]" tensorboard

# Recent compilers need <cstdint> in the 3DGS rasterizer.
H=$UP/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
grep -q "<cstdint>" "$H" || sed -i '1i #include <cstdint>' "$H"

# Newer CUDA headers no longer pull in <cfloat> (FLT_MAX) for simple-knn.
K=$UP/submodules/simple-knn/simple_knn.cu
grep -q "<cfloat>" "$K" || sed -i '1i #include <cfloat>' "$K"

# Build for both A100 (8.0) and H100 (9.0) so the same env works on any Rails GPU node;
# by default only the GPU of the current node is targeted.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
rm -rf "$UP/submodules/simple-knn/build" "$UP/submodules/depth-diff-gaussian-rasterization/build"
pip install -c "$PIN" --no-build-isolation "$UP/submodules/depth-diff-gaussian-rasterization" "$UP/submodules/simple-knn"
python -c "import diff_gaussian_rasterization, simple_knn._C, mmcv; print('4DGaussians ready')"
