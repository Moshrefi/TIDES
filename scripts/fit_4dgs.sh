#!/usr/bin/env bash
# Fit a 4D Gaussian Splatting scene (official 4DGaussians code) to a prepared sequence.
#   bash scripts/fit_4dgs.sh <scene_dir from prepare_scene.py> <name> [extra train.py args]
# Output: outputs/4dgs/<name>/ (point_cloud/iteration_*/{point_cloud.ply,deformation.pth}, cfg_args)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="$(cd "$1" && pwd)"; NAME="$2"; shift 2
UP="$ROOT/third_party/4DGaussians"
[ -d "$UP" ] || { echo "run scripts/setup_third_party.sh first"; exit 1; }
CONFIG="${TIDES_4DGS_CONFIG:-$UP/arguments/hypernerf/default.py}"
OUT="$ROOT/outputs/4dgs/$NAME"
mkdir -p "$OUT"
cd "$UP"
python train.py -s "$SCENE" --port 6017 --expname "tides_$NAME" --configs "$CONFIG" --model_path "$OUT" "$@"
# Some 4DGaussians versions ignore --model_path and write to ./output/<expname>.
if [ ! -d "$OUT/point_cloud" ] && [ -d "$UP/output/tides_$NAME/point_cloud" ]; then
  rmdir "$OUT" 2>/dev/null || true
  ln -sfn "$UP/output/tides_$NAME" "$OUT"
fi
echo "fitted scene -> $OUT"
