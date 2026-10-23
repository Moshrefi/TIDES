# Running TIDES on EDS and comparing with the paper

This guide reproduces one row of the paper's Table 1 (EDS). We start with EDS
because it is the simplest of the four datasets:

* it needs no registration form;
* it ships ground-truth camera poses, so you don't need COLMAP or Easi3R;
* its sequence `00_peanuts_dark` is the one shown in Fig. 3 of the paper.

Target setup: Rails, one A100, CUDA 11.8 or 12.x, conda. Not yet run end-to-end on real data.

## 0. What happens, in one picture

```
RGB frames + poses ──(1) fit 4DGS scene──> 3D Gaussian scene that moves over time
                                              │
                                   (2) TIDES "renders" it: at any time t it computes
                                       each pixel's brightness L and how fast it is
                                       changing, L_dot, straight from the 3D scene
                                              │
                                   (3) threshold crossings -> simulated events
                                              │
real events from EDS ──────────(4) metrics──> IG-NLL, Chamfer, Same-ts, Fano, ISI spike
```

"Rendering" here means turning the 3D Gaussian scene into an image at a chosen
time. TIDES also renders the image's time derivative, and that is what makes
the event timestamps continuous.

## 1. Environment (once)

```bash
git clone https://github.com/Moshrefi/TIDES.git && cd TIDES
conda create -n tides python=3.10 -y && conda activate tides
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA module
pip install -r requirements.txt
pytest -q                                  # 12 tests, all must pass

# 4DGaussians + its CUDA rasterizer. Run on a GPU node, with the CUDA toolkit
# that matches torch (python -c "import torch; print(torch.version.cuda)"):
module load cuda/12.1                      # or whatever matches
bash scripts/install_4dgs.sh               # clones, patches and builds; ends with an import check
```

`install_4dgs.sh` stops with a clear message if `nvcc` is missing or does not
match torch's CUDA version.

Quick check that the simulator runs on the GPU (about a minute):

```bash
python scripts/demo_synthetic.py --device cuda --out outputs/demo
```

## 2. Use your Evtween copy of EDS

Use the EDS sequences you already converted for Evtween (`EDS/eds.py`). That
conversion warps the RGB frames into the event camera's pixel grid, so
rendered frames and real events line up pixel for pixel. That is exactly what
the comparison needs. Each sequence folder looks like:

```
data/eds/{train,test}/<sequence>/frames/*.png  frame_ts.npy  ev_t.npy ev_x.npy ev_y.npy ev_p.npy
```

Pick `00_peanuts_dark`, which is in Fig. 3 of the paper:

```bash
SEQ=/path/to/evtween/data/eds/train/00_peanuts_dark   # or .../test/..., wherever it landed
ls $SEQ/frames | head -3 && python -c "import numpy as np; t=np.load('$SEQ/frame_ts.npy'); print(len(t), t[-1])"
```

If you don't have it converted, run `python scripts/download.py --dataset eds --seq 00_peanuts_dark`
in TIDES, then `python EDS/eds.py --raw <folder> --only 00_peanuts_dark --out data/eds`
from the Evtween `multi-dataset-loader` branch.

## 3. Build the training scene (camera poses from COLMAP)

The converted frames no longer match EDS's ground-truth pose frame, so we
estimate poses with COLMAP. EDS scenes are static and only the camera moves,
which is the case the paper also handles with COLMAP. Start with about 2
seconds (150 frames at 75 fps) so the first run is quick:

```bash
conda create -n colmap -c conda-forge colmap -y   # once; separate env, found automatically
python scripts/prepare_scene.py --preset evtween --seq_dir $SEQ \
    --out scenes/eds_00 --max_frames 150 --run_colmap
```

* The Evtween frames are in the event camera's pixel grid, with its lens
  distortion and an off-centre principal point. 4DGS only supports a centred,
  undistorted pinhole camera, so the script first resamples the frames to one
  (`scenes/eds_00/frames_pinhole`, about 432x314) using the EDS event-camera
  calibration. COLMAP then runs with these intrinsics fixed, so it only has to
  find the poses. The same mapping is applied to the real events in step 6.
* COLMAP prints how many frames it registered, and the script stops if fewer
  than half are registered. Then rerun with `--stride 2` (or 3), which gives
  larger baselines between frames.
* If you rerun this step, delete the old output first: `rm -rf scenes/eds_00 outputs/4dgs/eds_00`.

## 4. Fit the 4D Gaussian scene (the slow step, ~30-60 min on an A100)

```bash
bash scripts/fit_4dgs.sh scenes/eds_00 eds_00
ls outputs/4dgs/eds_00/point_cloud/        # should contain iteration_XXXXX folders
```

Sanity check: 4DGaussians writes test renders and PSNR values under
`outputs/4dgs/eds_00`. If the renders look blurry or wrong, fix this step
before going further. The events can only be as good as this scene.

Then check the analytic time derivatives on this scene (Sec. 3.2, Eq. 6-15),
about a minute on a GPU:

```bash
python scripts/check_derivative.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00
```

It compares every analytic derivative with a float64 finite difference, stage
by stage: deformation JVP (Eq. 6), activations (Eq. 7-8), camera, projection,
compositing and L_dot (Eq. 9-15). It prints one row per stage and ends with
`DERIVATIVE CHECK: PASS` or `FAIL (first failing stage: ...)`. The verdict is
saved to `outputs/4dgs/eds_00/derivative_check.json`.

## 5. Simulate events

```bash
# TIDES with the fixed 10x budget (Table 1 row "TIDES (10x)")
python scripts/simulate.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00 \
    --config configs/tides_10x.yaml --out outputs/events/eds_00_10x.h5

# TIDES adaptive (row "TIDES (adaptive)")
python scripts/simulate.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00 \
    --config configs/tides_adaptive.yaml --out outputs/events/eds_00_adaptive.h5

# Frame-differencing baseline in the same renderer (ESIM-style; "Finite-diff slope" in Table 4)
python scripts/simulate.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00 \
    --config configs/tides_10x.yaml --overlay configs/ablations/fd_slope.yaml \
    --out outputs/events/eds_00_fd.h5
```

Each run prints its progress and saves a `.json` next to the `.h5`, holding the
settings and timing. Contrast thresholds can be changed with, for example,
`--set comparator.C_pos=0.25 --set comparator.C_neg=0.25`.

## 6. Compare with the real events

```bash
python scripts/evaluate.py --gt_dataset evtween --gt_dir $SEQ --scene scenes/eds_00 \
    --pred outputs/events/eds_00_10x.h5 --pred outputs/events/eds_00_adaptive.h5 \
    --pred outputs/events/eds_00_fd.h5 --json outputs/eds_00_metrics.json
```

`--scene` moves the real events to the resampled pinhole grid that TIDES
renders. The evaluation only uses the time range covered by both streams.

## 7. How to read the result against the paper

Paper, Table 1 (EDS average over all sequences):

| | IG-NLL ↓ | Chamfer ↓ |
|---|---|---|
| ESIM | 0.004748 | 0.046932 |
| TIDES (10x) | 0.003840 | 0.043245 |
| TIDES (adaptive) | 0.003733 | 0.042622 |

**Compare the order, not the exact numbers.** The paper's metric details
(binning, windows, normalisation) are in a supplementary file that is not
public, so our absolute values will be on a different scale (see
`docs/assumptions.md`, item 11). The reimplementation agrees with the paper if:

1. TIDES adaptive ≤ TIDES 10x < frame-differencing on IG-NLL and Chamfer;
2. TIDES has a lower Same-ts and ISI-spike than frame-differencing (Table 2).

If we get the authors' supplementary or code later, only `configs/metrics.yaml`
needs to change to compare absolute numbers.

## 8. Then scale up

* Run steps 3-6 on the other EDS sequences and average the results, as the paper does.
* Baselines for the same row (V2E, DVS-Voltmeter, ICNS/IEBCS) already have adapters in Evtween's
  `benchmark/event-generation` branch. Their `events.npz` files (t, x, y, p) can be passed to
  `scripts/evaluate.py --pred`. Use the same frames, crop and time window as TIDES.
* DSEC: `--preset dsec --run_colmap`, with poses from COLMAP.
* HS-ERGB / BS-ERGB: poses come from Easi3R, and those datasets need their
  registration links.
* If anything fails, send me the full error output and the command you ran.
