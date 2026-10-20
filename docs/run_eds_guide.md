# Running TIDES on EDS and comparing with the paper

This guide reproduces one row of the paper's Table 1 (EDS). We start with EDS
because it is the simplest of the four datasets:

* it needs no registration form;
* it ships ground-truth camera poses, so you don't need COLMAP or Easi3R;
* its sequence `00_peanuts_dark` is the one shown in Fig. 3 of the paper.

Tested setup: a Linux GPU node with an A100, CUDA 11.8 or 12.x, and conda.

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

bash scripts/setup_third_party.sh          # clones 4DGaussians + Easi3R into third_party/
pip install -r third_party/4DGaussians/requirements.txt
pip install -e third_party/4DGaussians/submodules/depth-diff-gaussian-rasterization
pip install -e third_party/4DGaussians/submodules/simple-knn
```

If the two `submodules` installs fail, the CUDA toolkit is not visible. Load it
first, for example with `module load cuda/12.1` on your cluster, then retry.

Quick check that the simulator runs on the GPU (about a minute):

```bash
python scripts/demo_synthetic.py --device cuda --out outputs/demo
```

## 2. Download one EDS sequence (~a few GB)

```bash
python scripts/download.py --dataset eds --seq 00_peanuts_dark
ls data/eds/00_peanuts_dark
```

If the download link fails, open https://rpg.ifi.uzh.ch/eds.html, copy the
link for `00_peanuts_dark`, and pass it with `--url <link>`. If you already
have EDS on disk, skip the download and use that folder as `--seq_dir` below.

Check that the folder has the images, their timestamps, `stamped_groundtruth.txt`
(the poses), the events `.h5` file and a calibration file. Note the
names if they differ from `images/`, `images_timestamps.txt`, `events.h5`, `calib.yaml`.

## 3. Build the training scene

Start with a short window of about 3 seconds so the first run is quick:

```bash
python scripts/prepare_scene.py --preset eds --seq_dir data/eds/00_peanuts_dark \
    --out scenes/eds_00 --max_frames 90
```

* If it says `intrinsics unknown`, add `--intrinsics FX FY CX CY`, taking the
  numbers from the sequence's calibration file (RGB camera).
* If your files have other names, use `--frames_glob`, `--timestamps` and `--poses_tum`.
* At the end it prints the frame count and duration. Check that they look right.

## 4. Fit the 4D Gaussian scene (the slow step, ~30-60 min on an A100)

```bash
bash scripts/fit_4dgs.sh scenes/eds_00 eds_00
ls outputs/4dgs/eds_00/point_cloud/        # should contain iteration_XXXXX folders
```

Sanity check: 4DGaussians writes test renders and PSNR values under
`outputs/4dgs/eds_00`. If the renders look blurry or wrong, fix this step
before going further. The events can only be as good as this scene.

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
python scripts/evaluate.py --gt_dataset eds --gt_dir data/eds/00_peanuts_dark \
    --pred outputs/events/eds_00_10x.h5 --pred outputs/events/eds_00_adaptive.h5 \
    --pred outputs/events/eds_00_fd.h5 --json outputs/eds_00_metrics.json
```

The evaluation only uses the time range covered by both streams.

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
* DSEC: `--preset dsec --run_colmap`, with poses from COLMAP.
* HS-ERGB / BS-ERGB: poses come from Easi3R, and those datasets need their
  registration links.
* If anything fails, send me the full error output and the command you ran.
