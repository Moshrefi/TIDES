# TIDES reimplementation

Unofficial PyTorch reimplementation of **TIDES: Time-Derivative Event Simulation
via Deformable Reconstruction** (Thirgood, Ghosh, Hadfield, arXiv:2606.02058).
The authors' code is not public; this repository follows the paper as closely
as it allows. Every gap we had to fill is listed in
[docs/assumptions.md](docs/assumptions.md).

TIDES simulates event-camera streams from a dynamic 4D Gaussian Splatting scene.
Instead of differencing rendered frames, it differentiates the renderer in time:

1. **Time-tangent scene.** One forward-mode JVP through the 4DGS deformation
   network gives exact `Theta_dot(t)` for every Gaussian.
2. **Event rasterizer.** A single front-to-back pass outputs log-luminance `L`,
   its instantaneous derivative `L_dot`, opacity `alpha`, `alpha_dot`, and
   diagnostics `b`, `c`. All of them use the same visibility order as the image.
3. **Continuous-time comparator.** Threshold-crossing times are roots of a local
   linear or quadratic model, with several crossings allowed per step, so
   timestamps are not batched onto the render schedule.
4. **Risk-guided adaptive stepping.** Tiles with occlusion dynamics are
   re-rendered at sub-steps.
5. **Optional readout arbiter.** Per-tile throughput, drops and jitter depend on
   renderer-derived activity.

## Layout

```
tides/
  render/      camera + tangents, EWA projection with tangents, tangent rasterizer
  scene/       time-JVP scenes: 4DGS HexPlane (own + official wrapper), synthetic scenes
  events/      comparator, adaptive stepping, readout arbiter, sensor front end
  eval/        IG-NLL, Chamfer, Same-ts, ISI spike, Fano metrics
  simulator.py the TIDES loop
  sequence.py  frames + timestamps + poses shared by all scripts
scripts/
  download.py         datasets and pretrained weights (automatic, resumable)
  setup_third_party.sh  clones 4DGaussians (and Easi3R with --easi3r)
  install_4dgs.sh     setup_third_party.sh + builds the 4DGS CUDA extensions
  prepare_scene.py    dataset sequence -> 4DGS training scene + tides_sequence.json
  fit_4dgs.sh         fits the dynamic Gaussian scene
  simulate.py         runs TIDES
  evaluate.py         metrics against real events
  check_derivative.py float64 stage-by-stage check of the analytic derivatives (PASS / FAIL JSON)
  bench_evtween.py    TIDES + fidelity scoring inside Evtween's benchmark (docs/evtween_benchmark.md)
  demo_synthetic.py   no-download sanity check
configs/              10x and adaptive settings, ablation overlays, metric settings
tests/                derivative checks against finite differences, comparator, metrics
```

## Install

```bash
conda create -n tides python=3.10 -y && conda activate tides
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install -r requirements.txt
pytest -q                          # about 10 s on CPU
python scripts/demo_synthetic.py   # synthetic comparison, no downloads
```

To fit scenes you also need the official 4DGaussians code and its CUDA rasterizer:

```bash
module load cuda/<version matching torch.version.cuda>   # on a cluster
bash scripts/install_4dgs.sh      # clones 4DGaussians, builds its rasterizer and simple-knn
```

## Data and weights

```bash
python scripts/download.py --list
python scripts/download.py --weights required            # DUSt3R weights for Easi3R poses
python scripts/download.py --dataset eds                 # paper's EDS sequences (GT poses)
python scripts/download.py --dataset dsec --seq zurich_city_04_b
export TIDES_HSERGB_URL=...   # link from the TimeLens registration e-mail
python scripts/download.py --dataset hsergb
```

* Nothing is committed: `data/`, `weights/`, `outputs/`, `scenes/` and
  `third_party/` are git-ignored.
* HS-ERGB and BS-ERGB are released through a registration form, so you have to
  supply their link.
* Some host URLs could not be checked when this repo was written. If one fails,
  pass the current link with `--url`.
* The 4DGS scenes have no pretrained weights. They are fit per sequence, as in
  the paper.

## Full pipeline (one sequence)

Step-by-step walkthrough with expected results: [docs/run_eds_guide.md](docs/run_eds_guide.md).

The whole real-data test, with paper references and automatic checks for each claim, is in one notebook: [notebooks/tides_real_test.ipynb](notebooks/tides_real_test.ipynb).

```bash
# 1. sequence -> 4DGS scene (GT poses for EDS; --run_colmap for DSEC; --easi3r_dir for HS/BS-ERGB)
python scripts/prepare_scene.py --preset eds --seq_dir data/eds/00_peanuts_dark \
    --out scenes/eds_00 --intrinsics FX FY CX CY
# 2. fit the dynamic Gaussian scene
bash scripts/fit_4dgs.sh scenes/eds_00 eds_00
# 3. simulate events (fixed 10x budget or adaptive)
python scripts/simulate.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00 \
    --config configs/tides_adaptive.yaml --out outputs/events/eds_00_tides.h5
# 4. compare with the real events
python scripts/evaluate.py --gt_dataset eds --gt_dir data/eds/00_peanuts_dark \
    --pred outputs/events/eds_00_tides.h5
```

Ablations (Table 4) use overlays, for example
`--overlay configs/ablations/fd_slope.yaml` or `--overlay configs/ablations/no_multicrossing.yaml`.
You can override single values with `--set comparator.C_pos=0.25`.

## Status

* Done: the full simulator (Sec. 3.1-3.5), the metrics (Sec. 4.3), dataset and
  weight download, and the 4DGS fitting glue.
* Verified: `L_dot` and `alpha_dot` from the tangent rasterizer match central
  finite differences to about 1e-9 (float64), with moving Gaussians, a moving
  camera and occlusions. The HexPlane JVP matches finite differences too.
* Not yet: a fused CUDA tangent rasterizer. The PyTorch reference is exact and
  runs on GPU, but it is slower than a dedicated kernel. The baseline simulators
  and the downstream tasks are not wired in yet.

## Citation

```bibtex
@article{thirgood2026tides,
  title   = {TIDES: Time-Derivative Event Simulation via Deformable Reconstruction},
  author  = {Thirgood, Christopher and Ghosh, Dipon Kumar and Hadfield, Simon},
  journal = {arXiv preprint arXiv:2606.02058},
  year    = {2026}
}
```

This code depends on [4DGaussians](https://github.com/hustvl/4DGaussians) and
[Easi3R](https://github.com/Inception3D/Easi3R) (cloned, not vendored). Check their
licences before redistributing.
