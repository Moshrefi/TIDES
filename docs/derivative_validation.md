# Derivative correction and saved-scene diagnostics

## Confirmed code defect

The tangent rasterizer processes depth-sorted Gaussian contributors in chunks.
When a contributor would take transmittance below `T_min`, the 3DGS convention
rejects that contributor and stops processing that pixel permanently. The old
implementation applied this prefix mask only inside the current chunk. Since
the rejected splat does not reduce the stored transmittance, a later chunk could
resume compositing a weaker contributor. Both the primal image and tangent
could therefore depend on `gauss_chunk`.

The fix retains an `alive` mask across chunks. Rejected pixels never resume.
The depth-rank allocation now also has one entry per Gaussian instead of one
entry per tile duplicate (an allocation correction, not a derivative formula).

Regression tests construct a pixel where the third contributor terminates and
the fourth would previously resume. They compare primal/tangent outputs for
chunk sizes 1, 2, 3, 4 and 10, and compare the corrected derivative with a central
finite difference away from saturation boundaries.

This is a confirmed control-flow defect from code inspection. It is NOT yet
established as the cause of the reported real `eds_00` derivative-check failure.
Fixing a primal/tangent branch bug does not prove all real-scene derivatives are
correct. Existing event caches do not change when source code changes.

## Run without fitting or simulating again

Use the existing TIDES environment and the existing fitted model:

```bash
python -m pytest tests/test_render.py tests/test_chunk_termination.py -q
python scripts/check_saved_derivatives.py \
  --scene scenes/eds_00 \
  --model-dir outputs/4dgs/eds_00 \
  --out outputs/diagnostics/eds_00_derivatives_v2.json
```

Scene/model/output paths are configurable. A new output name is required if the
report already exists. This command never runs COLMAP, training or event
simulation. By default it checks 12 distributed tiles at three times inside
camera-pose intervals; use `--tile-count` to expand coverage.

The diagnostic uses h, h/2 and h/4 central differences. It reports error over
all moving pixels AND the FD-converged subset, along with subset coverage and
contributor-count stability. Equal counts do not guarantee identical visibility
branches; this limitation is retained. Keyframe crossings are avoided because
piecewise camera velocity may be nondifferentiable there. Float32 render
rounding remains even though subtraction uses float64.

- PASS: at least 90% of moving pixels pass FD convergence/branch-count checks,
  median relative tangent error below 5%, and over 90% of the eligible pixels
  within the declared absolute/relative tolerance, at every tested time.
- FAIL: converged comparisons disagree, or nonfinite rendered derivatives occur.
- INCONCLUSIVE: insufficient converged moving pixels or no motion. This is not
  silently converted to PASS or proof of a derivative bug.

Non-PASS exits with code 2 but still writes the report. Inspect its per-time
errors before changing tolerances or regenerating cached events. Once verified,
future events should be generated into new paths with the corrected commit and
generation settings recorded. Never relabel old NPZs as corrected outputs.

## Validation scope

Python sources compile locally. PyTorch/pytest and the fitted Rails model are
not available in the development environment, so the new Torch regressions and
real-scene derivative checks have NOT been executed here. No passing derivative
result, regenerated output, or reproduction of a paper table is claimed.

## Relation to the paper

This repository is explicitly an unofficial reimplementation. Our Evtween
comparison currently uses a single short EDS clip, saved cached TIDES output,
baseline frames, a common crop/remapping mask, and locally implemented metrics.
Generation provenance and GT-based calibration are not fully recovered for old
caches. Matching metric names is insufficient to compare absolute values with
published tables; datasets/splits/windows, input/render budgets, calibration,
metric definitions and aggregation must all match. New voxel MAE/RMSE scores
are additional benchmark metrics, not claimed paper metrics.
