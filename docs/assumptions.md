# What the paper specifies, and what we had to choose

TIDES (arXiv:2606.02058) describes the method but its code and supplementary
material are not public. This file lists every place where this
reimplementation had to make a choice. All of them are configurable.

## Implemented as written in the paper

| Paper | Code |
|---|---|
| Eq. 1-4: EWA splatting, front-to-back compositing, tau clamped at tau_max | `tides/render/rasterizer.py` |
| Eq. 5, 15: `L = log(Y + eps)`, `L_dot = Y_dot / (Y + eps)` | same |
| Eq. 6: Theta_dot from one forward-mode JVP of the deformation network along time | `tides/scene/deformation.py::time_jvp` |
| Eq. 7-8: analytic tangents through exp, sigmoid, quaternion normalisation | `tides/render/tangent_ops.py` |
| Tangents through projection Jacobian, screen covariance, SH colour and view direction | `tides/render/projection.py` |
| Eq. 9-14: p_dot, g_dot, tau_dot (zero when saturated), w_dot, T_dot, I_dot in the primal compositing order | `tides/render/rasterizer.py` |
| Non-static camera; velocity from neighbouring timestamps when not given | `tides/render/camera.py::CameraTrajectory` |
| Residual-state comparator, asymmetric thresholds, per-pixel jitter, leak | `tides/events/comparator.py` |
| Eq. 16-17: linear and quadratic local models, earliest root, multi-crossing up to N_max | same |
| Eq. 18: risk from b, c, alpha_dot; dt <- dt * kappa(r); no emission when b < b_min | `tides/events/adaptive.py` |
| Eq. 19: tile FIFO arbiter, K_m, p_drop, sigma_t conditioned on psi_m | `tides/events/arbiter.py` |
| Sensor front end F: PSF blur, photoreceptor low-pass, centre-surround | `tides/events/frontend.py` |
| Metrics: IG-NLL, motion-scaled Chamfer, Same-ts, ISI spike ratio, Fano error (Eq. 20) | `tides/eval/metrics.py` |
| 4DGS (Wu et al. 2024) scene substrate, COLMAP / Easi3R poses | `scripts/` |

## Our choices (the paper leaves these open)

1. **Rasterizer constants** follow 3DGS: 16x16 tiles, alpha skipped below 1/255,
   tau_max = 0.99, early termination at T < 1e-4, 0.3 px low-pass dilation.
2. **log epsilon** = 1e-3 on luminance in [0, 1]; Rec.601 luminance weights.
3. **Coverage proxy b** = accumulated blending weight sum_k w_k (= alpha). Set
   `render.coverage: tau_sum` to use sum_k tau_k instead.
4. **Risk normalisation**: norm(b) = b, norm(c) = min(c / 32, 1),
   norm(|alpha_dot|) = min(|alpha_dot| dt / 0.25, 1). Weights w_b = 0.2,
   w_c = 0.3, w_alpha = 0.5.
5. **kappa(r) = 1 - (1 - kappa_min) r** with kappa_min = 1/8. Risk is pooled per
   16x16 tile (max over pixels with b >= b_min); a tile with step factor kappa
   is split into ceil(1/kappa) sub-steps, and only those tiles are re-rendered
   at the sub-times. The paper says "adaptive sub-pose rendering request" and
   "tile risk" but not the exact scheme.
6. **Quadratic model by default.** Our loop always has the render at the end of
   the step, so the quadratic model (S0, S0_dot, S1) is used whenever possible;
   set `comparator.local_model: linear` for Eq. 16 only.
7. **Residual handling.** If the signal is already past a threshold at the start
   of a step (local-model mismatch), we emit the missing events at t0
   (`comparator.residual: emit`) or silently absorb them (`absorb`).
8. **Masked crossings** (b < b_min) still move S_ref, so un-masking does not
   produce a burst.
9. **Arbiter**: 32x32 px tiles, 1 us scanner tick, K0 = 4, beta_K = 0.75,
   p0 = 0, eta = 0.05, sigma0 = 0, gamma_sigma = 20 us. psi_m is a weighted sum
   of the tile means of |L_dot|/C (ref 2000 ev/px/s), |alpha_dot| (ref 50/s) and
   c (ref 32), computed at the start of each base interval. Events that are not
   queued keep their exact timestamp.
10. **Time mapping for 4DGS**: the HyperNeRF export writes `warp_id` in
    microseconds, so 4DGS's normalised time is linear in real time; the JVP
    tangent is scaled by d(scene time)/d(seconds).
11. **Metrics** (supplementary not available): 50 ms windows; v = 1000 px/s;
    1 px voxels; Chamfer = mean of both directions, divided by the image
    diagonal; IG fitted by maximum likelihood to pooled per-pixel ISIs (seconds,
    clamped at 1 us); ISI spike = ISI < 10 us (plus optional clock-quantum
    test); Fano over 8x8 px bins with 2 ms sub-bins inside each window.
    Absolute values will not match Tables 1-2, but rankings between simulators
    run under the same settings are meaningful.

## Not implemented yet

* A fused CUDA kernel for the tangent rasterizer (the PyTorch reference is exact
  but slower; see README).
* Baseline simulators (ESIM, v2e, DVS-Voltmeter, ICNS/IEBCS) inside the shared
  renderer protocol, and the downstream tasks (E2VID, TimeLens-XL, DAES, ESS).
