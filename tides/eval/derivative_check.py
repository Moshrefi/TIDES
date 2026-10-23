"""Stage-by-stage check of the analytic time derivatives (Sec. 3.2, Eq. 6-15).

Every quantity TIDES differentiates analytically is compared with a central
finite difference of the same quantity, in float64, at a few times:

  raw.*     deformation network outputs, one forward-mode JVP      (Eq. 6)
  state.*   activated Gaussians: exp scale, sigmoid opacity, unit q (Eq. 7-8)
  camera.*  interpolated camera pose
  screen.*  projected mean, conic (inverse 2D covariance), SH colour (Sec. 3.2)
  image.*   composited rgb, alpha and L = log(Y + eps)              (Eq. 9-15)

Why float64 and a tiny step: in float32, (f(t+h) - f(t-h)) / 2h loses most of
its digits to round-off unless h is large, and a large h crosses the renderer's
discontinuities (a splat crossing the 1/255 alpha cut-off or a tile border, a
depth swap, a ReLU kink, a HexPlane time knot). The finite difference is then
the noisy side. In float64 with h ~ 1e-5 frame intervals both problems vanish.

The few elements where the function itself is not differentiable inside
[t-h, t+h] are detected by comparing the left and right one-sided differences,
reported, and left out (there the analytic value is a one-sided derivative,
which is what the simulator should use). Everywhere else the analytic value
must match to a relative 1e-4 (plus the finite difference's own round-off).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..render.projection import project_gaussians
from ..scene.deformation import time_jvp

RTOL = 1e-4  # allowed relative disagreement on smooth elements
MAX_BAD = 1e-3  # a stage passes if at most this fraction of smooth elements disagree
MAX_NONSMOOTH = 0.05  # more non-differentiable elements than this -> inconclusive, use a smaller h

EQ = {
    "raw": "Eq. 6 (JVP through the 4DGS deformation)",
    "state": "Eq. 7-8 (activations)",
    "camera": "camera trajectory",
    "screen": "Sec. 3.2 (EWA projection, SH colour)",
    "image": "Eq. 9-15 (compositing, L = log(Y + eps))",
}


@dataclass
class Sample:
    """f(t-h), f(t), f(t+h) and the analytic derivative at t, all flattened."""

    minus: torch.Tensor
    zero: torch.Tensor
    plus: torch.Tensor
    analytic: torch.Tensor


def compare(name: str, s: Sample, h: float, rtol: float = RTOL) -> dict:
    fm, f0, fp, an = (x.detach().double().reshape(-1).cpu() for x in (s.minus, s.zero, s.plus, s.analytic))
    n = int(an.numel())
    if n == 0:
        return dict(stage=name, n=0, verdict="SKIP")
    eps = torch.finfo(torch.float64).eps if s.zero.dtype == torch.float64 else torch.finfo(torch.float32).eps
    fd = (fp - fm) / (2 * h)
    right, left = (fp - f0) / h, (f0 - fm) / h
    finite = torch.isfinite(fd) & torch.isfinite(an) & torch.isfinite(right) & torch.isfinite(left)
    scale = float(np.quantile(fd[finite].abs().numpy(), 0.99)) if finite.any() else 0.0
    if scale == 0.0 and finite.any():
        scale = float(fd[finite].abs().max())
    noise = 8 * eps * (fp.abs() + fm.abs() + f0.abs()) / h  # round-off floor of the finite difference
    atol = 1e-7 * scale + noise + 1e-300
    tol = rtol * fd.abs() + atol
    smooth = finite & ((right - left).abs() <= 2 * tol)
    err = (an - fd).abs()
    ok = err <= tol
    bad = smooth & ~ok
    moving = smooth & (fd.abs() > 1e-3 * scale)
    rel = (err / (fd.abs() + atol))[smooth]
    n_smooth = int(smooth.sum())
    frac_ns = 1.0 - n_smooth / n
    frac_bad = int(bad.sum()) / max(n_smooth, 1)
    if n_smooth == 0 or frac_ns > MAX_NONSMOOTH:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if frac_bad <= MAX_BAD else "FAIL"
    worst = []
    if bad.any():
        idx = torch.nonzero(bad).squeeze(-1)
        idx = idx[torch.argsort((err / tol)[idx], descending=True)[:5]]
        worst = [dict(index=int(i), analytic=float(an[i]), finite_diff=float(fd[i])) for i in idx]
    return dict(
        stage=name, n=n, n_smooth=n_smooth, n_moving=int(moving.sum()), nonsmooth_frac=frac_ns,
        bad=int(bad.sum()), bad_frac=frac_bad,
        median_rel=float(rel.median()) if n_smooth else math.nan,
        p999_rel=float(np.quantile(rel.numpy(), 0.999)) if n_smooth else math.nan,
        derivative_scale=scale, verdict=verdict, worst=worst,
    )


def _scene_time(scene, t):
    return (t - scene.time_offset) * scene.time_scale


def _ref(scene):
    for x in list(scene.parameters()) + list(scene.buffers()):
        return x.dtype, x.device
    return torch.float64, torch.device("cpu")


@torch.no_grad()
def stage_samples(sim, t: float, h: float) -> dict[str, Sample]:
    """All checked quantities at t (analytic) and t-h, t, t+h (values)."""
    scene, traj = sim.scene, sim.traj
    dtype, device = _ref(scene)
    out: dict[str, Sample] = {}

    # Eq. 6: raw outputs of the deformation network, one JVP.
    raw = {}
    for k, tk in (("minus", t - h), ("zero", t), ("plus", t + h)):
        tt = torch.tensor(_scene_time(scene, tk), dtype=dtype, device=device)
        raw[k] = scene.raw(tt)
    _, raw_d = time_jvp(scene.raw, _scene_time(scene, t), dtype, device, rate=scene.time_scale)
    for i, nm in enumerate(("mu", "scale", "rotation", "opacity", "sh")):
        out[f"raw.{nm}"] = Sample(raw["minus"][i], raw["zero"][i], raw["plus"][i], raw_d[i])

    # Eq. 7-8: activated state.
    st = {k: scene.state(tk) for k, tk in (("minus", t - h), ("zero", t), ("plus", t + h))}
    for nm, val, dot in (("mu", "means", "means_dot"), ("scale", "scales", "scales_dot"),
                         ("rotation", "rotations", "rotations_dot"), ("opacity", "opacities", "opacities_dot"),
                         ("sh", "sh", "sh_dot")):
        out[f"state.{nm}"] = Sample(getattr(st["minus"], val), getattr(st["zero"], val),
                                    getattr(st["plus"], val), getattr(st["zero"], dot))

    # Camera.
    cams = {k: traj(tk, dtype=dtype, device=device) for k, tk in (("minus", t - h), ("zero", t), ("plus", t + h))}
    out["camera.R"] = Sample(cams["minus"].R, cams["zero"].R, cams["plus"].R, cams["zero"].R_dot)
    out["camera.t"] = Sample(cams["minus"].t, cams["zero"].t, cams["plus"].t, cams["zero"].t_dot)

    # Projection: compare the Gaussians visible at all three times.
    rc = sim.cfg.render
    sg = {k: project_gaussians(st[k], cams[k], rc.scale_modifier, rc.near) for k in st}
    common = np.intersect1d(np.intersect1d(sg["minus"].index.cpu().numpy(), sg["zero"].index.cpu().numpy()),
                            sg["plus"].index.cpu().numpy())

    def pick(k, field):
        pos = {int(g): j for j, g in enumerate(sg[k].index.cpu().numpy())}
        rows = torch.as_tensor([pos[int(g)] for g in common], dtype=torch.long, device=device)
        return getattr(sg[k], field)[rows]

    for nm, field in (("uv", "uv"), ("conic", "conic"), ("color", "color")):
        out[f"screen.{nm}"] = Sample(pick("minus", field), pick("zero", field), pick("plus", field),
                                     pick("zero", field + "_dot"))

    # Eq. 9-15: composited image.
    r = {k: sim.render(tk) for k, tk in (("minus", t - h), ("zero", t), ("plus", t + h))}
    for nm, dot in (("rgb", "rgb_dot"), ("alpha", "alpha_dot"), ("L", "L_dot")):
        out[f"image.{nm}"] = Sample(getattr(r["minus"], nm), getattr(r["zero"], nm), getattr(r["plus"], nm),
                                    getattr(r["zero"], dot))
    return out


def default_times(traj, n: int = 3) -> list[float]:
    """n times spread over the trajectory, inside keyframe intervals (not on a keyframe)."""
    T = traj.times.numpy()
    if len(T) < 2:
        return [float(T[0])]
    ids = np.unique(np.linspace(0, len(T) - 2, n + 2)[1:-1].round().astype(int)) if len(T) > 3 else [0]
    return [float(T[i] + 0.37 * (T[i + 1] - T[i])) for i in ids]


def default_step(traj, rel: float = 1e-5) -> float:
    T = traj.times.numpy()
    dt = float(np.median(np.diff(T))) if len(T) > 1 else 1.0
    return rel * dt


def sampler_check(scene, t: float) -> dict | None:
    """The AD-friendly bilinear sampler must give the same deformation as the official F.grid_sample."""
    if not hasattr(scene, "raw_official"):
        return None
    dtype, device = _ref(scene)
    tt = torch.tensor(_scene_time(scene, t), dtype=dtype, device=device)
    with torch.no_grad():
        a, b = scene.raw(tt), scene.raw_official(tt)
    diffs = [float((x - y).detach().abs().max()) / (1.0 + float(y.detach().abs().max())) for x, y in zip(a, b)]
    tol = 1e-10 if dtype == torch.float64 else 1e-4
    return dict(stage="raw = official 4DGS forward", max_rel_diff=max(diffs),
                verdict="PASS" if max(diffs) <= tol else "FAIL")


def run_check(sim, times: list[float] | None = None, h: float | None = None, log=print) -> dict:
    times = times or default_times(sim.traj)
    h = h or default_step(sim.traj)
    dtype, _ = _ref(sim.scene)
    report = dict(dtype=str(dtype).replace("torch.", ""), h=h, times=times, rtol=RTOL, max_bad=MAX_BAD, stages={})
    sc = sampler_check(sim.scene, times[0])
    if sc:
        report["sampler"] = sc
        log(f"{sc['stage']:<32} max rel diff {sc['max_rel_diff']:.1e}  {sc['verdict']}")
    per_stage: dict[str, list[dict]] = {}
    for t in times:
        for name, s in stage_samples(sim, t, h).items():
            per_stage.setdefault(name, []).append(dict(t=t, **compare(name, s, h)))

    log(f"{'stage':<16}{'elements':>12}{'non-smooth':>12}{'bad':>8}{'median rel':>12}{'p99.9 rel':>11}  verdict")
    order = {"PASS": 0, "SKIP": 0, "INCONCLUSIVE": 1, "FAIL": 2}
    for name, rows in per_stage.items():
        rows_ = [r for r in rows if r["verdict"] != "SKIP"]
        if not rows_:
            report["stages"][name] = dict(verdict="SKIP", eq=EQ[name.split(".")[0]], per_time=rows)
            continue
        verdict = max((r["verdict"] for r in rows_), key=order.get)
        agg = dict(
            eq=EQ[name.split(".")[0]], verdict=verdict,
            n=sum(r["n"] for r in rows_), bad=sum(r["bad"] for r in rows_),
            nonsmooth_frac=float(np.mean([r["nonsmooth_frac"] for r in rows_])),
            median_rel=float(np.nanmedian([r["median_rel"] for r in rows_])),
            p999_rel=float(np.nanmax([r["p999_rel"] for r in rows_])),
            per_time=rows,
        )
        report["stages"][name] = agg
        log(f"{name:<16}{agg['n']:>12,}{agg['nonsmooth_frac']:>11.2%}{agg['bad']:>8,}"
            f"{agg['median_rel']:>12.1e}{agg['p999_rel']:>11.1e}  {verdict}")

    verdicts = [s["verdict"] for s in report["stages"].values()]
    if sc:
        verdicts.append(sc["verdict"])
    first_fail = next((n for n, s in report["stages"].items() if s["verdict"] == "FAIL"), None)
    if sc and sc["verdict"] == "FAIL":
        first_fail = first_fail or sc["stage"]
    if "FAIL" in verdicts:
        report["verdict"] = "FAIL"
        report["first_failure"] = first_fail
    elif "INCONCLUSIVE" in verdicts:
        report["verdict"] = "INCONCLUSIVE"
    else:
        report["verdict"] = "PASS"
    msg = report["verdict"] + (f" (first failing stage: {first_fail}, {EQ[first_fail.split('.')[0]]})"
                               if first_fail and first_fail in report["stages"] else "")
    log(f"\nDERIVATIVE CHECK: {msg}")
    return report
