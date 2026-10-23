#!/usr/bin/env python
"""Decisive check of TIDES' analytic time derivatives (Sec. 3.2, Eq. 6-15).

Compares every analytic derivative (deformation JVP, activations, camera,
projection, compositing, L_dot) with a float64 central finite difference on the
fitted scene, stage by stage, and writes a PASS / FAIL verdict as JSON.

  python scripts/check_derivative.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00

  # no data needed
  python scripts/check_derivative.py --synthetic

Exit code 0 on PASS, 1 on FAIL, 2 if inconclusive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simulate import load_cfg  # noqa: E402

from tides import TIDES, TIDESConfig  # noqa: E402
from tides.eval.derivative_check import run_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", help="4DGS output folder (point_cloud/iteration_*/...)")
    ap.add_argument("--scene", help="folder from prepare_scene.py (contains tides_sequence.json)")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--upstream", default=os.path.join(ROOT, "third_party/4DGaussians"))
    ap.add_argument("--fourdgs_config", default=None)
    ap.add_argument("--config", action="append", default=[], help="TIDES config (render settings)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--camera", choices=["rgb", "event"], default="event")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scale")
    ap.add_argument("--times", type=float, nargs="*", default=None, help="check times in seconds (default: 3 spread)")
    ap.add_argument("--n_times", type=int, default=3)
    ap.add_argument("--h", type=float, default=None, help="finite-difference step in seconds (default 1e-5 frame)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None, help="JSON verdict (default: <model_dir>/derivative_check.json)")
    a = ap.parse_args()

    cfg_dict = load_cfg(a.config or [os.path.join(ROOT, "configs/tides_adaptive.yaml")], a.set)
    cfg_dict["device"] = a.device
    cfg = TIDESConfig.from_dict(cfg_dict)
    dtype = getattr(torch, a.dtype)

    if a.synthetic:
        from tides.scene.synthetic import occlusion_scene, static_trajectory

        scene = occlusion_scene().set_time_mapping(0.0, 0.5)
        traj = static_trajectory(width=160, height=120, t1=0.5, n=16, pan=0.3)
        out = a.out or os.path.join(ROOT, "outputs", "derivative_check_synthetic.json")
    else:
        if not (a.model_dir and a.scene):
            ap.error("--model_dir and --scene are required (or --synthetic)")
        from tides.scene.deformation import Upstream4DGS
        from tides.sequence import Sequence

        seq = Sequence.load(os.path.join(a.scene, "tides_sequence.json"))
        scene = Upstream4DGS.from_model_dir(a.model_dir, a.iteration, a.upstream, a.fourdgs_config)
        scene.set_time_mapping(seq.t0, seq.t1)
        if "background" not in cfg_dict:
            cfg.background = scene.background
        traj = seq.trajectory(a.camera if seq.event_camera else "rgb", a.scale)
        out = a.out or os.path.join(a.model_dir, "derivative_check.json")

    scene = scene.to(dtype)
    sim = TIDES(scene, traj, cfg)
    n_g = sim.scene.state(traj.t_start).num
    print(f"[check] {n_g:,} Gaussians, {sim.W}x{sim.H} px, {a.dtype} on {a.device}, background {tuple(cfg.background)}")
    times = a.times or None
    if times is None:
        from tides.eval.derivative_check import default_times

        times = default_times(traj, a.n_times)
    tic = time.time()
    report = run_check(sim, times, a.h)
    report.update(model_dir=a.model_dir, scene=a.scene, synthetic=a.synthetic, n_gaussians=n_g,
                  size=[sim.W, sim.H], background=list(cfg.background), seconds=time.time() - tic)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[check] verdict written to {out}")
    sys.exit({"PASS": 0, "FAIL": 1}.get(report["verdict"], 2))


if __name__ == "__main__":
    main()
