#!/usr/bin/env python
"""Simulate an event stream with TIDES from a fitted 4DGS scene.

  python scripts/simulate.py --model_dir outputs/4dgs/eds_00 --scene scenes/eds_00 \
      --config configs/tides_adaptive.yaml --out outputs/events/eds_00_tides.h5

  # ablation: overlay one or more config files
  python scripts/simulate.py ... --config configs/tides_adaptive.yaml --overlay configs/ablations/fd_slope.yaml

  # no data needed: procedural occlusion scene
  python scripts/simulate.py --synthetic --out outputs/events/synthetic.h5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tides import TIDES, TIDESConfig  # noqa: E402
from tides.io import save_events  # noqa: E402


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out.get(k, {}) or {}, v) if isinstance(v, dict) else v
    return out


def load_cfg(paths, sets):
    d = {}
    for p in paths:
        d = deep_merge(d, yaml.safe_load(open(p)) or {})
    for s in sets or []:  # --set comparator.C_pos=0.25
        k, v = s.split("=", 1)
        node = d
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", help="4DGS output folder (point_cloud/iteration_*/...)")
    ap.add_argument("--scene", help="folder from prepare_scene.py (contains tides_sequence.json)")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--upstream", default="third_party/4DGaussians")
    ap.add_argument("--fourdgs_config", default=None, help="4DGS hyper-parameter file used for training")
    ap.add_argument("--camera", choices=["rgb", "event"], default="event",
                    help="render from the event camera when the sequence defines one (DSEC)")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scale")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], help="key=value overrides, e.g. comparator.C_pos=0.25")
    ap.add_argument("--t_start", type=float, default=None)
    ap.add_argument("--t_end", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg_dict = load_cfg(a.config + a.overlay, a.set)
    if a.device:
        cfg_dict["device"] = a.device
    cfg = TIDESConfig.from_dict(cfg_dict)

    if a.synthetic:
        from tides.scene.synthetic import occlusion_scene, static_trajectory

        scene = occlusion_scene().float()
        traj = static_trajectory(width=160, height=120, t1=0.5, n=16, dtype=__import__("torch").float32)
    else:
        from tides.scene.deformation import Upstream4DGS
        from tides.sequence import Sequence

        seq = Sequence.load(os.path.join(a.scene, "tides_sequence.json"))
        scene = Upstream4DGS.from_model_dir(a.model_dir, a.iteration, a.upstream, a.fourdgs_config)
        scene.set_time_mapping(seq.t0, seq.t1)
        if "background" not in cfg_dict:
            cfg.background = scene.background  # same background as 4DGS training
        cam = a.camera if seq.event_camera else "rgb"
        traj = seq.trajectory(cam, a.scale)

    sim = TIDES(scene, traj, cfg)
    out = sim.simulate(a.t_start, a.t_end)
    save_events(a.out, out)
    meta = dict(config=cfg.to_dict(), stats=out["stats"], n_events=int(len(out["t"])),
                width=out["width"], height=out["height"])
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[TIDES] {len(out['t'])} events -> {a.out}")
    print(json.dumps(out["stats"], indent=2, default=str))


if __name__ == "__main__":
    main()
