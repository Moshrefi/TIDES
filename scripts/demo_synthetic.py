#!/usr/bin/env python
"""Self-contained sanity check, no downloads needed.

Simulates a procedural occlusion scene (a blob sweeping over a textured wall)
with several variants, using a very densely sampled TIDES run as the
reference stream, and prints the paper's metrics for each:

  reference     TIDES, 60 queries / frame (dense proxy for ground truth)
  fd-1x         finite-difference slope, 1 query / frame (frame-based simulator)
  fd-10x        finite-difference slope, 10 queries / frame (ESIM-style, 10x budget)
  tides-10x     analytic time-tangent slope, 10 queries / frame
  tides-adapt   analytic slope + risk-guided adaptive sub-stepping

  python scripts/demo_synthetic.py [--device cuda] [--out outputs/demo]
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tides import TIDES, TIDESConfig  # noqa: E402
from tides.eval.metrics import Events, MetricConfig, all_metrics  # noqa: E402
from tides.io import save_events, to_event_frame  # noqa: E402
from tides.scene.synthetic import occlusion_scene, static_trajectory  # noqa: E402


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = t + d
        return len(d).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--duration", type=float, default=0.4)
    ap.add_argument("--out", default="outputs/demo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scene = occlusion_scene(speed=2.0)
    traj = static_trajectory(a.width, a.height, t1=a.duration, n=13, pan=0.2)  # 30 fps keyframes

    def run(name, **kw):
        cfg = TIDESConfig(device=a.device)
        cfg.comparator.C_pos = cfg.comparator.C_neg = 0.15
        cfg.comparator.n_max = 16
        for k, v in kw.items():
            setattr(cfg, k, v)
        out = TIDES(scene, traj, cfg).simulate(verbose=False)
        save_events(os.path.join(a.out, f"{name}.npz"), out)
        print(f"  {name:12s} {len(out['t']):8d} events  {out['stats']['wall_time_s']:.1f}s")
        return out

    print("simulating ...")
    ref = run("reference", queries_per_frame=60, adaptive=False)
    variants = {
        "fd-1x": run("fd-1x", queries_per_frame=1, adaptive=False, slope="finite_diff"),
        "fd-10x": run("fd-10x", queries_per_frame=10, adaptive=False, slope="finite_diff"),
        "tides-10x": run("tides-10x", queries_per_frame=10, adaptive=False),
        "tides-adapt": run("tides-adapt", queries_per_frame=10, adaptive=True),
    }

    mc = MetricConfig(window=0.05, v=200.0)
    R = Events.from_dict(ref)
    keys = ["IG-NLL", "Chamfer", "Same-ts", "Fano", "ISI-spike"]
    print(f"\n{'variant':12s}" + "".join(f"{k:>11s}" for k in keys))
    for name, out in variants.items():
        m = all_metrics(R, Events.from_dict(out), a.width, a.height, mc)
        print(f"{name:12s}" + "".join(f"{m[k]:11.4f}" for k in keys))

    t_mid = a.duration / 2
    for name, out in [("reference", ref)] + list(variants.items()):
        write_png(os.path.join(a.out, f"{name}.png"), to_event_frame(out, a.width, a.height, t_mid, t_mid + 0.01))
    print(f"\nevent frames and streams written to {a.out}/")


if __name__ == "__main__":
    main()
