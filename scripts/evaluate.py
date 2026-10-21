#!/usr/bin/env python
"""Compare simulated events with real ones (Tables 1-2 metrics).

  python scripts/evaluate.py --gt_dataset eds --gt_dir data/eds/00_peanuts_dark \
      --pred outputs/events/eds_00_tides.h5 [--pred other_sim.h5 ...] --width 640 --height 480

Time windows are restricted to the overlap of both streams. Results are
printed as a table and written to --json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tides.eval.metrics import Events, MetricConfig, all_metrics  # noqa: E402
from tides.io import READERS, load_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="ground-truth event file (h5/npz/txt)")
    ap.add_argument("--gt_dataset", choices=list(READERS))
    ap.add_argument("--gt_dir")
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--t_start", type=float, default=None)
    ap.add_argument("--t_end", type=float, default=None)
    ap.add_argument("--metrics_config", default="configs/metrics.yaml")
    ap.add_argument("--json", default=None)
    ap.add_argument("--scene", default=None, help="prepare_scene.py output: maps real events to the rendered pixel grid")
    a = ap.parse_args()

    mc = MetricConfig(**(yaml.safe_load(open(a.metrics_config)) if os.path.exists(a.metrics_config) else {}))
    gt = load_events(a.gt) if a.gt else READERS[a.gt_dataset](a.gt_dir)
    if a.scene:
        from tides.sequence import Sequence

        gt = Sequence.load(os.path.join(a.scene, "tides_sequence.json")).remap_events(gt)
    results = {}
    for p in a.pred:
        pr = load_events(p)
        W = a.width or pr.get("width") or int(max(gt["x"].max(), pr["x"].max()) + 1)
        H = a.height or pr.get("height") or int(max(gt["y"].max(), pr["y"].max()) + 1)
        t0 = max(gt["t"].min(), pr["t"].min()) if a.t_start is None else a.t_start
        t1 = min(gt["t"].max(), pr["t"].max()) if a.t_end is None else a.t_end
        G = Events.from_dict(gt).window(t0, t1)
        P = Events.from_dict(pr).window(t0, t1)
        results[os.path.basename(p)] = all_metrics(G, P, W, H, mc)

    keys = ["IG-NLL", "Chamfer", "Same-ts", "Fano", "ISI-spike", "n_sim", "n_real"]
    print(f"{'simulator':30s}" + "".join(f"{k:>12s}" for k in keys))
    for name, r in results.items():
        print(f"{name:30s}" + "".join(f"{r[k]:12.6f}" if isinstance(r[k], float) else f"{r[k]:12d}" for k in keys))
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
