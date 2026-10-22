#!/usr/bin/env python
"""TIDES inside Evtween's event-generation benchmark, plus fidelity scoring for every method.

Evtween's benchmark (branch benchmark/event-generation, benchmarks/event_generation)
cuts one shared clip per run:

    RUN/input.json            source sequence, frame indices, absolute_t0, duration, crop
    RUN/input/frames/*.png    centre-cropped frames given to every method
    RUN/ground_truth.npz      real events in the same crop and time base
    RUN/<method>/events.npz   t (s, relative to the first frame), x, y (crop pixels), p (+-1)

This script adds the missing pieces, without changing Evtween:

  tides   run TIDES for a benchmark run and write RUN/tides/events.npz in exactly that format
  score   fidelity metrics (Tables 1-2: IG-NLL, Chamfer, Same-ts, Fano, ISI spike) for every
          method in one or more runs, against ground_truth.npz -> CSV + per-dataset summary
  batch   for a list of sequences: cut the clip with Evtween's own prepare(), run the
          baselines with Evtween's run_model() and interpreters.json, run TIDES, then score

TIDES is a reconstruction method: it fits a 4DGS scene to the RGB frames *around*
the clip (default 150 frames, no events), then simulates only the clip interval.
Events are rendered on an undistorted pinhole grid and mapped back to the
benchmark's crop pixels (see tides/pinhole.py).

  python scripts/bench_evtween.py tides --run /path/to/evtween/benchmarks/event_generation/runs/<id>
  python scripts/bench_evtween.py score --runs runs/* --csv results.csv
  python scripts/bench_evtween.py batch --evtween /path/to/evtween --sequences seqs.txt --out runs_eds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ("eds", "hqevfi", "bsergb", "ced")
GAMMA = {"bsergb": 2.2}  # Evtween config.yaml: PNG value^gamma = linear light


# ----------------------------------------------------------------------------- helpers
def dataset_of(source: str) -> str:
    parts = [p.lower() for p in Path(source).parts]
    for d in DATASETS:
        if d in parts:
            return d
    return "unknown"


def save_bench_events(path, t, x, y, p, duration, width, height):
    """Same filtering, ordering and dtypes as Evtween's benchmark.save_events."""
    t, x, y, p = map(np.asarray, (t, x, y, p))
    m = (t >= 0) & (t < duration) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    o = np.argsort(t[m], kind="stable")
    np.savez_compressed(path, t=t[m][o].astype("float64"), x=x[m][o].astype("int32"),
                        y=y[m][o].astype("int32"), p=np.where(p[m][o] > 0, 1, -1).astype("int8"))


def run(cmd, log=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ, PYTHON=sys.executable)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT if log else None)


# ----------------------------------------------------------------------------- tides
def time_scale(source: str) -> float:
    """prepare_scene.py reads frame_ts.npy as microseconds when values exceed 1e4; follow the same rule."""
    ts = np.load(os.path.join(source, "frame_ts.npy"))
    return 1e-6 if np.nanmax(ts) > 1e4 else 1.0


def fit_window(source: str, indices: list, fit_frames: int):
    """Frame range (seconds, as prepare_scene.py reads them) for the 4DGS fit: fit_frames frames centred on the clip."""
    ts = np.load(os.path.join(source, "frame_ts.npy")).astype(np.float64) * time_scale(source)
    s = max(1, int(indices[1] - indices[0])) if len(indices) > 1 else 1
    c = 0.5 * (indices[0] + indices[-1])
    half = 0.5 * max(fit_frames, len(indices)) * s
    j0 = int(max(0, np.floor(c - half)))
    j1 = int(min(len(ts) - 1, np.ceil(c + half)))
    j0, j1 = min(j0, indices[0]), max(j1, indices[-1])
    return j0, j1, s, float(ts[j0]), float(ts[j1])


def build_scene(source, j0, j1, stride, t0, t1, key, a, log):
    scene_dir = ROOT / "scenes" / "bench" / key
    model_dir = ROOT / "outputs" / "4dgs" / f"bench_{key}"
    if not (scene_dir / "tides_sequence.json").exists():
        cmd = [sys.executable, "scripts/prepare_scene.py", "--preset", "evtween", "--seq_dir", source,
               "--out", scene_dir, "--t_start", f"{t0 - 1e-6:.9f}", "--t_end", f"{t1 + 1e-6:.9f}",
               "--stride", str(stride)]
        cmd += ["--easi3r_dir", a.easi3r_dir] if a.easi3r_dir else ["--run_colmap"]
        run(cmd, log)
    if not (model_dir / ".tides_fit_ok").exists():
        import shutil

        shutil.rmtree(model_dir, ignore_errors=True)
        run(["bash", "scripts/fit_4dgs.sh", scene_dir, f"bench_{key}"], log)
        (model_dir / ".tides_fit_ok").write_text("ok\n")
    return scene_dir, model_dir


def simulate_tides(scene_dir, model_dir, t0, t1, a, gamma, C=None):
    import torch  # noqa: F401
    import yaml

    from tides import TIDES, TIDESConfig
    from tides.scene.deformation import Upstream4DGS
    from tides.sequence import Sequence
    import importlib.util

    spec = importlib.util.spec_from_file_location("tides_simulate_cli", ROOT / "scripts" / "simulate.py")
    sim_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_cli)
    load_cfg = sim_cli.load_cfg

    seq = Sequence.load(str(scene_dir / "tides_sequence.json"))
    scene = Upstream4DGS.from_model_dir(str(model_dir), -1, str(ROOT / "third_party" / "4DGaussians"))
    scene.set_time_mapping(seq.t0, seq.t1)
    d = load_cfg([str(ROOT / c) for c in a.config], a.set)
    d.setdefault("background", list(scene.background))
    d.setdefault("render", {})["gamma"] = gamma
    if C is not None:
        d.setdefault("comparator", {}).update(C_pos=C, C_neg=C)
    if a.device:
        d["device"] = a.device
    cfg = TIDESConfig.from_dict(d)
    out = TIDES(scene, seq.trajectory("rgb"), cfg).simulate(t0, t1, verbose=False)
    return seq, out, cfg


def calibrate_C(scene_dir, model_dir, t0, t1, n_real, a, gamma):
    """Contrast threshold that matches the real event count on the clip (uses ground truth: opt-in)."""
    Cs, ns = [], []
    for C in (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5):
        _, out, _ = simulate_tides(scene_dir, model_dir, t0, t1, a, gamma, C)
        Cs.append(C)
        ns.append(len(out["t"]))
        print(f"  C = {C:.2f}: {ns[-1]:,} events (real {n_real:,})", flush=True)
    ns = np.array(ns, float)
    return float(np.exp(np.interp(np.log(n_real), np.log(ns[::-1] + 1), np.log(np.array(Cs)[::-1]))))


def cmd_tides(a):
    rdir = Path(a.run).resolve()
    meta = json.loads((rdir / "input.json").read_text())
    out_dir = rdir / a.name
    out_dir.mkdir(exist_ok=a.overwrite)
    source, idx = meta["source"], meta["indices"]
    ds = dataset_of(source)
    gamma = a.gamma if a.gamma is not None else GAMMA.get(ds, 1.0)
    j0, j1, stride, f0, f1 = fit_window(source, idx, a.fit_frames)
    key = f"{ds}_{Path(source).name}_{j0}_{j1}_s{stride}"
    begin = time.perf_counter()
    with open(out_dir / "run.log", "w") as log:
        scene_dir, model_dir = build_scene(source, j0, j1, stride, f0, f1, key, a, log)
    sf = time_scale(source)  # benchmark times are in frame_ts.npy units
    t_abs0 = float(meta["absolute_t0"]) * sf
    t_abs1 = t_abs0 + float(meta["duration"]) * sf
    C = a.C
    if a.calibrate_count:
        gt = np.load(rdir / "ground_truth.npz")
        C = calibrate_C(scene_dir, model_dir, t_abs0, t_abs1, len(gt["t"]), a, gamma)
        print(f"[tides] calibrated C = {C:.3f}")
    seq, ev, cfg = simulate_tides(scene_dir, model_dir, t_abs0, t_abs1, a, gamma, C)

    # pinhole grid -> dataset pixel grid -> benchmark crop
    x, y = np.asarray(ev["x"], np.float64), np.asarray(ev["y"], np.float64)
    r = (seq.meta or {}).get("remap")
    if r:
        from tides.pinhole import PinholeRemap

        x, y = PinholeRemap.from_dict(r).out_to_src(x, y)
    x0, y0, w, h = meta["crop"]
    xs = np.round(x).astype(np.int64) - x0
    ys = np.round(y).astype(np.int64) - y0
    save_bench_events(out_dir / "events.npz", (np.asarray(ev["t"]) - t_abs0) / sf, xs, ys, ev["p"],
                      meta["duration"], meta["width"], meta["height"])
    n_saved = len(np.load(out_dir / "events.npz")["t"])
    params = dict(config=a.config, set=a.set, C=C, gamma=gamma, fit_frames=a.fit_frames,
                  fit_indices=[j0, j1, stride], scene=str(scene_dir), model=str(model_dir),
                  poses="easi3r" if a.easi3r_dir else "colmap", calibrate_count=a.calibrate_count)
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    (out_dir / "execution.json").write_text(json.dumps(dict(model=a.name, seconds=time.perf_counter() - begin,
                                                            returncode=0, params=params, tides_config=cfg.to_dict()),
                                                       indent=2, default=str))
    print(f"[tides] {n_saved:,} events in the crop -> {out_dir / 'events.npz'}")


# ----------------------------------------------------------------------------- score
def score_runs(runs, a):
    import pandas as pd
    import yaml

    from tides.eval.metrics import Events, MetricConfig, all_metrics, isi_spike_ratio, same_ts_fraction

    mc = MetricConfig(**(yaml.safe_load(open(ROOT / a.metrics_config)) or {}))
    rows = []
    for rdir in map(Path, runs):
        if not (rdir / "input.json").exists() or not (rdir / "ground_truth.npz").exists():
            print(f"[skip] {rdir}: no input.json / ground_truth.npz")
            continue
        meta = json.loads((rdir / "input.json").read_text())
        W, H = int(meta["width"]), int(meta["height"])
        ts = np.load(rdir / "input" / "frame_ts.npy")
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        t0, t1 = a.trim_frames * dt, float(meta["duration"]) - a.trim_frames * dt
        G = Events.from_dict(dict(np.load(rdir / "ground_truth.npz"))).window(t0, t1)
        base = dict(run=rdir.name, dataset=dataset_of(meta["source"]), sequence=Path(meta["source"]).name,
                    t0=t0, t1=t1)
        rows.append(dict(base, method="(real camera)", n_sim=len(G), n_real=len(G),
                         **{"Same-ts": same_ts_fraction(G, mc), "ISI-spike": isi_spike_ratio(G, W, mc)}))
        for ev in sorted(rdir.glob("*/events.npz")):
            P = Events.from_dict(dict(np.load(ev))).window(t0, t1)
            m = all_metrics(G, P, W, H, mc) if len(P) else {}
            rows.append(dict(base, method=ev.parent.name, **m))
            print(f"  {rdir.name:28s} {ev.parent.name:14s} " +
                  " ".join(f"{k}={m[k]:.4g}" for k in ("IG-NLL", "Chamfer", "Fano") if k in m), flush=True)
        for vox in sorted(rdir.glob("*/voxels.npz")):
            print(f"  {rdir.name:28s} {vox.parent.name:14s} voxels only (no event timestamps): not scored")
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
        print("per-run results ->", a.csv)
    keys = [k for k in ("IG-NLL", "Chamfer", "Same-ts", "Fano", "ISI-spike") if k in df]
    if len(df):
        df["count_ratio"] = df["n_sim"] / df["n_real"]
        summ = df.groupby(["dataset", "method"])[keys + ["count_ratio"]].mean()
        summ["runs"] = df.groupby(["dataset", "method"]).size()
        print("\nmean over runs (lower is better for IG-NLL, Chamfer, Fano; Same-ts/ISI-spike: compare with real camera)")
        print(summ.to_string(float_format=lambda v: f"{v:.5g}"))
        if a.csv:
            summ.to_csv(Path(a.csv).with_name(Path(a.csv).stem + "_summary.csv"))
    return df


def cmd_score(a):
    runs = [r for pat in a.runs for r in sorted(glob.glob(pat))]
    score_runs(runs, a)


# ----------------------------------------------------------------------------- batch
def cmd_batch(a):
    bench_dir = Path(a.evtween).resolve() / "benchmarks" / "event_generation"
    if not (bench_dir / "benchmark.py").exists():
        raise SystemExit(f"{bench_dir}/benchmark.py not found: --evtween must be a checkout of the branch "
                         "benchmark/event-generation. Without touching your current checkout:\n"
                         f"  git -C {a.evtween} fetch origin benchmark/event-generation\n"
                         f"  git -C {a.evtween} worktree add {Path(a.evtween).resolve()}-bench origin/benchmark/event-generation\n"
                         f"then pass --evtween {Path(a.evtween).resolve()}-bench")
    sys.path.insert(0, str(bench_dir))
    import benchmark as B  # Evtween's own clip preparation and model runner

    seqs = [l.strip() for l in open(a.sequences) if l.strip() and not l.startswith("#")] \
        if os.path.isfile(a.sequences) else [str(p) for p in B.discover(a.sequences)]
    pyth = json.loads((bench_dir / "interpreters.json").read_text()) if (bench_dir / "interpreters.json").exists() else {}
    if a.evtween_python:
        pyth["evtween"] = a.evtween_python
    params = json.loads(a.params) if a.params else {}
    default_params = {"v2e": {"seed": 123, "pos_thres": 0.2, "neg_thres": 0.2}, "iebcs": {"seed": 123},
                      "dvs_voltmeter": {"seed": 123}, "v2ce": {"seed": 123, "batch_size": 1, "stage2_batch_size": 1},
                      "evis_core": {"seed": 123, "threshold": 0.15, "device": "cpu"}}
    methods = [m for m in a.methods.split(",") if m]
    out_root = Path(a.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for s in seqs:
        name = f"{dataset_of(s)}_{Path(s).name}_{a.start}_{a.count}_{a.stride}"
        rdir = out_root / name
        if not (rdir / "input.json").exists():
            B.prepare(s, rdir, a.start, a.count, a.stride, tuple(a.crop))
        for m in methods:
            if (rdir / m / "events.npz").exists() or (rdir / m / "voxels.npz").exists():
                continue
            if m == "tides":
                ta = argparse.Namespace(**{**vars(a), "run": str(rdir), "name": "tides", "overwrite": True})
                try:
                    cmd_tides(ta)
                except Exception as e:  # noqa: BLE001
                    print(f"[fail] tides on {name}: {e}")
                continue
            if m not in pyth:
                print(f"[skip] {m}: no interpreter (run Evtween's setup.sh, or --evtween_python for evtween)")
                continue
            p = {**default_params.get(m, {}), **params.get(m, {})}
            if m == "evtween" and "checkpoint" not in p:
                print("[skip] evtween: pass --params '{\"evtween\": {\"checkpoint\": \"...\", \"dataset\": \"eds\"}}'")
                continue
            if m == "evtween":
                p.setdefault("dataset", dataset_of(s))
            import shutil

            shutil.rmtree(rdir / m, ignore_errors=True)
            try:
                B.run_model(m, rdir, pyth[m], p)
            except Exception as e:  # noqa: BLE001
                print(f"[fail] {m} on {name}: {str(e)[-500:]}")
        runs.append(str(rdir))
    a.csv = a.csv or str(out_root / "results.csv")
    score_runs(runs, a)


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tides_args(p):
        p.add_argument("--config", action="append", default=None, help="TIDES config(s); default configs/tides_adaptive.yaml")
        p.add_argument("--set", action="append", default=[], help="key=value overrides, e.g. comparator.sigma_C=0.03")
        p.add_argument("--C", type=float, default=None, help="contrast threshold (default: from the config)")
        p.add_argument("--calibrate_count", action="store_true",
                       help="pick C so the event count matches ground truth (uses real events; off by default)")
        p.add_argument("--gamma", type=float, default=None, help="default: Evtween's per-dataset gamma (BS-ERGB 2.2)")
        p.add_argument("--fit_frames", type=int, default=150, help="RGB frames around the clip used to fit 4DGS")
        p.add_argument("--easi3r_dir", default=None, help="Easi3R poses for dynamic scenes instead of COLMAP")
        p.add_argument("--device", default=None)

    def score_args(p):
        p.add_argument("--metrics_config", default="configs/metrics.yaml")
        p.add_argument("--trim_frames", type=float, default=1.0,
                       help="frame intervals dropped at each end (boundary effects, see Evtween README)")
        p.add_argument("--csv", default=None)

    p = sub.add_parser("tides", help="run TIDES on one benchmark run")
    p.add_argument("--run", required=True)
    p.add_argument("--name", default="tides")
    p.add_argument("--overwrite", action="store_true")
    tides_args(p)

    p = sub.add_parser("score", help="fidelity metrics for every method in the given runs")
    p.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    score_args(p)

    p = sub.add_parser("batch", help="cut clips, run all methods + TIDES, score")
    p.add_argument("--evtween", required=True, help="Evtween checkout on branch benchmark/event-generation")
    p.add_argument("--sequences", required=True, help="text file with one sequence dir per line, or a dataset root")
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default="v2e,iebcs,dvs_voltmeter,v2ce,evis_core,evtween,tides")
    p.add_argument("--params", default=None, help='JSON per method, e.g. {"evtween": {"checkpoint": "..."}}')
    p.add_argument("--evtween_python", default=None, help="python of your Evtween training env")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=17)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--crop", type=int, nargs=2, default=(260, 346), metavar=("H", "W"))
    tides_args(p)
    score_args(p)

    a = ap.parse_args()
    if getattr(a, "config", "x") is None:
        a.config = ["configs/tides_adaptive.yaml"]
    {"tides": cmd_tides, "score": cmd_score, "batch": cmd_batch}[a.cmd](a)


if __name__ == "__main__":
    main()
