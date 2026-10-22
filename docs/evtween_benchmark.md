# TIDES in Evtween's event-generation benchmark

`scripts/bench_evtween.py` plugs TIDES into Evtween's benchmark
(`benchmark/event-generation`, `benchmarks/event_generation/`) without changing
Evtween. It also adds what that benchmark does not have yet: fidelity metrics
against the real events (the paper's Tables 1-2) for every method.

## What a run looks like

Evtween's `prepare()` cuts one clip per run, and every method gets exactly that clip:

```
RUN/input.json          source sequence, frame indices, absolute_t0, duration, crop [x0, y0, w, h]
RUN/input/frames/*.png  centre crop (default 260x346) of the selected frames
RUN/ground_truth.npz    real events, same crop, t relative to the first frame
RUN/<method>/events.npz t (s), x, y (crop pixels), p (+-1)   <- v2e, iebcs, dvs_voltmeter, v2ce, evis_core, evtween
RUN/tides/events.npz    written by this script, same format
```

## Commands

Run everything in the `tides` env on a GPU node (`module load cuda/12.8.1`).

```bash
# 1. Whole pipeline for a list of sequences (one directory per line, or a dataset root)
python scripts/bench_evtween.py batch \
    --evtween /u/marmary/airlab_v2e/evtween \
    --sequences seqs_eds.txt --out bench_runs/eds --count 17 \
    --methods v2e,iebcs,dvs_voltmeter,v2ce,evis_core,evtween,tides \
    --evtween_python /path/to/evtween/env/bin/python \
    --params '{"evtween": {"checkpoint": "/path/to/student/last.pt"}}'

# 2. Or add TIDES to a run you already made in compare_extended.ipynb
python scripts/bench_evtween.py tides --run /u/marmary/airlab_v2e/evtween/benchmarks/event_generation/runs/<id>

# 3. Score any set of runs
python scripts/bench_evtween.py score --runs "bench_runs/eds/*" --csv bench_runs/eds/results.csv
```

`batch` uses Evtween's own `prepare()` and `run_model()` and the interpreters
that Evtween's `setup.sh` wrote to `interpreters.json`. Methods without an
interpreter are skipped with a message. Finished methods are not run again, so
an interrupted batch can be restarted with the same command.

`score` writes one row per (run, method) and a per-dataset mean
(`results_summary.csv`). Lower is better for IG-NLL, Chamfer and Fano. Same-ts
and ISI-spike describe the timing statistics and are best read next to the
`(real camera)` row. `count_ratio` = simulated / real events. The first and last
frame interval of each clip are dropped (`--trim_frames 1`) because of the
boundary effects described in Evtween's README. V2V writes voxels without event
timestamps, so it is not scored.

## How TIDES is run

* **Scene fit.** TIDES needs a 4DGS scene. It is fitted to the RGB frames
  *around* the clip (`--fit_frames 150`, default), never to events. The fit is
  cached per sequence window (`scenes/bench/`, `outputs/4dgs/bench_*`), so
  several clips from the same window share one fit (~10 min on an H100).
* **Poses.** COLMAP by default, which needs a moving camera and a static scene
  (EDS). For dynamic scenes the paper uses Easi3R; pass `--easi3r_dir`.
* **Pixel grid.** EDS frames are first undistorted to a centred pinhole camera
  (4DGS needs one). TIDES events are mapped back to the dataset's pixel grid and
  then to the benchmark crop, so they line up with the other methods and the
  real events.
* **Gamma.** Events depend on linear light. As in Evtween's `config.yaml`,
  BS-ERGB frames are linearised with gamma 2.2; others use 1.0 (`--gamma`).
* **Contrast threshold.** From the TIDES config (0.2, same as the v2e default
  in the benchmark notebook). `--calibrate_count` instead picks C so the event
  count matches the real one. That uses the real events, so either use it for
  every method or for none, and report it.

## Status per dataset

| Dataset | Camera / scene | Poses | Status |
|---|---|---|---|
| EDS | moving camera, static scene | COLMAP | ready |
| HQ-EVFI | mostly static camera, moving objects | needs Easi3R | to do |
| BS-ERGB | dynamic | needs Easi3R | to do |
| CED | colour / Bayer | - | blocked by the benchmark itself (no colour protocol yet) |
