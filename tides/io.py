"""Event I/O and readers for the ground-truth event formats of the four datasets.

All readers return dict(t [s, float64], x, y, p [+1/-1]).
Dataset layouts follow the official releases; if your copy differs, adapt the
path patterns here (they are the only dataset-specific code in the simulator).
"""
from __future__ import annotations

import glob
import os

import numpy as np


def save_events(path: str, ev: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "w") as f:
            g = f.create_group("events")
            g.create_dataset("t", data=ev["t"], compression="gzip")
            g.create_dataset("x", data=ev["x"].astype(np.uint16), compression="gzip")
            g.create_dataset("y", data=ev["y"].astype(np.uint16), compression="gzip")
            g.create_dataset("p", data=(ev["p"] > 0).astype(np.uint8), compression="gzip")
            f.attrs["width"] = ev.get("width", 0)
            f.attrs["height"] = ev.get("height", 0)
    else:
        np.savez_compressed(path, t=ev["t"], x=ev["x"], y=ev["y"], p=ev["p"],
                            width=ev.get("width", 0), height=ev.get("height", 0))


def load_events(path: str) -> dict:
    """Load events written by save_events (npz/h5), or a plain txt (t x y p per line)."""
    if path.endswith(".npz"):
        d = np.load(path)
        return {k: d[k] for k in ("t", "x", "y", "p")} | {"width": int(d.get("width", 0)), "height": int(d.get("height", 0))}
    if path.endswith(".h5") or path.endswith(".hdf5"):
        return load_h5_events(path)
    a = np.loadtxt(path)
    return dict(t=a[:, 0], x=a[:, 1].astype(np.int64), y=a[:, 2].astype(np.int64), p=np.where(a[:, 3] > 0, 1, -1))


def load_h5_events(path: str, t_unit: float | None = None, t_offset_key: str = "t_offset") -> dict:
    """Generic `events/{t,x,y,p}` HDF5 (DSEC, EDS, ours). t_unit auto-detects us vs s."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["events"]
        t = g["t"][:].astype(np.float64)
        x, y, p = g["x"][:], g["y"][:], g["p"][:]
        off = float(f[t_offset_key][()]) if t_offset_key in f else 0.0
        w = int(f.attrs.get("width", 0))
        h = int(f.attrs.get("height", 0))
    t = t + off
    if t_unit is None:
        t_unit = 1e-6 if t.max() > 1e4 else 1.0  # integer microseconds vs seconds
    p = np.where(p > 0, 1, -1).astype(np.int8)
    return dict(t=t * t_unit, x=x.astype(np.int64), y=y.astype(np.int64), p=p, width=w, height=h)


# ---------------------------------------------------------------------------
# Dataset-specific readers
# ---------------------------------------------------------------------------
def load_dsec_events(seq_dir: str) -> dict:
    """DSEC: <seq>/events/left/events.h5 (t in us, plus t_offset)."""
    return load_h5_events(os.path.join(seq_dir, "events", "left", "events.h5"))


def load_eds_events(seq_dir: str) -> dict:
    """EDS: <seq>/events.h5 with events/{x,y,t,p}, t in us."""
    return load_h5_events(os.path.join(seq_dir, "events.h5"))


def load_timelens_events(seq_dir: str) -> dict:
    """HS-ERGB / BS-ERGB: <seq>/events/*.npz with x, y, timestamp (us), polarity."""
    files = sorted(glob.glob(os.path.join(seq_dir, "events", "*.npz")))
    if not files:
        raise FileNotFoundError(f"no event npz files under {seq_dir}/events")
    ts, xs, ys, ps = [], [], [], []
    for f in files:
        d = np.load(f)
        ts.append(d["timestamp"].astype(np.float64))
        xs.append(d["x"])
        ys.append(d["y"])
        ps.append(d["polarity"])
    t = np.concatenate(ts) * 1e-6
    o = np.argsort(t, kind="stable")
    p = np.concatenate(ps)
    return dict(t=t[o], x=np.concatenate(xs)[o].astype(np.int64), y=np.concatenate(ys)[o].astype(np.int64),
                p=np.where(p[o] > 0, 1, -1).astype(np.int8))


READERS = {"dsec": load_dsec_events, "eds": load_eds_events, "hsergb": load_timelens_events,
           "bsergb": load_timelens_events}


def to_event_frame(ev: dict, width: int, height: int, t0: float, t1: float) -> np.ndarray:
    """Red/blue polarity image for visual comparison (Fig. 3 style)."""
    m = (ev["t"] >= t0) & (ev["t"] < t1)
    img = np.full((height, width, 3), 255, np.uint8)
    x, y, p = ev["x"][m], ev["y"][m], ev["p"][m]
    img[y[p > 0], x[p > 0]] = (220, 40, 40)
    img[y[p <= 0], x[p <= 0]] = (40, 70, 220)
    return img
