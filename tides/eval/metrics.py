"""Event-stream fidelity metrics (Sec. 4.3).

  * IG-NLL: fit an inverse-Gaussian to the simulator's per-pixel inter-spike
    intervals, report the NLL of the real ISIs under it.
  * Motion-scaled spatiotemporal Chamfer: e = (x, y, v t), polarity matched,
    bidirectional nearest neighbour over fixed temporal windows, with
    voxel-grid density control.
  * Same-ts, ISI spike ratio, Fano-factor error (Eq. 20).

The paper defers binning / windowing / normalisation to its supplementary
material, which we do not have. Defaults are documented in docs/assumptions.md;
absolute values are therefore not directly comparable with Tables 1-2, but
rankings between simulators under the same settings are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Events:
    t: np.ndarray  # seconds, float64
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray  # +1 / -1

    def __len__(self):
        return len(self.t)

    def window(self, t0, t1):
        m = (self.t >= t0) & (self.t < t1)
        return Events(self.t[m], self.x[m], self.y[m], self.p[m])

    @staticmethod
    def from_dict(d):
        p = np.asarray(d["p"]).astype(np.int8)
        p = np.where(p > 0, 1, -1).astype(np.int8)
        return Events(np.asarray(d["t"], dtype=np.float64), np.asarray(d["x"]).astype(np.int64),
                      np.asarray(d["y"]).astype(np.int64), p)


@dataclass
class MetricConfig:
    window: float = 0.05  # s, temporal window for Chamfer / Same-ts / Fano
    v: float = 1000.0  # px/s motion scale in the Chamfer embedding
    voxel: float = 1.0  # voxel size (px) for density control in the embedding
    normalize: bool = True  # divide Chamfer distance by image diagonal
    max_points: int = 200_000  # random subsample cap per window and polarity
    isi_min: float = 1e-6  # ISIs below this are clamped for IG fitting
    isi_spike_eps: float = 1e-5  # ISIs below this count as spikes
    isi_quantum: float | None = None  # optional clock quantum (e.g. render step)
    isi_quantum_tol: float = 0.02
    same_ts_resolution: float = 1e-9
    fano_bin_px: int = 8
    fano_bin_t: float = 0.002
    seed: int = 0


# ---------------------------------------------------------------------------
def per_pixel_isi(ev: Events, width: int) -> np.ndarray:
    if len(ev) < 2:
        return np.zeros(0)
    pix = ev.y * width + ev.x
    o = np.lexsort((ev.t, pix))
    t, pix = ev.t[o], pix[o]
    same = pix[1:] == pix[:-1]
    return np.diff(t)[same]


def fit_inverse_gaussian(isi: np.ndarray):
    mu = isi.mean()
    lam = 1.0 / np.maximum(np.mean(1.0 / isi - 1.0 / mu), 1e-300)
    return mu, lam


def inverse_gaussian_nll(isi, mu, lam):
    x = isi
    ll = 0.5 * np.log(lam) - 0.5 * np.log(2 * np.pi * x**3) - lam * (x - mu) ** 2 / (2 * mu**2 * x)
    return -np.mean(ll)


def ig_nll(real: Events, sim: Events, width: int, cfg: MetricConfig, unit: float = 1.0) -> float:
    """NLL of real ISIs under an IG fitted to simulated ISIs. `unit` rescales time (e.g. 1e-3 for ms)."""
    si = np.maximum(per_pixel_isi(sim, width), cfg.isi_min) / unit
    ri = np.maximum(per_pixel_isi(real, width), cfg.isi_min) / unit
    if len(si) < 2 or len(ri) < 1:
        return float("nan")
    mu, lam = fit_inverse_gaussian(si)
    return float(inverse_gaussian_nll(ri, mu, lam))


# ---------------------------------------------------------------------------
def _embed(ev: Events, t0: float, cfg: MetricConfig, rng):
    e = np.stack([ev.x.astype(np.float64), ev.y.astype(np.float64), cfg.v * (ev.t - t0)], 1)
    if cfg.voxel > 0 and len(e):
        _, keep = np.unique(np.floor(e / cfg.voxel).astype(np.int64), axis=0, return_index=True)
        e = e[keep]
    if len(e) > cfg.max_points:
        e = e[rng.choice(len(e), cfg.max_points, replace=False)]
    return e


def chamfer_st(real: Events, sim: Events, width: int, height: int, cfg: MetricConfig) -> float:
    rng = np.random.default_rng(cfg.seed)
    t0, t1 = min(real.t.min(), sim.t.min()), max(real.t.max(), sim.t.max())
    scores = []
    for w0 in np.arange(t0, t1, cfg.window):
        rw, sw = real.window(w0, w0 + cfg.window), sim.window(w0, w0 + cfg.window)
        for pol in (1, -1):
            a = _embed(Events(*(arr[rw.p == pol] for arr in (rw.t, rw.x, rw.y, rw.p))), w0, cfg, rng)
            b = _embed(Events(*(arr[sw.p == pol] for arr in (sw.t, sw.x, sw.y, sw.p))), w0, cfg, rng)
            if len(a) == 0 or len(b) == 0:
                continue
            da, _ = cKDTree(b).query(a, k=1)
            db, _ = cKDTree(a).query(b, k=1)
            scores.append(0.5 * (da.mean() + db.mean()))
    if not scores:
        return float("nan")
    s = float(np.mean(scores))
    return s / np.hypot(width, height) if cfg.normalize else s


# ---------------------------------------------------------------------------
def same_ts_fraction(ev: Events, cfg: MetricConfig) -> float:
    if len(ev) == 0:
        return float("nan")
    fr = []
    ws = []
    for w0 in np.arange(ev.t.min(), ev.t.max() + 1e-12, cfg.window):
        w = ev.window(w0, w0 + cfg.window)
        if len(w) == 0:
            continue
        q = np.round(w.t / cfg.same_ts_resolution).astype(np.int64)
        _, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
        fr.append(np.mean(cnt[inv] > 1))
        ws.append(len(w))
    return float(np.average(fr, weights=ws))


def isi_spike_ratio(ev: Events, width: int, cfg: MetricConfig) -> float:
    isi = per_pixel_isi(ev, width)
    if len(isi) == 0:
        return float("nan")
    spike = isi < cfg.isi_spike_eps
    if cfg.isi_quantum:
        r = isi / cfg.isi_quantum
        spike |= (np.abs(r - np.round(r)) < cfg.isi_quantum_tol) & (np.round(r) >= 1)
    return float(spike.mean())


def _fano_per_bin(ev: Events, width: int, height: int, t0: float, t1: float, cfg: MetricConfig):
    nbx = (width + cfg.fano_bin_px - 1) // cfg.fano_bin_px
    nby = (height + cfg.fano_bin_px - 1) // cfg.fano_bin_px
    n_win = max(1, int(np.ceil((t1 - t0) / cfg.window)))
    n_sub = max(2, int(round(cfg.window / cfg.fano_bin_t)))
    counts = np.zeros((n_win, nby * nbx, n_sub))
    if len(ev):
        k = np.clip(((ev.t - t0) / cfg.window).astype(np.int64), 0, n_win - 1)
        s = np.clip((((ev.t - t0) - k * cfg.window) / cfg.fano_bin_t).astype(np.int64), 0, n_sub - 1)
        b = (ev.y // cfg.fano_bin_px) * nbx + (ev.x // cfg.fano_bin_px)
        np.add.at(counts, (k, b, s), 1)
    mean = counts.mean(-1)
    var = counts.var(-1)
    return (var / np.maximum(mean, 1e-12)).reshape(-1), mean.reshape(-1)


def fano_error(real: Events, sim: Events, width: int, height: int, cfg: MetricConfig, eps: float = 1e-3) -> float:
    """Eq. 20: mean over bins of |log(F_pred / F_gt)|, over bins active in both streams."""
    t0 = min(real.t.min(), sim.t.min())
    t1 = max(real.t.max(), sim.t.max()) + 1e-9
    Fp, mp = _fano_per_bin(sim, width, height, t0, t1, cfg)
    Fg, mg = _fano_per_bin(real, width, height, t0, t1, cfg)
    m = (mp > 0) & (mg > 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(np.log((Fp[m] + eps) / (Fg[m] + eps)))))


def all_metrics(real: Events, sim: Events, width: int, height: int, cfg: MetricConfig | None = None) -> dict:
    cfg = cfg or MetricConfig()
    return {
        "IG-NLL": ig_nll(real, sim, width, cfg),
        "Chamfer": chamfer_st(real, sim, width, height, cfg),
        "Same-ts": same_ts_fraction(sim, cfg),
        "Fano": fano_error(real, sim, width, height, cfg),
        "ISI-spike": isi_spike_ratio(sim, width, cfg),
        "n_real": len(real),
        "n_sim": len(sim),
    }
