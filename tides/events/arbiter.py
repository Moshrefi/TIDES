"""Derivative-conditioned finite-bandwidth readout (Sec. 3.5, Eq. 19).

Tiles keep FIFO queues; a scanner releases at most K_m events from tile m per
tick. Throughput, drop probability and timestamp jitter are conditioned on a
tile activity score psi_m in [0,1] built from renderer quantities:

    K_m = floor(K0 (1 - beta_K psi_m)),  p_drop = p0 + eta psi_m,  sigma_t = sigma0 + gamma_sigma psi_m
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..render.rasterizer import TangentRender, pixel_tile_ids

try:  # optional speed-up
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (a and callable(a[0])) else a[0]


@dataclass
class ArbiterConfig:
    enabled: bool = False
    tile: int = 32
    tick: float = 1e-6  # scanner period (s)
    K0: int = 4
    beta_K: float = 0.75
    p0: float = 0.0
    eta: float = 0.05
    sigma0: float = 0.0
    gamma_sigma: float = 20e-6
    w_rate: float = 0.5
    w_alpha: float = 0.25
    w_count: float = 0.25
    rate_ref: float = 2000.0  # events / pixel / s mapped to 1
    alpha_ref: float = 50.0  # |alpha_dot| (1/s) mapped to 1
    c_ref: float = 32.0
    seed: int = 0


def tile_activity(r: TangentRender, C: float, cfg: ArbiterConfig) -> torch.Tensor:
    """psi_m in [0,1] per arbiter tile, aggregating |L_dot|/C, |alpha_dot| and c."""
    H, W = r.L.shape
    tid = pixel_tile_ids(W, H, cfg.tile, r.L.device).reshape(-1)
    n = int(tid.max()) + 1

    def tmean(x):
        s = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, tid, x.reshape(-1))
        cnt = torch.bincount(tid, minlength=n).to(x.dtype)
        return s / cnt.clamp_min(1)

    rate = tmean(r.L_dot.abs() / C) / cfg.rate_ref
    al = tmean(r.alpha_dot.abs()) / cfg.alpha_ref
    cc = tmean(r.c) / cfg.c_ref
    psi = cfg.w_rate * rate.clamp(0, 1) + cfg.w_alpha * al.clamp(0, 1) + cfg.w_count * cc.clamp(0, 1)
    return psi.clamp(0, 1)


@njit(cache=True)
def _fifo(arrival, tile, K, tick):
    """Per-tile FIFO with capacity K[j] per tick. Events must be sorted by (tile, arrival)."""
    n = arrival.shape[0]
    out = np.empty(n)
    cur_tile = -1
    cur_tick = -1
    used = 0
    for j in range(n):
        if tile[j] != cur_tile:
            cur_tile = tile[j]
            cur_tick = -1
            used = 0
        a_tick = int(np.floor(arrival[j] / tick))
        if a_tick > cur_tick:
            cur_tick = a_tick
            used = 0
        if used >= K[j]:
            cur_tick += 1
            used = 0
        used += 1
        slot = cur_tick * tick
        out[j] = arrival[j] if slot <= arrival[j] else slot
    return out


def apply_readout(t, x, y, p, width, psi_grid, interval_starts, cfg: ArbiterConfig):
    """Apply the arbiter to an ideal event stream.

    psi_grid: (n_intervals, n_tiles) activity per base simulation interval.
    Returns filtered (t, x, y, p) sorted by time.
    """
    rng = np.random.default_rng(cfg.seed)
    if len(t) == 0:
        return t, x, y, p
    n_tx = (width + cfg.tile - 1) // cfg.tile
    tile = (y // cfg.tile) * n_tx + (x // cfg.tile)
    k = np.clip(np.searchsorted(interval_starts, t, side="right") - 1, 0, len(interval_starts) - 1)
    psi = psi_grid[k, tile]
    K = np.maximum(1, np.floor(cfg.K0 * (1 - cfg.beta_K * psi))).astype(np.int64)
    p_drop = np.clip(cfg.p0 + cfg.eta * psi, 0, 1)
    sigma = cfg.sigma0 + cfg.gamma_sigma * psi

    order = np.lexsort((t, tile))
    t_o = t - t.min()
    rel = _fifo(t_o[order].astype(np.float64), tile[order].astype(np.int64), K[order], float(cfg.tick))
    t_out = np.empty_like(t, dtype=np.float64)
    t_out[order] = rel + t.min()
    t_out = t_out + sigma * rng.standard_normal(len(t))
    keep = rng.random(len(t)) >= p_drop
    t_out, x, y, p = t_out[keep], x[keep], y[keep], p[keep]
    o = np.argsort(t_out, kind="stable")
    return t_out[o], x[o], y[o], p[o]
