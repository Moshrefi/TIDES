"""Residual-state comparator with continuous-time threshold crossings (Sec. 3.3).

Each pixel keeps a reference S_ref. Within a window (t0, t0+dt] the driving
signal is modelled locally, either linearly (Eq. 16) or, when the render at the
end of the window is available, quadratically from (S0, S0_dot, S1). Crossing
times are the earliest roots of S(t0+delta) = S_ref +/- C (Eq. 17); after each
event the reference moves by one threshold and the search repeats on the rest
of the window, up to n_max events per pixel per step (multi-crossing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ComparatorConfig:
    C_pos: float = 0.2
    C_neg: float = 0.2
    sigma_C: float = 0.0  # per-pixel threshold jitter (std), sampled once
    C_min: float = 0.01
    leak_rate: float = 0.0  # reference drift lambda_leak (log units / s)
    n_max: int = 8  # max events per pixel per step
    local_model: str = "quadratic"  # "linear" | "quadratic"
    residual: str = "emit"  # what to do if |S0 - S_ref| >= C at step start: "emit" | "absorb"
    seed: int = 0


@dataclass
class EventBuffer:
    t: list = field(default_factory=list)
    idx: list = field(default_factory=list)
    p: list = field(default_factory=list)

    def add(self, t, idx, p):
        if t.numel():
            self.t.append(t.detach().to("cpu", torch.float64))
            self.idx.append(idx.detach().to("cpu", torch.int64))
            self.p.append(p.detach().to("cpu", torch.int8))

    def cat(self):
        if not self.t:
            return (torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int8))
        return torch.cat(self.t), torch.cat(self.idx), torch.cat(self.p)


def _earliest_root(S0, S0d, a, level, lo, hi, quadratic: bool):
    """Smallest root delta of a d^2 + S0d d + (S0 - level) = 0 with lo < delta <= hi; inf if none."""
    inf = torch.full_like(S0, float("inf"))
    c = S0 - level
    # Linear roots (also used where the quadratic coefficient vanishes).
    lin = torch.where(S0d != 0, -c / torch.where(S0d != 0, S0d, torch.ones_like(S0d)), inf)
    lin = torch.where((lin > lo) & (lin <= hi), lin, inf)
    if not quadratic:
        return lin
    small_a = a.abs() < 1e-12
    disc = S0d * S0d - 4 * a * c
    ok = (disc >= 0) & ~small_a
    sq = torch.sqrt(disc.clamp_min(0))
    # Numerically stable pair of roots.
    qq = -0.5 * (S0d + torch.where(S0d >= 0, sq, -sq))
    a_safe = torch.where(small_a, torch.ones_like(a), a)
    q_safe = torch.where(qq == 0, torch.full_like(qq, 1e-30), qq)
    r1 = qq / a_safe
    r2 = c / q_safe
    r1 = torch.where(ok & (r1 > lo) & (r1 <= hi), r1, inf)
    r2 = torch.where(ok & (r2 > lo) & (r2 <= hi), r2, inf)
    quad = torch.minimum(r1, r2)
    return torch.where(small_a, lin, quad)


class ResidualComparator:
    def __init__(self, num_pixels: int, cfg: ComparatorConfig, device="cpu", dtype=torch.float32):
        self.cfg = cfg
        self.n = num_pixels
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        Cp = cfg.C_pos + cfg.sigma_C * torch.randn(num_pixels, generator=gen)
        Cn = cfg.C_neg + cfg.sigma_C * torch.randn(num_pixels, generator=gen)
        self.C_pos = Cp.clamp_min(cfg.C_min).to(device, dtype)
        self.C_neg = Cn.clamp_min(cfg.C_min).to(device, dtype)
        self.S_ref = None
        self.suppressed = 0
        self.buffer = EventBuffer()

    def reset(self, S0: torch.Tensor):
        self.S_ref = S0.reshape(-1).clone()

    def step(self, t0: float, dt: float, S0, S0_dot, S1=None, emit_mask=None, pix=None):
        """Advance pixels `pix` (flat indices, default all) over (t0, t0+dt]."""
        cfg = self.cfg
        S0 = S0.reshape(-1)
        S0d = S0_dot.reshape(-1)
        if pix is None:
            pix = torch.arange(self.n, device=S0.device)
        Sref = self.S_ref[pix]
        Cp, Cn = self.C_pos[pix], self.C_neg[pix]
        emit = torch.ones_like(S0, dtype=torch.bool) if emit_mask is None else emit_mask.reshape(-1)

        if cfg.leak_rate:
            Sref = Sref - cfg.leak_rate * dt

        # Residual: the signal already passed a threshold before this window starts
        # (local-model mismatch at the previous step boundary).
        d = S0 - Sref
        k_pos = torch.floor(d / Cp).clamp(0, cfg.n_max)
        k_neg = torch.floor(-d / Cn).clamp(0, cfg.n_max)
        if cfg.residual == "emit":
            for k, pol, C in ((k_pos, 1, Cp), (k_neg, -1, Cn)):
                kk = k.long()
                sel = kk > 0
                if bool(sel.any()):
                    rep = torch.repeat_interleave(torch.nonzero(sel).squeeze(-1), kk[sel])
                    tt = torch.full((len(rep),), t0, dtype=torch.float64, device=pix.device)
                    self._emit(tt, pix[rep], torch.full_like(rep, pol), emit[rep])
        Sref = Sref + k_pos * Cp - k_neg * Cn

        quadratic = cfg.local_model == "quadratic" and S1 is not None
        if quadratic:
            S1 = S1.reshape(-1)
            a = (S1 - S0 - S0d * dt) / (dt * dt)
        else:
            a = torch.zeros_like(S0)

        lo = torch.zeros_like(S0)
        active = torch.ones_like(S0, dtype=torch.bool)
        for _ in range(cfg.n_max):
            rp = _earliest_root(S0, S0d, a, Sref + Cp, lo, dt, quadratic)
            rn = _earliest_root(S0, S0d, a, Sref - Cn, lo, dt, quadratic)
            pos = rp <= rn
            r = torch.where(pos, rp, rn)
            fire = active & torch.isfinite(r)
            if not bool(fire.any()):
                break
            sel = torch.nonzero(fire).squeeze(-1)
            pol = torch.where(pos[sel], 1, -1)
            tt = t0 + r[sel].to(torch.float64)
            self._emit(tt, pix[sel], pol, emit[sel])
            Sref = torch.where(fire & pos, Sref + Cp, Sref)
            Sref = torch.where(fire & ~pos, Sref - Cn, Sref)
            lo = torch.where(fire, r, lo)
            active = fire

        self.S_ref[pix] = Sref

    def _emit(self, t, idx, pol, emit):
        # Suppressed crossings (mixed-visibility mask) still move S_ref but are not output.
        self.suppressed += int((~emit).sum())
        self.buffer.add(t[emit], idx[emit], pol[emit].to(torch.int8))

    def events(self):
        return self.buffer.cat()
