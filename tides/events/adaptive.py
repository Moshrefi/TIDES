"""Risk-guided adaptive time stepping (Sec. 3.4, Eq. 18).

r(u,t) = clip( w_b (1 - norm(b)) + w_c norm(c) + w_alpha norm(|alpha_dot|) )
dt <- dt * kappa(r),  kappa(r) in (0, 1]

The paper does not give norm(), kappa() or the weights; the choices below are
ours and are all configurable (see docs/assumptions.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..render.rasterizer import TangentRender, pixel_tile_ids


@dataclass
class RiskConfig:
    w_b: float = 0.2
    w_c: float = 0.3
    w_alpha: float = 0.5
    c_ref: float = 32.0  # contributor count mapped to norm(c) = 1
    alpha_ref: float = 0.25  # |alpha_dot| * dt that maps to norm = 1
    kappa_min: float = 0.125  # smallest step shrink (=> at most 8 sub-steps)
    b_min: float = 0.05  # mixed-visibility mask: no events where b < b_min
    tile_stat: str = "max"  # how pixel risks are pooled per tile: max | mean | p95
    use_b: bool = True
    use_c: bool = True
    use_alpha: bool = True


def risk_map(r: TangentRender, dt: float, cfg: RiskConfig) -> torch.Tensor:
    nb = r.b.clamp(0, 1)
    nc = (r.c / cfg.c_ref).clamp(0, 1)
    na = (r.alpha_dot.abs() * dt / cfg.alpha_ref).clamp(0, 1)
    risk = torch.zeros_like(nb)
    if cfg.use_b:
        risk = risk + cfg.w_b * (1 - nb)
    if cfg.use_c:
        risk = risk + cfg.w_c * nc
    if cfg.use_alpha:
        risk = risk + cfg.w_alpha * na
    return risk.clamp(0, 1)


def kappa(r: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    """Linear shrink from 1 (no risk) to kappa_min (max risk)."""
    return 1.0 - (1.0 - cfg.kappa_min) * r


def tile_substeps(risk: torch.Tensor, valid: torch.Tensor, cfg: RiskConfig, tile: int, max_sub: int) -> torch.Tensor:
    """Number of sub-steps per tile, n = ceil(1 / kappa(R_tile)). Pixels outside `valid` are ignored."""
    H, W = risk.shape
    tid = pixel_tile_ids(W, H, tile, risk.device).reshape(-1)
    n_tiles = int(tid.max()) + 1
    rv = torch.where(valid, risk, torch.zeros_like(risk)).reshape(-1)
    if cfg.tile_stat == "mean":
        s = torch.zeros(n_tiles, device=risk.device, dtype=risk.dtype).index_add_(0, tid, rv)
        n = torch.zeros(n_tiles, device=risk.device, dtype=risk.dtype).index_add_(0, tid, valid.reshape(-1).to(risk.dtype))
        R = s / n.clamp_min(1)
    else:
        R = torch.zeros(n_tiles, device=risk.device, dtype=risk.dtype).scatter_reduce_(0, tid, rv, reduce="amax")
        if cfg.tile_stat == "p95":
            R = 0.95 * R  # cheap proxy; exact percentiles per tile are rarely worth it
    n_sub = torch.ceil(1.0 / kappa(R, cfg) - 1e-6).clamp(1, max_sub).long()
    return n_sub


def max_substeps(cfg: RiskConfig) -> int:
    return int(math.ceil(1.0 / cfg.kappa_min - 1e-6))
