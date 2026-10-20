"""TIDES: continuous-time event simulation from a dynamic Gaussian scene.

Pipeline (Fig. 2 of the paper):
  1. Time-tangent scene: Theta(t), Theta_dot(t) via one forward-mode JVP.
  2. Event rasterization: (L, L_dot, alpha, alpha_dot, b, c) in one front-to-back pass.
  3. Event generation: continuous-time threshold crossings with residual updates,
     multi-crossing, and risk-guided adaptive sub-stepping per tile.
  4. Optional readout arbiter conditioned on the same renderer cues.
"""
from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .events.adaptive import RiskConfig, risk_map, tile_substeps
from .events.arbiter import ArbiterConfig, apply_readout, tile_activity
from .events.comparator import ComparatorConfig, ResidualComparator
from .events.frontend import FrontEndConfig, SensorFrontEnd
from .render.camera import CameraTrajectory
from .render.projection import project_gaussians
from .render.rasterizer import TangentRender, pixel_tile_ids, rasterize_tangent
from .scene.deformation import DynamicScene


@dataclass
class RenderConfig:
    tile: int = 16
    tau_max: float = 0.99
    alpha_min: float = 1.0 / 255.0
    T_min: float = 1e-4
    log_eps: float = 1e-3
    near: float = 0.2
    scale_modifier: float = 1.0
    tile_batch: int = 32
    gauss_chunk: int = 256
    coverage: str = "alpha"  # how b is defined: "alpha" (sum of weights) | "tau_sum"


@dataclass
class TIDESConfig:
    queries_per_frame: int = 10  # fixed query budget ("10x" in the paper)
    frame_dt: float | None = None  # camera frame interval; default: median keyframe spacing
    adaptive: bool = True
    max_sub: int = 8
    slope: str = "analytic"  # "analytic" (TIDES) | "finite_diff" (ablation)
    mask_mixed: bool = True
    background: tuple = (0.0, 0.0, 0.0)
    background_dot: tuple = (0.0, 0.0, 0.0)
    ba_rate: float = 0.0  # background-activity noise, events / pixel / s
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    render: RenderConfig = field(default_factory=RenderConfig)
    comparator: ComparatorConfig = field(default_factory=ComparatorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    frontend: FrontEndConfig = field(default_factory=FrontEndConfig)

    @staticmethod
    def from_dict(d: dict) -> "TIDESConfig":
        sub = dict(render=RenderConfig, comparator=ComparatorConfig, risk=RiskConfig,
                   arbiter=ArbiterConfig, frontend=FrontEndConfig)
        kw = {}
        for k, v in (d or {}).items():
            if k in sub:
                kw[k] = sub[k](**(v or {}))
            elif k in ("background", "background_dot"):
                kw[k] = tuple(v)
            else:
                kw[k] = v
        return TIDESConfig(**kw)

    def to_dict(self):
        return asdict(self)


class TIDES:
    def __init__(self, scene: DynamicScene, trajectory: CameraTrajectory, cfg: TIDESConfig | None = None):
        self.cfg = cfg or TIDESConfig()
        self.device = torch.device(self.cfg.device)
        self.scene = scene.to(self.device)
        self.traj = trajectory
        _, _, _, _, W, H = trajectory.intr
        self.W, self.H = W, H
        self.tile_map = pixel_tile_ids(W, H, self.cfg.render.tile, self.device).reshape(-1)
        self.frontend = SensorFrontEnd(self.cfg.frontend)
        self.stats = dict(full_renders=0, tile_renders=0, sub_steps=0)

    # ------------------------------------------------------------------
    def render(self, t: float, tiles: torch.Tensor | None = None) -> TangentRender:
        rc = self.cfg.render
        g = self.scene.state(t)
        cam = self.traj(t, dtype=g.means.dtype, device=self.device)
        sg = project_gaussians(g, cam, rc.scale_modifier, rc.near)
        bg = torch.tensor(self.cfg.background, dtype=g.means.dtype, device=self.device)
        bgd = torch.tensor(self.cfg.background_dot, dtype=g.means.dtype, device=self.device)
        if tiles is None:
            self.stats["full_renders"] += 1
        else:
            self.stats["tile_renders"] += int(len(tiles))
        return rasterize_tangent(sg, self.W, self.H, bg, bgd, tiles=tiles, tile=rc.tile, tau_max=rc.tau_max,
                                 alpha_min=rc.alpha_min, T_min=rc.T_min, log_eps=rc.log_eps,
                                 tile_batch=rc.tile_batch, gauss_chunk=rc.gauss_chunk, coverage=rc.coverage)

    def _signal(self, r: TangentRender):
        X, Xd = self.frontend.spatial(r.L, r.L_dot)
        return X.reshape(-1), Xd.reshape(-1), r.b.reshape(-1)

    def _advance(self, comp, pix, t0, dt, X0, X0d, X1, b0):
        cfg = self.cfg
        if self.cfg.frontend.lowpass_tau > 0:
            S0, S0d, S1 = self.frontend.lowpass(pix, X0, X0d, dt)
        else:
            S0, S0d, S1 = X0, X0d, X1
        if cfg.slope == "finite_diff":
            S0d = (S1 - S0) / dt
            S1 = None  # linear model through the two samples (frame-differencing)
        emit = (b0 >= cfg.risk.b_min) if cfg.mask_mixed else None
        comp.step(t0, dt, S0, S0d, S1, emit_mask=emit, pix=pix)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def simulate(self, t_start: float | None = None, t_end: float | None = None, verbose: bool = True) -> dict:
        cfg = self.cfg
        t_start = self.traj.t_start if t_start is None else t_start
        t_end = self.traj.t_end if t_end is None else t_end
        frame_dt = cfg.frame_dt
        if frame_dt is None:
            kt = self.traj.times
            frame_dt = float(torch.median(kt[1:] - kt[:-1])) if len(kt) > 1 else (t_end - t_start)
        n_steps = max(1, int(round((t_end - t_start) / frame_dt * cfg.queries_per_frame)))
        times = np.linspace(t_start, t_end, n_steps + 1)

        HW = self.W * self.H
        comp = ResidualComparator(HW, cfg.comparator, device=self.device)
        Ra = self.render(times[0])
        Xa, Xda, ba = self._signal(Ra)
        comp.reset(Xa)
        if cfg.frontend.lowpass_tau > 0:
            self.frontend.S = Xa.clone()
            comp.reset(Xa)

        psi_rows = []
        all_pix = torch.arange(HW, device=self.device)
        t_wall = _time.time()
        for i in range(n_steps):
            ta, tb = float(times[i]), float(times[i + 1])
            dt = tb - ta
            Rb = self.render(tb)
            Xb, Xdb, bb = self._signal(Rb)

            if cfg.arbiter.enabled:
                psi_rows.append(tile_activity(Ra, cfg.comparator.C_pos, cfg.arbiter).cpu().numpy())

            if cfg.adaptive:
                risk = torch.maximum(risk_map(Ra, dt, cfg.risk), risk_map(Rb, dt, cfg.risk))
                valid = ((Ra.b >= cfg.risk.b_min) | (Rb.b >= cfg.risk.b_min)) if cfg.mask_mixed else torch.ones_like(risk, dtype=torch.bool)
                n_sub = tile_substeps(risk, valid, cfg.risk, cfg.render.tile, cfg.max_sub)
            else:
                n_sub = None

            if n_sub is None or bool((n_sub == 1).all()):
                self._advance(comp, all_pix, ta, dt, Xa, Xda, Xb, ba)
            else:
                for k in torch.unique(n_sub).tolist():
                    tiles_k = torch.nonzero(n_sub == k).squeeze(-1)
                    pix = torch.nonzero(torch.isin(self.tile_map, tiles_k)).squeeze(-1)
                    if k == 1:
                        self._advance(comp, pix, ta, dt, Xa[pix], Xda[pix], Xb[pix], ba[pix])
                        continue
                    h = dt / k
                    p0, pd0, pb0 = Xa[pix], Xda[pix], ba[pix]
                    for j in range(1, k + 1):
                        tj = ta + j * h
                        if j < k:
                            Rj = self.render(tj, tiles=tiles_k)
                            Xj, Xdj, bj = self._signal(Rj)
                            Xj, Xdj, bj = Xj[pix], Xdj[pix], bj[pix]
                        else:
                            Xj, Xdj, bj = Xb[pix], Xdb[pix], bb[pix]
                        self._advance(comp, pix, tj - h, h, p0, pd0, Xj, pb0)
                        self.stats["sub_steps"] += 1
                        p0, pd0, pb0 = Xj, Xdj, bj
            Ra, Xa, Xda, ba = Rb, Xb, Xdb, bb
            if verbose and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
                n_ev = sum(len(x) for x in comp.buffer.t)
                print(f"[TIDES] step {i + 1}/{n_steps}  t={tb:.6f}s  events={n_ev}  ({_time.time() - t_wall:.1f}s)")

        t, idx, p = comp.events()
        t = t.numpy()
        idx = idx.numpy()
        p = p.numpy()
        x = (idx % self.W).astype(np.int32)
        y = (idx // self.W).astype(np.int32)

        if cfg.ba_rate > 0:
            rng = np.random.default_rng(cfg.seed)
            n = rng.poisson(cfg.ba_rate * (t_end - t_start) * HW)
            nidx = rng.integers(0, HW, n)
            t = np.concatenate([t, rng.uniform(t_start, t_end, n)])
            x = np.concatenate([x, (nidx % self.W).astype(np.int32)])
            y = np.concatenate([y, (nidx // self.W).astype(np.int32)])
            p = np.concatenate([p, rng.choice(np.array([-1, 1], dtype=np.int8), n)])

        ideal = None
        if cfg.arbiter.enabled and len(psi_rows):
            ideal = dict(t=t.copy(), x=x.copy(), y=y.copy(), p=p.copy())
            t, x, y, p = apply_readout(t, x, y, p, self.W, np.stack(psi_rows), times[:-1], cfg.arbiter)
        o = np.argsort(t, kind="stable")
        out = dict(t=t[o], x=x[o], y=y[o], p=p[o], width=self.W, height=self.H,
                   stats=dict(self.stats, suppressed=comp.suppressed, n_steps=n_steps,
                              wall_time_s=_time.time() - t_wall))
        if ideal is not None:
            out["ideal"] = ideal
        return out
