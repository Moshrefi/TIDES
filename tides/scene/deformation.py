"""Time-conditioned Gaussian scenes and their exact time derivatives (Sec. 3.2, Eq. 6-8).

A scene maps a scalar time t to raw Gaussian parameters
    Theta(t) = (mu, s_raw, q_raw, sigma_raw, a)
The time derivative Theta_dot(t) is obtained with ONE forward-mode Jacobian-vector
product along the time axis (torch.func.jvp), never by finite differences. The
element-wise activations are then differentiated analytically.

Three scene types are provided:
  * HexPlaneDeformation: a self-contained re-implementation of the 4DGS
    (Wu et al., CVPR 2024) HexPlane + MLP deformation field, with an
    AD-friendly bilinear sampler. Its parameter names mirror the official
    code so official `deformation.pth` checkpoints can be loaded.
  * Upstream4DGS: wraps the official 4DGaussians GaussianModel directly.
  * AnalyticScene: closed-form motion, used for tests and demos.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from ..render.projection import GaussianState
from ..render.tangent_ops import exp_tangent, normalize_tangent, sigmoid_tangent


def time_jvp(fn: Callable[[torch.Tensor], tuple], t: float, dtype=torch.float32, device="cpu", rate: float = 1.0):
    """Evaluate fn(t) and d fn / dt with a single forward-mode JVP.

    `rate` is d(scene time)/d(wall time): the tangent fed to the JVP, so the
    returned derivatives are per second of simulated time.
    """
    tt = torch.tensor(float(t), dtype=dtype, device=device)
    tangent = torch.full_like(tt, float(rate))
    return torch.func.jvp(fn, (tt,), (tangent,))


class DynamicScene(nn.Module):
    """Base class: subclasses implement raw(t) -> (mu, s_raw, q_raw, sigma_raw, sh)."""

    sh_degree: int = 3
    # Scene time = (t_seconds - time_offset) * time_scale. 4DGS models are
    # trained on normalised time in [0,1]; set these from the sequence timestamps.
    time_offset: float = 0.0
    time_scale: float = 1.0

    def set_time_mapping(self, t_first: float, t_last: float):
        self.time_offset = float(t_first)
        self.time_scale = 1.0 / max(float(t_last) - float(t_first), 1e-12)
        return self

    def raw(self, t: torch.Tensor):  # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def state(self, t: float) -> GaussianState:
        p = next(iter(self.parameters()), None)
        buf = next(iter(self.buffers()), None)
        ref = p if p is not None else buf
        dtype = ref.dtype if ref is not None else torch.float32
        device = ref.device if ref is not None else "cpu"
        (mu, s_raw, q_raw, o_raw, sh), (mud, s_rawd, q_rawd, o_rawd, shd) = time_jvp(
            self.raw, (t - self.time_offset) * self.time_scale, dtype, device, rate=self.time_scale)
        s, sd = exp_tangent(s_raw, s_rawd)  # Eq. 7
        o, od = sigmoid_tangent(o_raw.squeeze(-1), o_rawd.squeeze(-1))  # Eq. 8
        q, qd = normalize_tangent(q_raw, q_rawd)  # quaternion normalisation, propagated consistently
        return GaussianState(mu, s, q, o, sh, mud, sd, qd, od, shd, self.sh_degree)


# ---------------------------------------------------------------------------
# AD-friendly bilinear sampling (replaces F.grid_sample, align_corners=True,
# padding_mode='border'). Built from gathers and arithmetic only, so forward-mode
# AD works on every PyTorch version.
# ---------------------------------------------------------------------------
def bilinear_sample(plane: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """plane: (1,C,H,W); coords: (N,2) in [-1,1], coords[:,0] -> W axis. Returns (N,C)."""
    _, C, H, W = plane.shape
    x = ((coords[:, 0] + 1) * 0.5 * (W - 1)).clamp(0, W - 1)
    y = ((coords[:, 1] + 1) * 0.5 * (H - 1)).clamp(0, H - 1)
    x0 = x.detach().floor().clamp(0, max(W - 2, 0)).long()
    y0 = y.detach().floor().clamp(0, max(H - 2, 0)).long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = (x - x0.to(x.dtype)).unsqueeze(-1)
    wy = (y - y0.to(y.dtype)).unsqueeze(-1)
    flat = plane[0].reshape(C, H * W).T  # (H*W, C)
    f00 = flat[y0 * W + x0]
    f01 = flat[y0 * W + x1]
    f10 = flat[y1 * W + x0]
    f11 = flat[y1 * W + x1]
    return (f00 * (1 - wx) * (1 - wy) + f01 * wx * (1 - wy) + f10 * (1 - wx) * wy + f11 * wx * wy)


class HexPlaneField(nn.Module):
    """Multi-resolution HexPlane (K-Planes) field over (x,y,z,t), as in 4DGS."""

    def __init__(self, bounds: float = 1.6, resolution=(64, 64, 64, 25), out_dim: int = 32, multires=(1, 2)):
        super().__init__()
        aabb = torch.tensor([[bounds] * 3, [-bounds] * 3], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.combs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for r in multires:
            res = [resolution[0] * r, resolution[1] * r, resolution[2] * r, resolution[3]]
            planes = nn.ParameterList()
            for (a, b) in self.combs:
                p = torch.empty(1, out_dim, res[b], res[a])
                if 3 in (a, b):
                    nn.init.ones_(p)
                else:
                    nn.init.uniform_(p, 0.1, 0.5)
                planes.append(nn.Parameter(p))
            self.grids.append(planes)
            self.feat_dim += out_dim

    def set_aabb(self, xyz_max, xyz_min):
        self.aabb.data = torch.stack([torch.as_tensor(xyz_max), torch.as_tensor(xyz_min)]).float().to(self.aabb.device)

    def forward(self, pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        aabb = self.aabb
        pts = (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
        x = torch.cat([pts, t], dim=-1)
        feats = []
        for planes in self.grids:
            f = 1.0
            for (a, b), plane in zip(self.combs, planes):
                f = f * bilinear_sample(plane, x[:, [a, b]])
            feats.append(f)
        return torch.cat(feats, dim=-1)


class Deformation(nn.Module):
    """4DGS deformation decoder (feature_out MLP + per-attribute heads)."""

    def __init__(self, D: int = 1, W: int = 64, grid: HexPlaneField | None = None,
                 no_dx=False, no_ds=False, no_dr=False, no_do=True, no_dshs=True):
        super().__init__()
        self.grid = grid or HexPlaneField()
        layers = [nn.Linear(self.grid.feat_dim, W)]
        for _ in range(D - 1):
            layers += [nn.ReLU(), nn.Linear(W, W)]
        self.feature_out = nn.Sequential(*layers)

        def head(o):
            return nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, o))

        self.pos_deform = head(3)
        self.scales_deform = head(3)
        self.rotations_deform = head(4)
        self.opacity_deform = head(1)
        self.shs_deform = head(16 * 3)
        self.flags = dict(no_dx=no_dx, no_ds=no_ds, no_dr=no_dr, no_do=no_do, no_dshs=no_dshs)

    def forward(self, xyz, scales, rots, opacity, shs, t):
        h = self.feature_out(self.grid(xyz, t))
        f = self.flags
        pts = xyz if f["no_dx"] else xyz + self.pos_deform(h)
        sc = scales if f["no_ds"] else scales + self.scales_deform(h)
        rt = rots if f["no_dr"] else rots + self.rotations_deform(h)
        op = opacity if f["no_do"] else opacity + self.opacity_deform(h)
        sh = shs if f["no_dshs"] else shs + self.shs_deform(h).reshape(shs.shape[0], 16, 3)[:, : shs.shape[1]]
        return pts, sc, rt, op, sh


class HexPlaneDeformation(DynamicScene):
    """Canonical Gaussians + 4DGS deformation field."""

    def __init__(self, xyz, scales_raw, rots_raw, opacity_raw, sh, deformation: Deformation | None = None, sh_degree=3):
        super().__init__()
        self.xyz = nn.Parameter(xyz)
        self.scales_raw = nn.Parameter(scales_raw)
        self.rots_raw = nn.Parameter(rots_raw)
        self.opacity_raw = nn.Parameter(opacity_raw.reshape(-1, 1))
        self.sh = nn.Parameter(sh)
        self.deformation_net = deformation or Deformation()
        self.sh_degree = sh_degree

    def raw(self, t: torch.Tensor):
        tt = t.reshape(1, 1).expand(self.xyz.shape[0], 1)
        return self.deformation_net(self.xyz, self.scales_raw, self.rots_raw, self.opacity_raw, self.sh, tt)

    def load_official_deformation(self, path: str, strict: bool = False):
        """Load an official 4DGaussians `deformation.pth` (keys `deformation_net.*`)."""
        sd = torch.load(path, map_location="cpu")
        sd = {k[len("deformation_net."):]: v for k, v in sd.items() if k.startswith("deformation_net.")}
        missing, unexpected = self.deformation_net.load_state_dict(sd, strict=strict)
        return missing, unexpected


class Upstream4DGS(DynamicScene):
    """Wrap the official 4DGaussians GaussianModel (third_party/4DGaussians).

    Load with `Upstream4DGS.from_model_dir(model_dir, iteration)`. The official
    HexPlane uses F.grid_sample; to guarantee forward-mode AD support we swap
    it for `bilinear_sample` at call time.
    """

    def __init__(self, gaussians, sh_degree=3):
        super().__init__()
        self.g = gaussians
        self.sh_degree = sh_degree
        self.register_buffer("_ref", gaussians.get_xyz.detach()[:1].clone())
        # Background the scene was trained with; TIDES must composite over the same one (Eq. 4).
        self.background = tuple(getattr(gaussians, "tides_background", (1.0, 1.0, 1.0)))

    @staticmethod
    def from_model_dir(model_dir: str, iteration: int = -1, upstream_root: str = "third_party/4DGaussians",
                       config: str | None = None):
        from .upstream import load_upstream_gaussians

        return Upstream4DGS(load_upstream_gaussians(model_dir, iteration, upstream_root, config))

    def _apply(self, fn, *args, **kwargs):
        # The official GaussianModel is not an nn.Module: move / cast its tensors and
        # its deformation network too, so scene.double() and scene.to(device) work.
        super()._apply(fn, *args, **kwargs)
        g = self.g
        for name, v in list(vars(g).items()):
            if isinstance(v, nn.Module):
                v._apply(fn)
            elif isinstance(v, torch.Tensor):
                nv = fn(v.detach())
                setattr(g, name, nn.Parameter(nv, requires_grad=v.requires_grad) if isinstance(v, nn.Parameter) else nv)
        return self

    def _deform(self, t: torch.Tensor):
        g = self.g
        n = g.get_xyz.shape[0]
        tt = t.reshape(1, 1).expand(n, 1)
        return g._deformation(g._xyz, g._scaling, g._rotation, g._opacity, g.get_features, tt)

    def raw(self, t: torch.Tensor):
        from .upstream import patched_grid_sample

        with patched_grid_sample():
            mu, s, q, o, sh = self._deform(t)
        return mu, s, q, o, sh

    def raw_official(self, t: torch.Tensor):
        """Same forward pass with the official F.grid_sample (no tangent support), for checking."""
        return self._deform(t)


class AnalyticScene(DynamicScene):
    """Closed-form motion: mu(t) = mu0 + v t + a sin(w t); rotation about z with rate omega."""

    def __init__(self, xyz, scales_raw, rots_raw, opacity_raw, sh, velocity=None, amp=None, freq=1.0,
                 spin=None, opacity_rate=None, sh_degree=0):
        super().__init__()
        P = xyz.shape[0]
        z = torch.zeros_like(xyz)
        self.register_buffer("xyz", xyz)
        self.register_buffer("scales_raw", scales_raw)
        self.register_buffer("rots_raw", rots_raw)
        self.register_buffer("opacity_raw", opacity_raw.reshape(P, 1))
        self.register_buffer("sh", sh)
        self.register_buffer("velocity", velocity if velocity is not None else z)
        self.register_buffer("amp", amp if amp is not None else z)
        self.register_buffer("spin", spin if spin is not None else torch.zeros(P))
        self.register_buffer("opacity_rate", opacity_rate if opacity_rate is not None else torch.zeros(P))
        self.freq = freq
        self.sh_degree = sh_degree

    def raw(self, t: torch.Tensor):
        mu = self.xyz + self.velocity * t + self.amp * torch.sin(2 * math.pi * self.freq * t)
        half = 0.5 * self.spin * t
        dq = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
        q = _quat_mul(dq, self.rots_raw)
        o = self.opacity_raw + self.opacity_rate.unsqueeze(-1) * t
        return mu, self.scales_raw + 0 * t, q, o, self.sh + 0 * t


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)
