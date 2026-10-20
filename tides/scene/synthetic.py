"""Small procedural dynamic scenes for tests and a no-download demo."""
from __future__ import annotations

import math

import torch

from ..render.camera import CameraTrajectory, intrinsics_from_fov, look_at
from .deformation import AnalyticScene

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return ((rgb - 0.5) / SH_C0).unsqueeze(1)  # (P,1,3)


def occlusion_scene(n_back: int = 400, n_front: int = 60, seed: int = 0, dtype=torch.float64,
                    speed: float = 1.5) -> AnalyticScene:
    """A textured static back wall plus a blob of Gaussians sweeping across in front of it.

    The moving front layer produces occlusion / disocclusion edges, which is
    the regime where TIDES' visibility-consistent derivatives matter.
    """
    g = torch.Generator().manual_seed(seed)
    # Back wall at z = 4 (world), textured with random colours.
    xb = (torch.rand(n_back, generator=g, dtype=dtype) - 0.5) * 4.0
    yb = (torch.rand(n_back, generator=g, dtype=dtype) - 0.5) * 3.0
    zb = 4.0 + 0.05 * torch.randn(n_back, generator=g, dtype=dtype)
    back = torch.stack([xb, yb, zb], -1)
    # Front blob at z = 2, moving along +x.
    cf = torch.tensor([-1.2, 0.0, 2.0], dtype=dtype)
    front = cf + 0.25 * torch.randn(n_front, 3, generator=g, dtype=dtype) * torch.tensor([1.0, 1.0, 0.2], dtype=dtype)
    xyz = torch.cat([back, front])
    P = xyz.shape[0]
    scales = torch.cat([
        torch.full((n_back, 3), math.log(0.12), dtype=dtype),
        torch.full((n_front, 3), math.log(0.08), dtype=dtype),
    ]) + 0.2 * torch.randn(P, 3, generator=g, dtype=dtype)
    rots = torch.randn(P, 4, generator=g, dtype=dtype)
    rots = rots / rots.norm(dim=-1, keepdim=True)
    opac = torch.cat([torch.full((n_back,), 2.0, dtype=dtype), torch.full((n_front,), 1.0, dtype=dtype)])
    rgb = torch.rand(P, 3, generator=g, dtype=dtype)
    rgb[n_back:] = torch.tensor([0.95, 0.85, 0.2], dtype=dtype) + 0.05 * torch.rand(n_front, 3, generator=g, dtype=dtype)
    vel = torch.zeros(P, 3, dtype=dtype)
    vel[n_back:, 0] = speed
    spin = torch.zeros(P, dtype=dtype)
    spin[n_back:] = 3.0
    return AnalyticScene(xyz, scales, rots, opac, rgb_to_sh0(rgb), velocity=vel, spin=spin, sh_degree=0)


def static_trajectory(width=64, height=48, fov=60.0, t0=0.0, t1=1.0, n=11, pan: float = 0.0,
                      dtype=torch.float64) -> CameraTrajectory:
    """Camera at the origin looking down +z, optionally panning sideways."""
    times = torch.linspace(t0, t1, n, dtype=dtype)
    Rs, ts = [], []
    for k in range(n):
        s = (times[k] - t0) / max(t1 - t0, 1e-9)
        eye = torch.tensor([pan * s, 0.0, 0.0], dtype=dtype)
        R, t = look_at(eye, eye + torch.tensor([0.0, 0.0, 1.0], dtype=dtype))
        Rs.append(R)
        ts.append(t)
    fx, fy, cx, cy = intrinsics_from_fov(width, height, fov)
    return CameraTrajectory(times, torch.stack(Rs), torch.stack(ts), fx, fy, cx, cy, width, height)
