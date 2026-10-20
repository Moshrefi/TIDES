"""EWA projection of 3D Gaussians to screen space, with time tangents.

Computes per-Gaussian (u_k, u_dot_k, Q_k, Q_dot_k, c_k, c_dot_k, sigma_k,
sigma_dot_k) as described in Sec. 3.2 of the paper. The projection follows
3DGS (Kerbl et al. 2023): local affine approximation J of the perspective map,
screen covariance J W Sigma W^T J^T + 0.3 I, and the same frustum clamping.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import Camera
from .tangent_ops import covariance3d_tangent, sh_to_color_tangent


@dataclass
class GaussianState:
    """Activated Gaussian parameters at one time instant, plus time derivatives."""

    means: torch.Tensor  # (P,3)
    scales: torch.Tensor  # (P,3) activated (exp)
    rotations: torch.Tensor  # (P,4) normalized quaternions
    opacities: torch.Tensor  # (P,) activated (sigmoid)
    sh: torch.Tensor  # (P,K,3)
    means_dot: torch.Tensor
    scales_dot: torch.Tensor
    rotations_dot: torch.Tensor
    opacities_dot: torch.Tensor
    sh_dot: torch.Tensor
    sh_degree: int = 3

    @property
    def num(self) -> int:
        return self.means.shape[0]

    @staticmethod
    def static(means, scales, rotations, opacities, sh, sh_degree=3):
        z = torch.zeros_like
        return GaussianState(means, scales, rotations, opacities, sh,
                             z(means), z(scales), z(rotations), z(opacities), z(sh), sh_degree)


@dataclass
class ScreenGaussians:
    """Screen-space primitives used by the rasterizer (only visible Gaussians)."""

    uv: torch.Tensor  # (N,2) projected mean in pixels
    uv_dot: torch.Tensor  # (N,2)
    conic: torch.Tensor  # (N,2,2) precision Q
    conic_dot: torch.Tensor  # (N,2,2)
    opacity: torch.Tensor  # (N,)
    opacity_dot: torch.Tensor  # (N,)
    color: torch.Tensor  # (N,3)
    color_dot: torch.Tensor  # (N,3)
    depth: torch.Tensor  # (N,)
    radius: torch.Tensor  # (N,) pixel radius (3 sigma), for tile binning
    index: torch.Tensor  # (N,) index into the original Gaussian set


def project_gaussians(g: GaussianState, cam: Camera, scale_modifier: float = 1.0,
                      near: float = 0.2, dilation: float = 0.3) -> ScreenGaussians:
    R, t, Rd, td = cam.R, cam.t, cam.R_dot, cam.t_dot
    mu, mud = g.means, g.means_dot

    # Camera-space position and its derivative: p = R mu + t.
    p = mu @ R.T + t
    pd = mu @ Rd.T + mud @ R.T + td
    x, y, z = p.unbind(-1)
    xd, yd, zd = pd.unbind(-1)

    visible = z > near
    zc = torch.where(visible, z, torch.ones_like(z))

    # Projected mean (pixels): u = fx x/z + cx.
    u = cam.fx * x / zc + cam.cx
    v = cam.fy * y / zc + cam.cy
    ud = cam.fx * (xd * zc - x * zd) / zc**2
    vd = cam.fy * (yd * zc - y * zd) / zc**2

    # Jacobian of the perspective map with 3DGS frustum clamping.
    limx, limy = 1.3 * cam.tan_fovx, 1.3 * cam.tan_fovy
    rx, ry = x / zc, y / zc
    clx, cly = rx.clamp(-limx, limx), ry.clamp(-limy, limy)
    tx, ty = clx * zc, cly * zc
    # d/dt of the clamped coordinate: free -> x_dot; clamped -> lim*sign * z_dot.
    txd = torch.where(rx.abs() <= limx, xd, clx * zd)
    tyd = torch.where(ry.abs() <= limy, yd, cly * zd)

    zeros = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([cam.fx / zc, zeros, -cam.fx * tx / zc**2], -1),
        torch.stack([zeros, cam.fy / zc, -cam.fy * ty / zc**2], -1),
    ], -2)  # (P,2,3)
    Jd = torch.stack([
        torch.stack([-cam.fx * zd / zc**2, zeros, -cam.fx * (txd / zc**2 - 2 * tx * zd / zc**3)], -1),
        torch.stack([zeros, -cam.fy * zd / zc**2, -cam.fy * (tyd / zc**2 - 2 * ty * zd / zc**3)], -1),
    ], -2)

    Sigma, Sigma_d = covariance3d_tangent(g.scales, g.scales_dot, g.rotations, g.rotations_dot, scale_modifier)

    A = J @ R  # (P,2,3)
    Ad = Jd @ R + J @ Rd
    S2 = A @ Sigma @ A.transpose(-1, -2)
    S2d = Ad @ Sigma @ A.transpose(-1, -2) + A @ Sigma_d @ A.transpose(-1, -2) + A @ Sigma @ Ad.transpose(-1, -2)
    S2 = S2 + dilation * torch.eye(2, dtype=S2.dtype, device=S2.device)

    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] * S2[:, 1, 0]
    visible = visible & (det > 0)
    det_safe = torch.where(det > 0, det, torch.ones_like(det))
    Q = torch.stack([
        torch.stack([S2[:, 1, 1], -S2[:, 0, 1]], -1),
        torch.stack([-S2[:, 1, 0], S2[:, 0, 0]], -1),
    ], -2) / det_safe[:, None, None]
    Qd = -Q @ S2d @ Q  # d(S^-1) = -S^-1 dS S^-1

    mid = 0.5 * (S2[:, 0, 0] + S2[:, 1, 1])
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))

    uv = torch.stack([u, v], -1)
    visible = visible & (radius > 0)
    # Frustum cull: drop Gaussians whose 3-sigma box misses the image.
    visible = visible & (u + radius > 0) & (u - radius < cam.width) & (v + radius > 0) & (v - radius < cam.height)

    # View-dependent colour; derivative through view direction and SH coefficients.
    C, Cd = cam.center, cam.center_dot
    color, color_d = sh_to_color_tangent(g.sh, g.sh_dot, mu - C, mud - Cd, g.sh_degree)

    idx = torch.nonzero(visible).squeeze(-1)
    return ScreenGaussians(
        uv=uv[idx], uv_dot=torch.stack([ud, vd], -1)[idx], conic=Q[idx], conic_dot=Qd[idx],
        opacity=g.opacities[idx], opacity_dot=g.opacities_dot[idx], color=color[idx], color_dot=color_d[idx],
        depth=z[idx], radius=radius[idx], index=idx,
    )
