"""Primal + tangent (time-derivative) versions of the per-Gaussian activations.

Every function takes a value and its time derivative and returns both after the
operation, so time derivatives are propagated analytically (paper Sec. 3.2, Eq. 7-8).
"""
from __future__ import annotations

import torch


def exp_tangent(x: torch.Tensor, xd: torch.Tensor):
    """s = exp(s_raw), s_dot = s * s_raw_dot."""
    y = torch.exp(x)
    return y, y * xd


def sigmoid_tangent(x: torch.Tensor, xd: torch.Tensor):
    """sigma = sigmoid(raw), sigma_dot = sigma (1 - sigma) raw_dot."""
    y = torch.sigmoid(x)
    return y, y * (1.0 - y) * xd


def normalize_tangent(q: torch.Tensor, qd: torch.Tensor, eps: float = 1e-12):
    """q_hat = q / |q|; q_hat_dot = (q_dot - q_hat (q_hat . q_dot)) / |q|."""
    n = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    qh = q / n
    qhd = (qd - qh * (qh * qd).sum(-1, keepdim=True)) / n
    return qh, qhd


def quat_to_rotmat_tangent(q: torch.Tensor, qd: torch.Tensor):
    """Rotation matrix from a unit quaternion (w, x, y, z), plus its time derivative.

    The 3DGS convention is used (quaternion stored as r, x, y, z).
    """
    r, x, y, z = q.unbind(-1)
    rd, xd, yd, zd = qd.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))

    Rd = torch.stack(
        [
            -4 * (y * yd + z * zd), 2 * (xd * y + x * yd - rd * z - r * zd), 2 * (xd * z + x * zd + rd * y + r * yd),
            2 * (xd * y + x * yd + rd * z + r * zd), -4 * (x * xd + z * zd), 2 * (yd * z + y * zd - rd * x - r * xd),
            2 * (xd * z + x * zd - rd * y - r * yd), 2 * (yd * z + y * zd + rd * x + r * xd), -4 * (x * xd + y * yd),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R, Rd


def covariance3d_tangent(scale, scale_d, rot, rot_d, scale_modifier: float = 1.0):
    """Sigma = M M^T with M = R S; Sigma_dot = M_dot M^T + M M_dot^T.

    scale: (P,3) activated scales; rot: (P,4) normalized quaternion.
    """
    R, Rd = quat_to_rotmat_tangent(rot, rot_d)
    S = scale * scale_modifier
    Sd = scale_d * scale_modifier
    M = R * S.unsqueeze(-2)  # R @ diag(S)
    Md = Rd * S.unsqueeze(-2) + R * Sd.unsqueeze(-2)
    Sigma = M @ M.transpose(-1, -2)
    Sigma_d = Md @ M.transpose(-1, -2) + M @ Md.transpose(-1, -2)
    return Sigma, Sigma_d


# ---------------------------------------------------------------------------
# Spherical harmonics (same constants as 3DGS), primal + tangent.
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
]


def sh_basis(d: torch.Tensor, degree: int) -> torch.Tensor:
    """Real SH basis evaluated at unit directions d (P,3) -> (P, (degree+1)^2)."""
    x, y, z = d.unbind(-1)
    out = [torch.full_like(x, SH_C0)]
    if degree >= 1:
        out += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [
            SH_C2[0] * xy,
            SH_C2[1] * yz,
            SH_C2[2] * (2 * zz - xx - yy),
            SH_C2[3] * xz,
            SH_C2[4] * (xx - yy),
        ]
    if degree >= 3:
        out += [
            SH_C3[0] * y * (3 * xx - yy),
            SH_C3[1] * xy * z,
            SH_C3[2] * y * (4 * zz - xx - yy),
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            SH_C3[4] * x * (4 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3 * yy),
        ]
    return torch.stack(out, dim=-1)


def sh_to_color_tangent(sh, sh_d, dirs, dirs_d, degree: int):
    """Colour from SH coefficients with derivative through both the coefficients
    (a_dot) and the view direction (v_dot), as described in Sec. 3.2.

    sh: (P, K, 3). dirs: (P,3) unnormalized view direction (mu - cam_center).
    Returns (c, c_dot), each (P,3). Negative colours are clamped as in 3DGS and
    their derivative is zeroed.
    """
    K = (degree + 1) ** 2
    sh = sh[:, :K]
    sh_d = sh_d[:, :K]
    d, dd = normalize_tangent(dirs, dirs_d)
    # Forward-mode derivative of the polynomial basis wrt the direction.
    basis, basis_d = torch.func.jvp(lambda v: sh_basis(v, degree), (d,), (dd,))
    c = (basis.unsqueeze(-1) * sh).sum(1) + 0.5
    cd = (basis_d.unsqueeze(-1) * sh).sum(1) + (basis.unsqueeze(-1) * sh_d).sum(1)
    mask = c > 0
    return torch.where(mask, c, torch.zeros_like(c)), torch.where(mask, cd, torch.zeros_like(cd))
