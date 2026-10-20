"""Time-dependent pinhole camera with analytic pose derivatives (Sec. 3.2).

When the ground-truth camera velocity is not available it is estimated from
neighbouring camera timestamps: rotation is SLERP-interpolated (constant angular
velocity between keyframes) and translation is linearly interpolated, so the
pose and its derivative are defined at any continuous time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """World-to-camera pose x_c = R x_w + t, and its time derivative."""

    R: torch.Tensor  # (3,3)
    t: torch.Tensor  # (3,)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R_dot: torch.Tensor | None = None
    t_dot: torch.Tensor | None = None
    znear: float = 0.01

    def __post_init__(self):
        if self.R_dot is None:
            self.R_dot = torch.zeros_like(self.R)
        if self.t_dot is None:
            self.t_dot = torch.zeros_like(self.t)

    @property
    def center(self) -> torch.Tensor:
        return -self.R.T @ self.t

    @property
    def center_dot(self) -> torch.Tensor:
        return -(self.R_dot.T @ self.t + self.R.T @ self.t_dot)

    @property
    def tan_fovx(self) -> float:
        return self.width / (2.0 * self.fx)

    @property
    def tan_fovy(self) -> float:
        return self.height / (2.0 * self.fy)

    def to(self, device):
        return Camera(
            self.R.to(device), self.t.to(device), self.fx, self.fy, self.cx, self.cy,
            self.width, self.height, self.R_dot.to(device), self.t_dot.to(device), self.znear,
        )


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------
def hat(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack(
        [torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])]
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = w.norm()
    K = hat(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    if theta < 1e-8:
        return I + K
    return I + (torch.sin(theta) / theta) * K + ((1 - torch.cos(theta)) / theta**2) * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    cos = ((torch.trace(R) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.arccos(cos)
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * v
    return theta / (2 * torch.sin(theta)) * v


class CameraTrajectory:
    """Piecewise SLERP/linear trajectory through keyframe poses.

    Rs: (N,3,3) world-to-camera rotations, ts: (N,3) translations, times: (N,).
    """

    def __init__(self, times, Rs, ts, fx, fy, cx, cy, width, height, velocities=None):
        self.times = torch.as_tensor(times, dtype=torch.float64)
        self.Rs = torch.as_tensor(Rs, dtype=torch.float64)
        self.ts = torch.as_tensor(ts, dtype=torch.float64)
        self.intr = (float(fx), float(fy), float(cx), float(cy), int(width), int(height))
        # Optional ground-truth velocities: (R_dot (N,3,3), t_dot (N,3)).
        self.velocities = velocities

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def __call__(self, t: float, dtype=torch.float32, device="cpu") -> Camera:
        times = self.times
        if len(times) == 1:
            R, tr = self.Rs[0], self.ts[0]
            Rd, td = torch.zeros_like(R), torch.zeros_like(tr)
        else:
            i = int(torch.searchsorted(times, torch.tensor(t, dtype=torch.float64), right=True)) - 1
            i = max(0, min(i, len(times) - 2))
            t0, t1 = float(times[i]), float(times[i + 1])
            dt = t1 - t0
            s = (t - t0) / dt
            R0, R1 = self.Rs[i], self.Rs[i + 1]
            # Body-frame angular velocity: R(s) = R0 exp(s * log(R0^T R1)).
            w = so3_log(R0.T @ R1)
            R = R0 @ so3_exp(s * w)
            Rd = R @ hat(w / dt)
            tr = (1 - s) * self.ts[i] + s * self.ts[i + 1]
            td = (self.ts[i + 1] - self.ts[i]) / dt
            if self.velocities is not None:
                Rdots, tdots = self.velocities
                Rd = (1 - s) * Rdots[i] + s * Rdots[i + 1]
                td = (1 - s) * tdots[i] + s * tdots[i + 1]
        fx, fy, cx, cy, W, H = self.intr
        cast = lambda x: x.to(dtype=dtype, device=device)
        return Camera(cast(R), cast(tr), fx, fy, cx, cy, W, H, cast(Rd), cast(td))


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """World-to-camera (R, t) for an OpenCV-style camera (x right, y down, z forward)."""
    eye = torch.as_tensor(eye, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    up = torch.as_tensor(up, dtype=torch.float64)
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(up, z)
    if x.norm() < 1e-8:
        x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])  # rows = camera axes in world coords
    t = -R @ eye
    return R, t


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float):
    fx = width / (2 * math.tan(math.radians(fov_x_deg) / 2))
    return fx, fx, width / 2.0, height / 2.0
