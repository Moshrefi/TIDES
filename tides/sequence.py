"""Sequence description shared by scene preparation, 4DGS fitting and simulation.

A sequence is RGB frames with timestamps (seconds), pinhole intrinsics,
world-to-camera poses and (optionally) sparse 3D points. It is stored as
`tides_sequence.json` next to the HyperNeRF-format scene that 4DGaussians trains on.
An optional `event_camera` block renders events from a different sensor
(e.g. DSEC, where the event camera is a separate, lower-resolution sensor).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .render.camera import CameraTrajectory


def quat_to_R(q, order="xyzw"):
    q = np.asarray(q, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(R):
    R = np.asarray(R)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def slerp_pose_c2w(t_query, t_poses, R_c2w, p_c2w):
    """Interpolate camera-to-world poses (rotation SLERP via quaternions, linear position)."""
    from scipy.spatial.transform import Rotation, Slerp

    t_query = np.clip(t_query, t_poses[0], t_poses[-1])
    sl = Slerp(t_poses, Rotation.from_matrix(R_c2w))
    R = sl(t_query).as_matrix()
    p = np.stack([np.interp(t_query, t_poses, p_c2w[:, k]) for k in range(3)], -1)
    return R, p


@dataclass
class Sequence:
    frames: list  # image paths
    timestamps: np.ndarray  # (N,) seconds
    intrinsics: tuple  # fx, fy, cx, cy (pixels, at full resolution)
    size: tuple  # width, height
    R_w2c: np.ndarray  # (N,3,3)
    t_w2c: np.ndarray  # (N,3)
    points: np.ndarray | None = None  # (M,3)
    colors: np.ndarray | None = None  # (M,3) in [0,1]
    event_camera: dict | None = None  # {"T_ev_rgb": 4x4, "intrinsics": [fx,fy,cx,cy], "size": [W,H]}
    meta: dict = field(default_factory=dict)

    @property
    def t0(self) -> float:
        return float(self.timestamps[0])

    @property
    def t1(self) -> float:
        return float(self.timestamps[-1])

    def save(self, path: str):
        d = dict(frames=[os.path.abspath(f) for f in self.frames], timestamps=self.timestamps.tolist(),
                 intrinsics=list(self.intrinsics), size=list(self.size), R_w2c=self.R_w2c.tolist(),
                 t_w2c=self.t_w2c.tolist(), event_camera=self.event_camera, meta=self.meta)
        with open(path, "w") as f:
            json.dump(d, f)

    @staticmethod
    def load(path: str) -> "Sequence":
        with open(path) as f:
            d = json.load(f)
        return Sequence(d["frames"], np.asarray(d["timestamps"]), tuple(d["intrinsics"]), tuple(d["size"]),
                        np.asarray(d["R_w2c"]), np.asarray(d["t_w2c"]), event_camera=d.get("event_camera"),
                        meta=d.get("meta", {}))

    def remap_events(self, ev: dict) -> dict:
        """Map real events from the dataset's pixel grid to the grid this sequence renders."""
        r = (self.meta or {}).get("remap")
        if not r:
            return dict(ev, width=self.size[0], height=self.size[1])
        from .pinhole import PinholeRemap

        return PinholeRemap.from_dict(r).remap_events(ev)

    def trajectory(self, camera: str = "rgb", scale: float = 1.0) -> CameraTrajectory:
        """Camera trajectory for rendering. camera='event' applies the event-camera extrinsics."""
        R, t = self.R_w2c.copy(), self.t_w2c.copy()
        (fx, fy, cx, cy), (W, H) = self.intrinsics, self.size
        if camera == "event" and self.event_camera:
            T = np.asarray(self.event_camera["T_ev_rgb"], dtype=np.float64)  # x_ev = T x_rgb
            R = np.einsum("ij,njk->nik", T[:3, :3], R)
            t = t @ T[:3, :3].T + T[:3, 3]
            fx, fy, cx, cy = self.event_camera["intrinsics"]
            W, H = self.event_camera["size"]
        s = scale
        return CameraTrajectory(torch.tensor(self.timestamps), torch.tensor(R), torch.tensor(t),
                                fx * s, fy * s, cx * s, cy * s, int(round(W * s)), int(round(H * s)))
