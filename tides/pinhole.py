"""Resample images (and event coordinates) to an ideal, centred pinhole camera.

4DGaussians, like 3DGS, assumes a pinhole camera with fx = fy and the principal
point at the image centre: its camera loader keeps only one focal length and
the field of view. TIDES renders the same camera. Real sequences rarely satisfy
this (lens distortion, off-centre principal point, cropped frames), so every
sequence is resampled once to such a camera before COLMAP and 4DGS see it. The
same mapping is applied to the real event coordinates for evaluation.

Conventions: `K_src` and the distortion are OpenCV/Kalibr (pixel centres at
integer coordinates). The returned TIDES intrinsics use pixel centres at
+0.5, the same as COLMAP and the rasterizer, so (f, f, W/2, H/2).
"""
from __future__ import annotations

import numpy as np

# EDS dual setup, event camera (Kalibr camchain from uzh-rpg/bundles-eds,
# config/data/dual_setup/03_calib), as used by Evtween's EDS converter.
EDS_EVENT_CAM = dict(intrinsics=[560.8520948927032, 560.6295819972383, 313.00733235019237, 217.32858679842997],
                     dist=[-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789])


def radtan_distort(x, y, k):
    k1, k2, p1, p2 = k
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def radtan_undistort(xd, yd, k, iters=30):
    x, y = np.array(xd, dtype=np.float64), np.array(yd, dtype=np.float64)
    for _ in range(iters):
        dx, dy = radtan_distort(x, y, k)
        x, y = x - (dx - xd), y - (dy - yd)
    return x, y


class PinholeRemap:
    """Map from a (distorted, off-centre) source camera to a centred pinhole camera."""

    def __init__(self, K_src, dist, size_src, margin: float = 1.0, f_out: float | None = None):
        self.K_src = [float(v) for v in K_src]
        self.dist = [float(v) for v in (dist if dist is not None else [0, 0, 0, 0])]
        self.size_src = (int(size_src[0]), int(size_src[1]))
        fx, fy, _, _ = self.K_src
        self.f = float(f_out or 0.5 * (fx + fy))
        self.size_out = self._largest_valid(margin)

    # -- output pixel (cv) -> source pixel (cv)
    def out_to_src(self, u, v, size_out=None):
        Wo, Ho = size_out or self.size_out
        cxo, cyo = Wo / 2 - 0.5, Ho / 2 - 0.5
        xn, yn = (u - cxo) / self.f, (v - cyo) / self.f
        xd, yd = radtan_distort(xn, yn, self.dist)
        fx, fy, cx, cy = self.K_src
        return fx * xd + cx, fy * yd + cy

    def src_to_out(self, x, y):
        fx, fy, cx, cy = self.K_src
        xn, yn = radtan_undistort((np.asarray(x, np.float64) - cx) / fx, (np.asarray(y, np.float64) - cy) / fy, self.dist)
        Wo, Ho = self.size_out
        return self.f * xn + Wo / 2 - 0.5, self.f * yn + Ho / 2 - 0.5

    def _valid(self, Wo, Ho, margin):
        W, H = self.size_src
        u = np.concatenate([np.arange(Wo), np.arange(Wo), np.zeros(Ho), np.full(Ho, Wo - 1)])
        v = np.concatenate([np.zeros(Wo), np.full(Wo, Ho - 1), np.arange(Ho), np.arange(Ho)])
        x, y = self.out_to_src(u, v, (Wo, Ho))
        return bool((x >= margin).all() and (x <= W - 1 - margin).all() and (y >= margin).all() and (y <= H - 1 - margin).all())

    def _largest_valid(self, margin):
        """Largest output size (same aspect as the source) whose every pixel maps inside the source."""
        W, H = self.size_src
        s = 1.0
        while s > 0.2:
            Wo, Ho = int(s * W / 2) * 2, int(s * H / 2) * 2
            if self._valid(Wo, Ho, margin):
                return Wo, Ho
            s -= 0.005
        raise ValueError("no valid pinhole crop; check the intrinsics / distortion")

    @property
    def intrinsics(self):
        """TIDES / COLMAP convention (pixel centres at +0.5)."""
        Wo, Ho = self.size_out
        return (self.f, self.f, Wo / 2, Ho / 2)

    def remap_image(self, img: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        Wo, Ho = self.size_out
        v, u = np.mgrid[0:Ho, 0:Wo].astype(np.float64)
        x, y = self.out_to_src(u, v)
        img = np.asarray(img)
        chans = [map_coordinates(img[..., c].astype(np.float32), [y, x], order=1, mode="nearest") for c in range(img.shape[2])]
        return np.clip(np.stack(chans, -1), 0, 255).round().astype(np.uint8)

    def remap_events(self, ev: dict) -> dict:
        """Move real events to the output pixel grid (nearest pixel); events falling outside are dropped."""
        u, v = self.src_to_out(ev["x"], ev["y"])
        xi, yi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        Wo, Ho = self.size_out
        keep = (xi >= 0) & (xi < Wo) & (yi >= 0) & (yi < Ho)
        out = {k: np.asarray(v)[keep] for k, v in ev.items() if k in ("t", "p")}
        out.update(x=xi[keep], y=yi[keep], width=Wo, height=Ho)
        return out

    def to_dict(self):
        return dict(K_src=self.K_src, dist=self.dist, size_src=list(self.size_src), f=self.f, size_out=list(self.size_out))

    @staticmethod
    def from_dict(d):
        r = PinholeRemap.__new__(PinholeRemap)
        r.K_src, r.dist, r.size_src, r.f = d["K_src"], d["dist"], tuple(d["size_src"]), d["f"]
        r.size_out = tuple(d["size_out"])
        return r
