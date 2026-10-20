"""Optional sensor front-end F (Sec. 3.3): S = F(L).

Default is identity (S = L, S_dot = L_dot). Optional blocks, applied in order:
  * optical PSF: Gaussian blur applied to both L and L_dot (linear, so exact);
  * centre-surround: S <- S - k * blur_surround(S) (also linear);
  * first-order photoreceptor low-pass tau dS/dt = X - S, integrated exactly
    under a locally linear input. It is stateful per pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F


@dataclass
class FrontEndConfig:
    psf_sigma: float = 0.0  # pixels; 0 disables
    surround_sigma: float = 0.0
    surround_gain: float = 0.0
    lowpass_tau: float = 0.0  # seconds; 0 disables


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    y = x[None, None]
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y[0, 0]


class SensorFrontEnd:
    def __init__(self, cfg: FrontEndConfig):
        self.cfg = cfg
        self.S = None  # low-pass state (flat)
        self.t_state = None

    @property
    def identity(self) -> bool:
        c = self.cfg
        return c.psf_sigma <= 0 and c.surround_gain == 0 and c.lowpass_tau <= 0

    def spatial(self, L: torch.Tensor, L_dot: torch.Tensor):
        """Stateless linear filters on full (H,W) maps."""
        c = self.cfg
        X, Xd = _blur(L, c.psf_sigma), _blur(L_dot, c.psf_sigma)
        if c.surround_gain:
            X = X - c.surround_gain * _blur(X, c.surround_sigma)
            Xd = Xd - c.surround_gain * _blur(Xd, c.surround_sigma)
        return X, Xd

    def lowpass(self, pix, X0, X0d, dt):
        """Return (S0, S0_dot, S1) for pixels `pix` over a step of length dt, and advance state."""
        tau = self.cfg.lowpass_tau
        S0 = self.S[pix]
        S0d = (X0 - S0) / tau
        # tau S' = X0 + X0d s - S  =>  S(s) = X(s) - tau X0d + (S0 - X0 + tau X0d) e^{-s/tau}
        e = math.exp(-dt / tau)
        S1 = X0 + X0d * dt - tau * X0d + (S0 - X0 + tau * X0d) * e
        self.S[pix] = S1
        return S0, S0d, S1
