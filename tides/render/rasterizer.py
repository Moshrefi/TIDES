"""Forward-mode time-derivative Gaussian splatting rasterizer (paper Sec. 3.1-3.2).

This is the reference implementation in PyTorch. It produces, in a single
front-to-back traversal and in exactly the same compositing order as the primal
image:

    rgb, rgb_dot           Eq. 4 and Eq. 14
    L, L_dot               Eq. 5 and Eq. 15
    alpha, alpha_dot       alpha = 1 - T_K, alpha_dot = -T_dot_K
    b, c                   coverage proxy and contributor count (Sec. 3.4)

The traversal follows the 3DGS CUDA rasterizer: 16x16 tiles, depth-sorted
Gaussians per tile, alpha skipped below 1/255, saturation at tau_max, and early
termination when transmittance would fall below 1e-4. Every one of these
decisions is taken on the primal values and the tangent is propagated through
the same decision (Eq. 11), which is what makes L_dot visibility-consistent.

`tiles` lets the caller render only a subset of tiles, which the adaptive
stepper uses to refine time only where risk is high.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .projection import ScreenGaussians

REC601 = (0.299, 0.587, 0.114)


@dataclass
class TangentRender:
    rgb: torch.Tensor  # (3,H,W)
    rgb_dot: torch.Tensor  # (3,H,W)
    Y: torch.Tensor  # (H,W) luminance
    L: torch.Tensor  # (H,W) log luminance
    L_dot: torch.Tensor  # (H,W)
    alpha: torch.Tensor  # (H,W)
    alpha_dot: torch.Tensor  # (H,W)
    b: torch.Tensor  # (H,W) coverage proxy (accumulated blending mass)
    c: torch.Tensor  # (H,W) contributor count
    tau_sum: torch.Tensor  # (H,W) sum of per-splat opacities that were composited
    rendered: torch.Tensor  # (H,W) bool, which pixels were actually rendered


def _bin_gaussians(sg: ScreenGaussians, width: int, height: int, tile: int):
    """Return (tile_ids, gauss_ids) sorted by (tile, depth), like duplicateWithKeys + radix sort."""
    n_tx = (width + tile - 1) // tile
    n_ty = (height + tile - 1) // tile
    device = sg.uv.device
    if sg.uv.shape[0] == 0:
        e = torch.zeros(0, dtype=torch.long, device=device)
        return e, e, n_tx, n_ty
    u, v, r = sg.uv[:, 0], sg.uv[:, 1], sg.radius
    x0 = ((u - r) / tile).floor().clamp(0, n_tx - 1).long()
    x1 = ((u + r) / tile).floor().clamp(0, n_tx - 1).long()
    y0 = ((v - r) / tile).floor().clamp(0, n_ty - 1).long()
    y1 = ((v + r) / tile).floor().clamp(0, n_ty - 1).long()
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    counts = nx * ny
    gid = torch.repeat_interleave(torch.arange(len(u), device=device), counts)
    start = torch.cumsum(counts, 0) - counts
    local = torch.arange(int(counts.sum()), device=device) - start[gid]
    tx = x0[gid] + local % nx[gid]
    ty = y0[gid] + local // nx[gid]
    tid = ty * n_tx + tx
    depth_rank = torch.empty_like(gid)
    depth_rank[torch.argsort(sg.depth)] = torch.arange(len(u), device=device)
    key = tid * len(u) + depth_rank[gid]
    order = torch.argsort(key)
    return tid[order], gid[order], n_tx, n_ty


def rasterize_tangent(
    sg: ScreenGaussians,
    width: int,
    height: int,
    bg: torch.Tensor | None = None,
    bg_dot: torch.Tensor | None = None,
    tiles: torch.Tensor | None = None,
    tile: int = 16,
    tau_max: float = 0.99,
    alpha_min: float = 1.0 / 255.0,
    T_min: float = 1e-4,
    log_eps: float = 1e-3,
    lum_weights=REC601,
    tile_batch: int = 32,
    gauss_chunk: int = 256,
    coverage: str = "alpha",
) -> TangentRender:
    device = sg.uv.device
    dtype = sg.uv.dtype
    if bg is None:
        bg = torch.zeros(3, dtype=dtype, device=device)
    if bg_dot is None:
        bg_dot = torch.zeros(3, dtype=dtype, device=device)

    tid, gid, n_tx, n_ty = _bin_gaussians(sg, width, height, tile)
    n_tiles = n_tx * n_ty
    if tiles is None:
        tiles = torch.arange(n_tiles, device=device)
    tiles = tiles.to(device).long()

    # Per-tile ranges in the sorted list.
    tile_count = torch.bincount(tid, minlength=n_tiles)
    tile_start = torch.cumsum(tile_count, 0) - tile_count

    HWp = n_ty * tile * n_tx * tile
    out = {k: torch.zeros(HWp, dtype=dtype, device=device) for k in
           ("T", "T_dot", "b", "c", "tau_sum")}
    out["T"].fill_(1.0)
    rgb = torch.zeros(HWp, 3, dtype=dtype, device=device)
    rgb_dot = torch.zeros(HWp, 3, dtype=dtype, device=device)
    rendered = torch.zeros(HWp, dtype=torch.bool, device=device)

    # Pixel offsets inside a tile; pixel centres at integer + 0.5.
    oy, ox = torch.meshgrid(torch.arange(tile, device=device), torch.arange(tile, device=device), indexing="ij")
    ox, oy = ox.reshape(-1), oy.reshape(-1)
    n_px = tile * tile
    padded_w = n_tx * tile

    # Process the most expensive tiles together so padding is small.
    tiles = tiles[torch.argsort(tile_count[tiles], descending=True)]

    for bstart in range(0, len(tiles), tile_batch):
        tb = tiles[bstart:bstart + tile_batch]
        B = len(tb)
        txs, tys = tb % n_tx, tb // n_tx
        px = (txs[:, None] * tile + ox[None]).to(dtype) + 0.5  # (B,n_px)
        py = (tys[:, None] * tile + oy[None]).to(dtype) + 0.5
        flat = ((tys[:, None] * tile + oy[None]) * padded_w + txs[:, None] * tile + ox[None])  # (B,n_px)
        rendered[flat.reshape(-1)] = True

        T = torch.ones(B, n_px, dtype=dtype, device=device)
        Td = torch.zeros(B, n_px, dtype=dtype, device=device)
        acc = torch.zeros(B, n_px, 3, dtype=dtype, device=device)
        acc_d = torch.zeros(B, n_px, 3, dtype=dtype, device=device)
        cnt = torch.zeros(B, n_px, dtype=dtype, device=device)
        bsum = torch.zeros(B, n_px, dtype=dtype, device=device)
        tsum = torch.zeros(B, n_px, dtype=dtype, device=device)

        counts = tile_count[tb]
        starts = tile_start[tb]
        max_m = int(counts.max()) if B > 0 else 0
        for c0 in range(0, max_m, gauss_chunk):
            m = min(gauss_chunk, max_m - c0)
            j = torch.arange(c0, c0 + m, device=device)
            valid_g = j[None] < counts[:, None]  # (B,m)
            pos = (starts[:, None] + j[None]).clamp(max=max(len(gid) - 1, 0))
            gi = gid[pos]  # (B,m)

            uv = sg.uv[gi]  # (B,m,2)
            uvd = sg.uv_dot[gi]
            Q = sg.conic[gi]  # (B,m,2,2)
            Qd = sg.conic_dot[gi]
            op = sg.opacity[gi]
            opd = sg.opacity_dot[gi]
            col = sg.color[gi]  # (B,m,3)
            cold = sg.color_dot[gi]

            dx = px[:, :, None] - uv[:, None, :, 0]  # (B,n_px,m)
            dy = py[:, :, None] - uv[:, None, :, 1]
            qa, qb, qc = Q[..., 0, 0][:, None], Q[..., 0, 1][:, None], Q[..., 1, 1][:, None]
            qad, qbd, qcd = Qd[..., 0, 0][:, None], Qd[..., 0, 1][:, None], Qd[..., 1, 1][:, None]
            # Eq. 9: p = -1/2 d^T Q d.
            power = -0.5 * (qa * dx * dx + 2 * qb * dx * dy + qc * dy * dy)
            # Eq. 10: d_dot = -u_dot,  p_dot = -1/2 (d^T Q_dot d + 2 d_dot^T Q d).
            ddx = -uvd[:, None, :, 0]
            ddy = -uvd[:, None, :, 1]
            Qd_x = qa * dx + qb * dy
            Qd_y = qb * dx + qc * dy
            power_dot = -0.5 * ((qad * dx * dx + 2 * qbd * dx * dy + qcd * dy * dy) + 2 * (ddx * Qd_x + ddy * Qd_y))

            g = torch.exp(power.clamp(max=0.0))
            g_dot = g * power_dot
            tau_raw = op[:, None] * g  # Eq. 2
            tau_raw_dot = opd[:, None] * g + op[:, None] * g_dot
            saturated = tau_raw >= tau_max
            tau = torch.where(saturated, torch.full_like(tau_raw, tau_max), tau_raw)
            tau_dot = torch.where(saturated, torch.zeros_like(tau_raw_dot), tau_raw_dot)  # Eq. 11

            keep = valid_g[:, None, :] & (power <= 0) & (tau >= alpha_min)
            tau = torch.where(keep, tau, torch.zeros_like(tau))
            tau_dot = torch.where(keep, tau_dot, torch.zeros_like(tau_dot))

            # Early termination (3DGS): a splat is composited only while
            # T_before * (1 - tau) >= T_min. T is monotone, so this is a prefix.
            one_m = 1.0 - tau
            T_incl = T[..., None] * torch.cumprod(one_m, dim=-1)
            included = (T_incl >= T_min) & keep
            tau = torch.where(included, tau, torch.zeros_like(tau))
            tau_dot = torch.where(included, tau_dot, torch.zeros_like(tau_dot))
            one_m = 1.0 - tau

            # Front-to-back transmittance and its derivative (Eq. 12-13).
            cp = torch.cumprod(one_m, dim=-1)
            T_after = T[..., None] * cp
            T_before = torch.cat([T[..., None], T_after[..., :-1]], dim=-1)
            # T_k = T_{k-1} (1 - tau_k) => T_dot_k / T_k = T_dot_{k-1}/T_{k-1} - tau_dot_k / (1 - tau_k)
            log_rate = torch.cumsum(-tau_dot / one_m, dim=-1)
            Td_after = T_after * (Td / T)[..., None] + T_after * log_rate
            Td_before = torch.cat([Td[..., None], Td_after[..., :-1]], dim=-1)

            w = T_before * tau
            w_dot = Td_before * tau + T_before * tau_dot  # Eq. 12
            acc = acc + torch.einsum("bpm,bmc->bpc", w, col)
            acc_d = acc_d + torch.einsum("bpm,bmc->bpc", w_dot, col) + torch.einsum("bpm,bmc->bpc", w, cold)
            cnt = cnt + included.to(dtype).sum(-1)
            bsum = bsum + w.sum(-1)
            tsum = tsum + tau.sum(-1)
            T = T_after[..., -1]
            Td = Td_after[..., -1]
            if bool((T < T_min).all()):
                break

        f = flat.reshape(-1)
        out["T"][f] = T.reshape(-1)
        out["T_dot"][f] = Td.reshape(-1)
        out["c"][f] = cnt.reshape(-1)
        out["b"][f] = bsum.reshape(-1)
        out["tau_sum"][f] = tsum.reshape(-1)
        rgb[f] = acc.reshape(-1, 3)
        rgb_dot[f] = acc_d.reshape(-1, 3)

    T, Td = out["T"], out["T_dot"]
    # Eq. 4 and Eq. 14: add the background through the final transmittance.
    rgb = rgb + T[:, None] * bg[None]
    rgb_dot = rgb_dot + Td[:, None] * bg[None] + T[:, None] * bg_dot[None]
    lw = torch.tensor(lum_weights, dtype=dtype, device=device)
    Y = rgb @ lw
    Yd = rgb_dot @ lw
    L = torch.log(Y + log_eps)  # Eq. 5
    Ld = Yd / (Y + log_eps)  # Eq. 15

    Hp, Wp = n_ty * tile, n_tx * tile

    def crop(x):
        x = x.reshape(Hp, Wp, *x.shape[1:])[:height, :width]
        return x.permute(2, 0, 1).contiguous() if x.dim() == 3 else x

    b = out["b"] if coverage == "alpha" else out["tau_sum"]
    return TangentRender(
        rgb=crop(rgb), rgb_dot=crop(rgb_dot), Y=crop(Y), L=crop(L), L_dot=crop(Ld),
        alpha=crop(1.0 - T), alpha_dot=crop(-Td), b=crop(b), c=crop(out["c"]),
        tau_sum=crop(out["tau_sum"]), rendered=crop(rendered),
    )


def tile_grid(width: int, height: int, tile: int = 16):
    return (width + tile - 1) // tile, (height + tile - 1) // tile


def pixel_tile_ids(width: int, height: int, tile: int = 16, device="cpu") -> torch.Tensor:
    """(H,W) map from pixel to tile id, matching rasterize_tangent's tiling."""
    n_tx, _ = tile_grid(width, height, tile)
    ys = torch.arange(height, device=device) // tile
    xs = torch.arange(width, device=device) // tile
    return ys[:, None] * n_tx + xs[None, :]
