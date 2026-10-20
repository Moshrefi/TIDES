"""Glue for the official 4DGaussians code base (https://github.com/hustvl/4DGaussians).

`scripts/setup_third_party.sh` clones it into third_party/4DGaussians. We use it
to *fit* the per-sequence dynamic scene (training needs its CUDA rasterizer),
then load the trained model here and differentiate it in time.
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.nn.functional as F

from .deformation import bilinear_sample

_orig_grid_sample = F.grid_sample


def _ad_grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    """Drop-in for the HexPlane call pattern: input (1,C,H,W), grid (1,1,N,2) or (1,N,1,2)."""
    if input.dim() != 4 or input.shape[0] != 1 or mode != "bilinear" or not align_corners:
        return _orig_grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    g = grid.reshape(-1, 2)
    out = bilinear_sample(input, g)  # (N,C); border padding
    return out.T.reshape(1, input.shape[1], *grid.shape[1:3])


@contextlib.contextmanager
def patched_grid_sample():
    F.grid_sample = _ad_grid_sample
    torch.nn.functional.grid_sample = _ad_grid_sample
    try:
        yield
    finally:
        F.grid_sample = _orig_grid_sample
        torch.nn.functional.grid_sample = _orig_grid_sample


def _latest_iteration(model_dir: str) -> int:
    pc = os.path.join(model_dir, "point_cloud")
    its = [int(d.split("_")[-1]) for d in os.listdir(pc) if d.startswith("iteration_")]
    return max(its)


def load_upstream_gaussians(model_dir: str, iteration: int = -1, upstream_root: str = "third_party/4DGaussians",
                            config: str | None = None):
    """Rebuild the official GaussianModel from a trained output folder."""
    upstream_root = os.path.abspath(upstream_root)
    if not os.path.isdir(upstream_root):
        raise FileNotFoundError(f"{upstream_root} not found; run scripts/setup_third_party.sh first")
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)

    from argparse import ArgumentParser, Namespace

    from arguments import ModelHiddenParams  # type: ignore
    from scene.gaussian_model import GaussianModel  # type: ignore

    parser = ArgumentParser()
    hp = ModelHiddenParams(parser)
    args = parser.parse_args([])
    cfg_path = os.path.join(model_dir, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = eval(f.read(), {"Namespace": Namespace})  # written by the official train.py
        for k, v in vars(saved).items():
            setattr(args, k, v)
    if config:
        import mmcv  # type: ignore
        from utils.params_utils import merge_hparams  # type: ignore

        args = merge_hparams(args, mmcv.Config.fromfile(config))
    sh_degree = getattr(args, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, hp.extract(args))
    it = _latest_iteration(model_dir) if iteration == -1 else iteration
    d = os.path.join(model_dir, "point_cloud", f"iteration_{it}")
    gaussians.load_ply(os.path.join(d, "point_cloud.ply"))
    gaussians.load_model(d)
    gaussians._deformation.eval()
    return gaussians
