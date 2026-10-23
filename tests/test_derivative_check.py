"""The stage-by-stage derivative check passes on correct code and catches small errors."""
import dataclasses
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

from tides import TIDES, TIDESConfig
from tides.eval.derivative_check import run_check
from tides.scene.synthetic import occlusion_scene, static_trajectory

UPSTREAM = os.environ.get("TIDES_4DGS", os.path.join(os.path.dirname(__file__), "..", "third_party", "4DGaussians"))


def _sim(scene, traj):
    return TIDES(scene, traj, TIDESConfig(device="cpu", background=(1.0, 1.0, 1.0)))


def _quiet(*a):
    pass


def test_check_passes_on_synthetic_scene():
    sim = _sim(occlusion_scene().set_time_mapping(0.0, 0.5), static_trajectory(width=96, height=72, t1=0.5, n=16, pan=0.3))
    assert run_check(sim, log=_quiet)["verdict"] == "PASS"


def test_check_catches_a_one_percent_error():
    sim = _sim(occlusion_scene().set_time_mapping(0.0, 0.5), static_trajectory(width=96, height=72, t1=0.5, n=16, pan=0.3))
    render = sim.render
    sim.render = lambda t, tiles=None: dataclasses.replace(render(t, tiles), L_dot=1.01 * render(t, tiles).L_dot)
    rep = run_check(sim, log=_quiet)
    assert rep["verdict"] == "FAIL" and rep["first_failure"] == "image.L"


class _MockGaussians:
    """Just the fields of the official GaussianModel that the deformation call uses."""

    def __init__(self, P, deformation, g):
        self._xyz = nn.Parameter((torch.rand(P, 3, generator=g) - 0.5) * 2.0 + torch.tensor([0.0, 0.0, 3.0]))
        self._scaling = nn.Parameter(torch.full((P, 3), -2.5) + 0.3 * torch.randn(P, 3, generator=g))
        self._rotation = nn.Parameter(torch.randn(P, 4, generator=g))
        self._opacity = nn.Parameter(torch.randn(P, 1, generator=g))
        self._features_dc = nn.Parameter(torch.randn(P, 1, 3, generator=g) * 0.5)
        self._features_rest = nn.Parameter(torch.randn(P, 15, 3, generator=g) * 0.05)
        self._deformation = deformation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)


@pytest.mark.skipif(not os.path.isdir(os.path.join(UPSTREAM, "scene")), reason="official 4DGaussians code not found")
def test_check_passes_through_the_official_4dgs_deformation():
    from tides.scene.deformation import Upstream4DGS

    sys.path.insert(0, os.path.abspath(UPSTREAM))
    sys.modules.setdefault("tkinter", types.SimpleNamespace(W=None))  # imported but unused upstream
    if "scene" not in sys.modules:  # load scene/*.py without scene/__init__.py (dataset readers, torchvision)
        pkg = types.ModuleType("scene")
        pkg.__path__ = [os.path.join(os.path.abspath(UPSTREAM), "scene")]
        sys.modules["scene"] = pkg
    from argparse import ArgumentParser

    from arguments import ModelHiddenParams  # type: ignore
    from scene.deformation import deform_network  # type: ignore

    ap = ArgumentParser()
    hp = ModelHiddenParams(ap)
    args = hp.extract(ap.parse_args([]))
    # arguments/hypernerf/default.py, the config scripts/fit_4dgs.sh trains with
    args.kplanes_config = dict(grid_dimensions=2, input_coordinate_dim=4, output_coordinate_dim=16,
                               resolution=[64, 64, 64, 150])
    args.multires, args.defor_depth, args.net_width = [1, 2, 4], 1, 128
    torch.manual_seed(0)
    net = deform_network(args)
    net.deformation_net.set_aabb(torch.tensor([1.2, 1.2, 4.2]).tolist(), torch.tensor([-1.2, -1.2, 1.8]).tolist())
    with torch.no_grad():  # trained-looking, time-varying planes and non-zero heads
        for p in net.deformation_net.grid.parameters():
            p.add_(0.3 * torch.randn_like(p))
        for name, p in net.named_parameters():
            if "grid" not in name and p.dim() == 2:
                p.copy_(torch.randn_like(p) / p.shape[1] ** 0.5)
    g = _MockGaussians(300, net, torch.Generator().manual_seed(1))
    scene = Upstream4DGS(g).set_time_mapping(0.0, 1.0).to(torch.float64)
    assert g._xyz.dtype == torch.float64 and next(net.parameters()).dtype == torch.float64
    traj = static_trajectory(width=96, height=72, t1=1.0, n=150, pan=0.2)
    rep = run_check(_sim(scene, traj), log=_quiet)
    assert rep["sampler"]["verdict"] == "PASS"
    assert rep["stages"]["raw.mu"]["verdict"] == "PASS"
    assert rep["verdict"] == "PASS", {k: v["verdict"] for k, v in rep["stages"].items()}
