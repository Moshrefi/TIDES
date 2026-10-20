import torch
import torch.nn.functional as F

from tides.render.projection import project_gaussians
from tides.render.rasterizer import rasterize_tangent
from tides.scene.deformation import Deformation, HexPlaneDeformation, HexPlaneField, bilinear_sample
from tides.scene.synthetic import occlusion_scene, static_trajectory

BG = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)


def _render(scene, traj, t, **kw):
    g = scene.state(t)
    cam = traj(t, dtype=torch.float64)
    return rasterize_tangent(project_gaussians(g, cam), traj.intr[4], traj.intr[5], BG, **kw)


def test_time_derivative_matches_finite_difference_static_camera():
    sc, tr = occlusion_scene(), static_trajectory()
    r = _render(sc, tr, 0.5)
    h = 1e-6
    fd = (_render(sc, tr, 0.5 + h).L - _render(sc, tr, 0.5 - h).L) / (2 * h)
    assert r.L_dot.abs().max() > 1.0  # the scene actually moves
    assert (fd - r.L_dot).abs().max() < 1e-5
    fa = (_render(sc, tr, 0.5 + h).alpha - _render(sc, tr, 0.5 - h).alpha) / (2 * h)
    assert (fa - r.alpha_dot).abs().max() < 1e-5


def test_time_derivative_with_moving_camera_and_time_mapping():
    sc = occlusion_scene().set_time_mapping(0.0, 0.5)
    tr = static_trajectory(pan=0.3, t1=0.5)
    r = _render(sc, tr, 0.23)
    h = 1e-7
    fd = (_render(sc, tr, 0.23 + h).L - _render(sc, tr, 0.23 - h).L) / (2 * h)
    assert (fd - r.L_dot).abs().max() < 1e-4 * (1 + r.L_dot.abs().max())


def test_partial_tile_render_matches_full():
    sc, tr = occlusion_scene(), static_trajectory()
    full = _render(sc, tr, 0.3)
    part = _render(sc, tr, 0.3, tiles=torch.tensor([0, 5, 7]))
    m = part.rendered
    assert m.any() and not m.all()
    assert torch.allclose(full.L[m], part.L[m]) and torch.allclose(full.L_dot[m], part.L_dot[m])


def test_bilinear_sample_matches_grid_sample():
    torch.manual_seed(0)
    plane = torch.randn(1, 5, 7, 9)
    coords = torch.rand(100, 2) * 2 - 1
    ref = F.grid_sample(plane, coords.view(1, 1, -1, 2), align_corners=True, mode="bilinear", padding_mode="border")
    assert torch.allclose(bilinear_sample(plane, coords), ref[0, :, 0].T, atol=1e-5)


def test_hexplane_jvp_matches_finite_difference():
    torch.manual_seed(0)
    P = 50
    deform = Deformation(D=2, W=32, grid=HexPlaneField(bounds=1.0, resolution=(8, 8, 8, 6), out_dim=4))
    for planes in deform.grid.grids:  # make time planes non-constant so motion is non-trivial
        for p in planes:
            p.data.uniform_(0.5, 1.5)
    sc = HexPlaneDeformation(torch.rand(P, 3) - 0.5, torch.full((P, 3), -2.0), torch.randn(P, 4),
                             torch.zeros(P), torch.randn(P, 16, 3) * 0.1, deform).double()
    s = sc.state(0.37)
    h = 1e-6
    fd = (sc.state(0.37 + h).means - sc.state(0.37 - h).means) / (2 * h)
    assert s.means_dot.abs().max() > 0
    assert torch.allclose(fd, s.means_dot, atol=1e-6, rtol=1e-5)
