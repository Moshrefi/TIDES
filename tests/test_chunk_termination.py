"""Regression: early-stopped pixels must not resume in subsequent chunks."""
import torch
from tides.render.projection import ScreenGaussians
from tides.render.rasterizer import rasterize_tangent


def primitives(t=0.):
    dtype = torch.float64
    # The third splat triggers termination; the fourth is weak enough that
    # the old chunked implementation incorrectly resumes compositing it.
    opacity = torch.tensor([.99, .98, .99, .01, .02], dtype=dtype)
    rate = torch.tensor([.01, -.02, .01, .04, -.03], dtype=dtype)
    n = len(opacity)
    return ScreenGaussians(
        uv=torch.full((n, 2), .5, dtype=dtype), uv_dot=torch.zeros(n, 2, dtype=dtype),
        conic=torch.eye(2, dtype=dtype).repeat(n, 1, 1), conic_dot=torch.zeros(n, 2, 2, dtype=dtype),
        opacity=opacity+t*rate, opacity_dot=rate,
        color=torch.tensor([[.2, .3, .4]]*n, dtype=dtype), color_dot=torch.zeros(n, 3, dtype=dtype),
        depth=torch.arange(n, dtype=dtype)+1, radius=torch.ones(n, dtype=dtype), index=torch.arange(n))


def test_chunk_invariance_of_primal_and_tangent():
    reference = rasterize_tangent(primitives(-.01), 1, 1, tile=1, gauss_chunk=10)
    for chunk in (1, 2, 3, 4):
        result = rasterize_tangent(primitives(-.01), 1, 1, tile=1, gauss_chunk=chunk)
        for field in ('rgb', 'rgb_dot', 'L', 'L_dot', 'alpha', 'alpha_dot', 'c'):
            torch.testing.assert_close(getattr(result, field), getattr(reference, field), atol=1e-12, rtol=1e-12)
        assert result.c.item() == 2


def test_terminated_tangent_matches_finite_difference():
    t, h = -.01, 1e-6
    def render(time):
        return rasterize_tangent(primitives(time), 1, 1, tile=1, gauss_chunk=3)
    fd = (render(t+h).L-render(t-h).L)/(2*h)
    torch.testing.assert_close(render(t).L_dot, fd, atol=1e-8, rtol=1e-6)
