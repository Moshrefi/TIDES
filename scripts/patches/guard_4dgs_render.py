"""Patch 4DGaussians' renderer to mask non-finite Gaussians before rasterization.

A single NaN/inf mean, scale or rotation from the deformation network makes the
CUDA rasterizer index out of bounds, which surfaces as an unrelated-looking CUDA
error ("operation not supported on global/shared address space"). The patch
replaces such Gaussians with invisible ones and prints how many there were.
Idempotent: python scripts/patches/guard_4dgs_render.py third_party/4DGaussians
"""
import sys

path = f"{sys.argv[1]}/gaussian_renderer/__init__.py"
src = open(path).read()
TAG = "# [tides] non-finite guard"
if TAG in src:
    sys.exit(0)
anchor = "    rendered_image, radii, depth = rasterizer("
assert src.count(anchor) == 1, "unexpected 4DGaussians version; renderer anchor not found"
guard = f'''    {TAG}
    _bad = ~(torch.isfinite(means3D_final).all(-1) & torch.isfinite(scales_final).all(-1)
             & torch.isfinite(rotations_final).all(-1) & torch.isfinite(opacity).all(-1))
    if bool(_bad.any()):
        print(f"[tides] masked {{int(_bad.sum())}} non-finite Gaussians", flush=True)
        _m = _bad[:, None]
        means3D_final = torch.where(_m, torch.zeros_like(means3D_final), means3D_final)
        scales_final = torch.where(_m, torch.full_like(scales_final, 1e-6), scales_final)
        rotations_final = torch.where(_m, torch.tensor([1.0, 0, 0, 0], device=_m.device).expand_as(rotations_final), rotations_final)
        opacity = torch.where(_m, torch.zeros_like(opacity), opacity)
'''
open(path, "w").write(src.replace(anchor, guard + anchor))
print("[tides] patched", path)
