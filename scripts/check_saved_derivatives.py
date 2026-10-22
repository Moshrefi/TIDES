"""Diagnose an existing fitted scene. Never runs COLMAP, training or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--upstream-root', type=Path, default=ROOT / 'third_party/4DGaussians')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tile-count', type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new output path; prior diagnostics are preserved')
    import torch
    from tides.sequence import Sequence
    from tides.scene.deformation import Upstream4DGS
    from tides.render.projection import project_gaussians
    from tides.render.rasterizer import rasterize_tangent
    from tides.render.derivative_check import check_derivative
    scene_file = args.scene / 'tides_sequence.json' if args.scene.is_dir() else args.scene
    seq = Sequence.load(str(scene_file))
    scene = Upstream4DGS.from_model_dir(str(args.model_dir), args.iteration, str(args.upstream_root))
    scene.set_time_mapping(seq.t0, seq.t1)
    scene.to(args.device)
    trajectory = seq.trajectory('rgb')
    times = trajectory.times.cpu().numpy()
    if len(times) < 2 or args.tile_count < 1:
        raise ValueError('Need multiple camera poses and a positive tile count')
    w, h = trajectory.intr[4:]
    tile_total = ((w+15)//16)*((h+15)//16)
    tiles = torch.linspace(0, tile_total-1, min(args.tile_count, tile_total), device=args.device).long().unique()

    @torch.no_grad()
    def render(t):
        g = scene.state(t)
        camera = trajectory(t, dtype=g.means.dtype, device=args.device)
        bg = torch.tensor(scene.background, dtype=g.means.dtype, device=args.device)
        return rasterize_tangent(project_gaussians(g, camera), w, h, bg=bg, tiles=tiles)

    reports = []
    for fraction in (.25, .5, .75):
        index = min(int((len(times)-1)*fraction), len(times)-2)
        dt = float(times[index+1]-times[index])
        # Stay inside a pose segment: central FD across a keyframe is not the
        # derivative of either piece of the piecewise camera trajectory.
        t = float(times[index] + .37*dt)
        report = check_derivative(render, t, .05*dt)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    paths = [ROOT / 'tides/render/rasterizer.py', ROOT / 'tides/render/derivative_check.py', scene_file]
    record = dict(reports=reports, tiles=tiles.cpu().tolist(), torch_version=torch.__version__,
                  commit=commit, model_dir=str(args.model_dir), requested_iteration=args.iteration,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  status='PASS' if all(r['status'] == 'PASS' for r in reports) else
                         'FAIL' if any(r['status'] == 'FAIL' for r in reports) else 'INCONCLUSIVE',
                  limitations=['Sampled tiles/times only; not proof of full-scene correctness.',
                               'No cached event output was regenerated or retroactively validated.',
                               'Changing renderer code can invalidate previously generated event caches.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print('Overall:', record['status'], '| Report:', args.out)
    return 0 if record['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
