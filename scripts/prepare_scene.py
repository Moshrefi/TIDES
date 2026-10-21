#!/usr/bin/env python
"""Turn a dataset sequence into (a) a HyperNeRF-format scene that 4DGaussians can
fit and (b) tides_sequence.json used by the simulator.

Poses come from, in order of preference (Sec. 4.2 of the paper):
  1. ground-truth 6-DoF poses (EDS)               --poses_tum
  2. COLMAP for static scenes (DSEC)              --colmap_dir  or --run_colmap
  3. Easi3R for dynamic scenes (HS-ERGB, BS-ERGB) --easi3r_dir

Examples
  python scripts/prepare_scene.py --preset eds --seq_dir data/eds/00_peanuts_dark --out scenes/eds_00 \
      --intrinsics 766.5 767.0 291.9 261.7
  python scripts/prepare_scene.py --preset dsec --seq_dir data/dsec/zurich_city_04_b --out scenes/dsec_04b --run_colmap
  python scripts/prepare_scene.py --preset hsergb --seq_dir data/hsergb/close/test/spinning_umbrella \
      --out scenes/hs_umbrella --easi3r_dir third_party/Easi3R/results/spinning_umbrella

Dataset folder layouts differ between releases; every preset path can be
overridden (--frames_glob, --timestamps, --poses_tum, --intrinsics).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tides.pinhole import EDS_EVENT_CAM, PinholeRemap  # noqa: E402
from tides.sequence import Sequence, quat_to_R, slerp_pose_c2w  # noqa: E402

PRESETS = {
    "eds": dict(frames_glob="images/*.png", timestamps="images_timestamps.txt", poses_tum="stamped_groundtruth.txt",
                calib="calib.yaml"),
    "dsec": dict(frames_glob="images/left/rectified/*.png", timestamps="*_image_timestamps.txt",
                 calib="calibration/cam_to_cam.yaml"),
    "hsergb": dict(frames_glob="images/*.png", timestamps="images/timestamp.txt"),
    "bsergb": dict(frames_glob="images/*.png", timestamps="images/timestamp.txt"),
    # Sequences converted by the Evtween converters (frames/, frame_ts.npy, ev_*.npy).
    # Frames are already in the event camera's pixel grid, so events need no extra camera.
    "evtween": dict(frames_glob="frames/*.png", timestamps="frame_ts.npy"),
}


def _first(seq_dir, pattern):
    if not pattern:
        return None
    m = sorted(glob.glob(os.path.join(seq_dir, pattern)))
    return m[0] if m else None


def read_timestamps(path, n_frames):
    """One timestamp per line (last column used), or a .npy array. Units auto-detected: us if values > 1e4."""
    if path.endswith(".npy"):
        t = np.load(path).astype(np.float64).reshape(-1)
        if len(t) != n_frames:
            print(f"[warn] {len(t)} timestamps for {n_frames} frames; truncating to the shorter")
        return t * (1e-6 if np.nanmax(t) > 1e4 else 1.0)
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    t = np.array([float(r[-1].replace(",", "")) for r in rows])
    if len(t) != n_frames:
        print(f"[warn] {len(t)} timestamps for {n_frames} frames; truncating to the shorter")
    t = t * (1e-6 if np.nanmax(t) > 1e4 else 1.0)
    return t


def read_tum(path, unit=None):
    a = np.loadtxt(path, comments="#")
    t = a[:, 0]
    if unit is None:
        unit = 1e-6 if t.max() > 1e4 else 1.0
    R = np.stack([quat_to_R(q, "xyzw") for q in a[:, 4:8]])
    return t * unit, R, a[:, 1:4]  # camera-to-world


def read_colmap_text(colmap_dir):
    cams = {}
    for ln in open(os.path.join(colmap_dir, "cameras.txt")):
        if ln.startswith("#") or not ln.strip():
            continue
        e = ln.split()
        model, W, H, prm = e[1], int(e[2]), int(e[3]), list(map(float, e[4:]))
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = prm[0]
            cx, cy = prm[1], prm[2]
        else:  # PINHOLE, OPENCV, ...
            fx, fy, cx, cy = prm[:4]
        cams[int(e[0])] = (fx, fy, cx, cy, W, H)
    imgs = {}
    lines = [ln for ln in open(os.path.join(colmap_dir, "images.txt")) if not ln.startswith("#")]
    for ln in lines[::2]:
        e = ln.split()
        if len(e) < 10:
            continue
        q = list(map(float, e[1:5]))  # qw qx qy qz, world-to-camera
        tvec = np.array(list(map(float, e[5:8])))
        imgs[e[9]] = (quat_to_R(q, "wxyz"), tvec, int(e[8]))
    pts, cols = [], []
    pf = os.path.join(colmap_dir, "points3D.txt")
    if os.path.exists(pf):
        for ln in open(pf):
            if ln.startswith("#") or not ln.strip():
                continue
            e = ln.split()
            pts.append(list(map(float, e[1:4])))
            cols.append([int(e[4]) / 255, int(e[5]) / 255, int(e[6]) / 255])
    return cams, imgs, (np.array(pts) if pts else None), (np.array(cols) if cols else None)


def run_colmap(frames, work, K=None):
    """Sparse reconstruction. K (TIDES/COLMAP convention, pixel centres at +0.5) is kept fixed when given."""
    if shutil.which("colmap") is None:
        raise SystemExit("colmap not found; install it (conda install -c conda-forge colmap) or pass --colmap_dir")
    if os.path.isdir(work):
        shutil.rmtree(work)  # stale database / models from an earlier attempt
    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in frames:
        os.symlink(os.path.abspath(f), os.path.join(img_dir, os.path.basename(f)))
    db = os.path.join(work, "database.db")
    sp = os.path.join(work, "sparse")
    os.makedirs(sp, exist_ok=True)
    run = lambda *a: subprocess.check_call(["colmap", *a])  # noqa: E731
    if K is not None:
        cam = ["--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", ",".join(f"{v:.6f}" for v in K)]
        refine = ["--Mapper.ba_refine_focal_length", "0"]
    else:  # one focal length, principal point fixed at the image centre (what 4DGS assumes)
        cam = ["--ImageReader.camera_model", "SIMPLE_PINHOLE"]
        refine = []
    run("feature_extractor", "--database_path", db, "--image_path", img_dir, "--ImageReader.single_camera", "1", *cam)
    if len(frames) <= 300:
        run("exhaustive_matcher", "--database_path", db)
    else:
        run("sequential_matcher", "--database_path", db, "--SequentialMatching.overlap", "20")
    # Video frames have small baselines: allow a smaller triangulation angle to initialise.
    run("mapper", "--database_path", db, "--image_path", img_dir, "--output_path", sp,
        "--Mapper.init_min_tri_angle", "4", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", *refine)
    best, best_n = None, -1
    for m in sorted(os.listdir(sp)):
        txt = os.path.join(work, f"sparse_txt_{m}")
        os.makedirs(txt, exist_ok=True)
        run("model_converter", "--input_path", os.path.join(sp, m), "--output_path", txt, "--output_type", "TXT")
        n = sum(1 for ln in open(os.path.join(txt, "images.txt")) if ln.strip() and not ln.startswith("#")) // 2
        print(f"[colmap] model {m}: {n} images")
        if n > best_n:
            best, best_n = txt, n
    if best is None:
        raise SystemExit("COLMAP produced no model; try --stride 2 or 3 (larger baselines between frames)")
    return best


def read_easi3r(d, n_frames):
    """Easi3R / MonST3R outputs: pred_traj.txt (t x y z qw qx qy qz, camera-to-world) and pred_intrinsics.txt."""
    a = np.loadtxt(os.path.join(d, "pred_traj.txt"))
    R_c2w = np.stack([quat_to_R(q, "wxyz") for q in a[:, 4:8]])
    p_c2w = a[:, 1:4]
    K = np.loadtxt(os.path.join(d, "pred_intrinsics.txt")).reshape(-1, 3, 3)
    if len(a) != n_frames:
        print(f"[warn] Easi3R has {len(a)} poses for {n_frames} frames; frames are subsampled to match")
    return R_c2w, p_c2w, K


def parse_calib(path, preset):
    import yaml

    c = yaml.safe_load(open(path))
    if preset == "dsec":
        K1 = c["intrinsics"]["camRect1"]["camera_matrix"]
        K0 = c["intrinsics"]["camRect0"]["camera_matrix"]
        size0 = c["intrinsics"]["camRect0"]["resolution"]
        T10 = np.asarray(c["extrinsics"]["T_10"])
        Rr0 = np.asarray(c["extrinsics"]["R_rect0"])
        Rr1 = np.asarray(c["extrinsics"]["R_rect1"])
        # x_rect0 = R_rect0 T_01 R_rect1^T x_rect1
        T = np.eye(4)
        T01 = np.linalg.inv(T10)
        T[:3, :3] = Rr0 @ T01[:3, :3] @ Rr1.T
        T[:3, 3] = Rr0 @ T01[:3, 3]
        ev = dict(T_ev_rgb=T.tolist(), intrinsics=list(map(float, K0)), size=list(size0))
        return tuple(map(float, K1)), ev
    # Generic: first camera with an `intrinsics: [fx, fy, cx, cy]` entry.
    def find(o):
        if isinstance(o, dict):
            if "intrinsics" in o and isinstance(o["intrinsics"], list) and len(o["intrinsics"]) == 4:
                return o["intrinsics"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
        return None

    k = find(c)
    return (tuple(map(float, k)) if k else None), None


def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        for p in ("x", "y", "z", "nx", "ny", "nz"):
            f.write(f"property float {p}\n")
        for p in ("red", "green", "blue"):
            f.write(f"property uchar {p}\n")
        f.write("end_header\n")
        for p, c in zip(xyz, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)):
            f.write(f"{p[0]} {p[1]} {p[2]} 0 0 0 {c[0]} {c[1]} {c[2]}\n")


def random_points(seq: Sequence, n=20000, near=0.5, far=10.0, seed=0):
    """Fallback initial cloud: random points inside the camera frusta."""
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = seq.intrinsics
    W, H = seq.size
    idx = rng.integers(0, len(seq.timestamps), n)
    u, v = rng.uniform(0, W, n), rng.uniform(0, H, n)
    z = np.exp(rng.uniform(np.log(near), np.log(far), n))
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    R, t = seq.R_w2c[idx], seq.t_w2c[idx]
    pw = np.einsum("nji,nj->ni", R, pc - t)  # R^T (pc - t)
    return pw, np.full((n, 3), 0.5)


def export_hypernerf(seq: Sequence, out: str, near: float, far: float):
    """HyperNeRF/Nerfies layout read by 4DGaussians (arguments/hypernerf/*.py)."""
    from PIL import Image

    for sub in ("rgb/1x", "rgb/2x", "camera"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    fx, fy, cx, cy = seq.intrinsics
    W, H = seq.size
    ids, meta = [], {}
    t0 = seq.t0
    for i, f in enumerate(seq.frames):
        fid = f"{i:06d}"
        ids.append(fid)
        im = Image.open(f).convert("RGB")
        im.save(os.path.join(out, "rgb/1x", fid + ".png"))
        # 4DGaussians' HyperNeRF loader always reads rgb/2x; we put full-resolution frames there so the
        # scene is fit at the resolution TIDES renders (the loader derives the FoV, not pixels, from the camera).
        im.save(os.path.join(out, "rgb/2x", fid + ".png"))
        R, t = seq.R_w2c[i], seq.t_w2c[i]
        cam = dict(orientation=R.tolist(), position=(-R.T @ t).tolist(), focal_length=float(fx),
                   principal_point=[float(cx), float(cy)], skew=0.0, pixel_aspect_ratio=float(fy / fx),
                   radial_distortion=[0.0, 0.0, 0.0], tangential_distortion=[0.0, 0.0], image_size=[int(W), int(H)])
        json.dump(cam, open(os.path.join(out, "camera", fid + ".json"), "w"))
        # warp_id in microseconds: 4DGS normalises by its max, so scene time is linear in real time.
        tid = int(round((seq.timestamps[i] - t0) * 1e6))
        meta[fid] = dict(warp_id=tid, appearance_id=i, camera_id=0, time_id=i)
    json.dump(dict(count=len(ids), num_exemplars=len(ids), ids=ids, train_ids=ids, val_ids=ids[::10]),
              open(os.path.join(out, "dataset.json"), "w"))
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"))
    json.dump(dict(scale=1.0, center=[0.0, 0.0, 0.0], near=near, far=far), open(os.path.join(out, "scene.json"), "w"))
    pts, cols = (seq.points, seq.colors) if seq.points is not None else random_points(seq, near=near, far=far)
    write_ply(os.path.join(out, "points3D_downsample2.ply"), pts, cols if cols is not None else np.full_like(pts, 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames_glob")
    ap.add_argument("--timestamps")
    ap.add_argument("--poses_tum")
    ap.add_argument("--tum_time_unit", type=float, default=None)
    ap.add_argument("--colmap_dir")
    ap.add_argument("--run_colmap", action="store_true")
    ap.add_argument("--easi3r_dir")
    ap.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                    help="final pinhole intrinsics, pixel centres at +0.5 (COLMAP convention), no resampling")
    ap.add_argument("--src_intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                    help="source camera in OpenCV convention; frames are resampled to a centred pinhole")
    ap.add_argument("--src_dist", type=float, nargs=4, default=None, metavar=("K1", "K2", "P1", "P2"))
    ap.add_argument("--no_eds_calib", action="store_true", help="evtween preset: do not apply the EDS event-camera calibration")
    ap.add_argument("--t_start", type=float, default=None, help="seconds, absolute")
    ap.add_argument("--t_end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--near", type=float, default=0.1)
    ap.add_argument("--far", type=float, default=50.0)
    a = ap.parse_args()

    pre = PRESETS.get(a.preset, {})
    frames_glob = a.frames_glob or pre.get("frames_glob", "images/*.png")
    frames = sorted(glob.glob(os.path.join(a.seq_dir, frames_glob)))
    if not frames:
        raise SystemExit(f"no frames match {os.path.join(a.seq_dir, frames_glob)}")
    ts_path = a.timestamps or _first(a.seq_dir, pre.get("timestamps"))
    if not ts_path:
        raise SystemExit("no timestamp file found; pass --timestamps")
    ts = read_timestamps(ts_path, len(frames))
    n = min(len(ts), len(frames))
    frames, ts = frames[:n], ts[:n]

    sel = np.ones(n, bool)
    if a.t_start is not None:
        sel &= ts >= a.t_start
    if a.t_end is not None:
        sel &= ts <= a.t_end
    idx = np.nonzero(sel)[0][:: a.stride]
    if a.max_frames:
        idx = idx[: a.max_frames]
    frames = [frames[i] for i in idx]
    ts = ts[idx]

    from PIL import Image

    W, H = Image.open(frames[0]).size
    K, ev_cam = None, None
    calib = _first(a.seq_dir, pre.get("calib"))
    if calib:
        try:
            K, ev_cam = parse_calib(calib, a.preset)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not parse {calib}: {e}")
    points = colors = None

    # Resample to a centred pinhole camera when the source camera is known (see tides/pinhole.py).
    remap = None
    align = os.path.join(a.seq_dir, "align.yaml")
    if a.src_intrinsics:
        remap = PinholeRemap(a.src_intrinsics, a.src_dist, (W, H))
    elif a.preset == "evtween" and os.path.exists(align) and not a.no_eds_calib:
        import yaml

        y0, x0, _, _ = yaml.safe_load(open(align))["crop_yxyx"]
        fx, fy, cx, cy = EDS_EVENT_CAM["intrinsics"]
        remap = PinholeRemap((fx, fy, cx - x0, cy - y0), EDS_EVENT_CAM["dist"], (W, H))
        print(f"[cam ] EDS event camera (Evtween crop at x0={x0}, y0={y0}), undistorted to a centred pinhole")
    if remap is not None:
        out_dir = os.path.join(a.out, "frames_pinhole")
        os.makedirs(out_dir, exist_ok=True)
        new = []
        for i, f in enumerate(frames):
            dst = os.path.join(out_dir, f"{i:06d}.png")
            Image.fromarray(remap.remap_image(np.asarray(Image.open(f).convert("RGB")))).save(dst)
            new.append(dst)
        frames = new
        W, H = remap.size_out
        K = remap.intrinsics
        print(f"[cam ] pinhole {W}x{H}, f = {K[0]:.1f}")

    tum = a.poses_tum or _first(a.seq_dir, pre.get("poses_tum"))
    colmap_dir = a.colmap_dir
    if a.run_colmap and not colmap_dir:
        colmap_dir = run_colmap(frames, os.path.join(a.out, "colmap"), K if remap is not None else None)
    if tum:
        tp, Rc, pc = read_tum(tum, a.tum_time_unit)
        R_c2w, p_c2w = slerp_pose_c2w(ts, tp, Rc, pc)
        print(f"[pose] ground truth from {tum}")
    elif colmap_dir:
        cams, imgs, points, colors = read_colmap_text(colmap_dir)
        keep = [i for i, f in enumerate(frames) if os.path.basename(f) in imgs]
        if len(keep) < len(frames):
            print(f"[warn] COLMAP registered {len(keep)}/{len(frames)} frames; dropping the rest")
        if len(keep) < min(len(frames), max(10, 0.5 * len(frames))):
            raise SystemExit(f"COLMAP registered only {len(keep)}/{len(frames)} frames. The camera moves too little "
                             "between frames: rerun with --stride 2 (or 3), or a longer --max_frames.")
        frames = [frames[i] for i in keep]
        ts = ts[keep]
        Rw = np.stack([imgs[os.path.basename(f)][0] for f in frames])
        tw = np.stack([imgs[os.path.basename(f)][1] for f in frames])
        R_c2w = Rw.transpose(0, 2, 1)
        p_c2w = -np.einsum("nij,nj->ni", R_c2w, tw)
        fx, fy, cx, cy, cw, ch = cams[imgs[os.path.basename(frames[0])][2]]
        sx = W / cw
        if remap is None:
            K = K or (fx * sx, fy * sx, cx * sx, cy * sx)
        print(f"[pose] COLMAP from {colmap_dir}")
    elif a.easi3r_dir:
        R_c2w, p_c2w, Ks = read_easi3r(a.easi3r_dir, len(frames))
        if len(R_c2w) != len(frames):
            sub = np.linspace(0, len(frames) - 1, len(R_c2w)).round().astype(int)
            frames = [frames[i] for i in sub]
            ts = ts[sub]
        k = Ks[0]
        sx = W / (2 * k[0, 2])  # Easi3R works at a reduced resolution
        K = K or (k[0, 0] * sx, k[1, 1] * sx, k[0, 2] * sx, k[1, 2] * sx)
        print(f"[pose] Easi3R from {a.easi3r_dir}")
    else:
        raise SystemExit("no pose source: pass --poses_tum, --colmap_dir/--run_colmap or --easi3r_dir")
    if a.intrinsics:
        K = tuple(a.intrinsics)
    if K is None:
        raise SystemExit("intrinsics unknown; pass --intrinsics FX FY CX CY")

    R_w2c = R_c2w.transpose(0, 2, 1)
    t_w2c = -np.einsum("nij,nj->ni", R_w2c, p_c2w)
    seq = Sequence(frames, ts, K, (W, H), R_w2c, t_w2c, points, colors, ev_cam,
                   meta=dict(preset=a.preset, seq_dir=os.path.abspath(a.seq_dir),
                             remap=remap.to_dict() if remap is not None else None))
    os.makedirs(a.out, exist_ok=True)
    export_hypernerf(seq, a.out, a.near, a.far)
    seq.save(os.path.join(a.out, "tides_sequence.json"))
    print(f"[done] {len(frames)} frames, {ts[-1] - ts[0]:.3f}s -> {a.out}")
    print(f"       next: bash scripts/fit_4dgs.sh {a.out} <name>")


if __name__ == "__main__":
    main()
