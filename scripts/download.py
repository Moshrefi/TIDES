#!/usr/bin/env python
"""Download the datasets and pretrained weights used by this reimplementation.

    python scripts/download.py --list
    python scripts/download.py --weights required          # pose-estimation weights
    python scripts/download.py --weights all               # + downstream-task weights
    python scripts/download.py --dataset eds --seq 00_peanuts_dark
    python scripts/download.py --dataset dsec --seq zurich_city_04_b
    TIDES_HSERGB_URL=<link from the TimeLens e-mail> python scripts/download.py --dataset hsergb

Everything lands under ./data and ./weights (both git-ignored). Downloads are
resumable (HTTP Range) and skipped when the target already exists. HS-ERGB and
BS-ERGB are distributed through a registration form by their authors, so their
URLs must be supplied via environment variables or --url.

The Gaussian scenes themselves have no pretrained weights: a 4DGS scene is fit
per sequence with scripts/fit_4dgs.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Registry. "verify" marks URLs that follow the hosting pattern of the official
# release but could not be checked from the environment this repo was written
# in; if one 404s, copy the current link from the project page and pass --url.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "dust3r": dict(
        group="required",
        url="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        dest="weights/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        note="DUSt3R backbone used by Easi3R for camera poses of dynamic scenes (HS-ERGB, BS-ERGB).",
    ),
    "monst3r": dict(
        group="optional",
        hf_repo="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
        dest="weights/monst3r",
        note="Optional MonST3R backbone for Easi3R (needs `pip install huggingface_hub`).",
    ),
    "e2vid": dict(
        group="downstream",
        url="http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar",
        dest="weights/e2vid/E2VID_lightweight.pth.tar",
        note="E2VID reconstruction (downstream task 1); also handy for visual checks.",
    ),
    "timelens": dict(
        group="downstream",
        url="http://download.ifi.uzh.ch/rpg/web/data/timelens/data2/checkpoint.bin",
        dest="weights/timelens/checkpoint.bin",
        note="TimeLens checkpoint (VFI reference; the paper trains TimeLens-XL on simulated events).",
    ),
}

DSEC_BASE = "https://download.ifi.uzh.ch/rpg/DSEC/train/{seq}/{seq}_{part}"
DSEC_PARTS = ["events_left.zip", "images_rectified_left.zip", "image_timestamps.txt", "calibration.zip"]

DATASETS = {
    "eds": dict(
        page="https://rpg.ifi.uzh.ch/eds.html",
        url="https://download.ifi.uzh.ch/rpg/eds/dataset/{seq}/{seq}.tgz",
        verify=True,
        default_seqs=["00_peanuts_dark", "01_peanuts_light", "07_ziggy_and_fuzz_hdr", "08_peanuts_running"],
        note="Ego-motion with ground-truth 6-DoF poses (beamsplitter RGB + events).",
    ),
    "dsec": dict(
        page="https://dsec.ifi.uzh.ch/dsec-datasets/download/",
        verify=True,
        default_seqs=["zurich_city_04_b"],
        note="Driving, ego-motion. Whole training split is ~125 GB of events alone; download per sequence.",
    ),
    "hsergb": dict(
        page="https://rpg.ifi.uzh.ch/timelens",
        env="TIDES_HSERGB_URL",
        note="Gated: fill in the TimeLens form, then export TIDES_HSERGB_URL=<zip link from the e-mail>.",
    ),
    "bsergb": dict(
        page="https://rpg.ifi.uzh.ch/timelens/timelens++download.html",
        env="TIDES_BSERGB_URL",
        note="Gated: fill in the TimeLens++ form, then export TIDES_BSERGB_URL=<zip link>.",
    ),
}


# ---------------------------------------------------------------------------
def _human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def fetch(url: str, dest: str, sha256: str | None = None, dry_run: bool = False) -> str:
    dest = os.path.join(ROOT, dest) if not os.path.isabs(dest) else dest
    if os.path.exists(dest):
        print(f"  [skip] {os.path.relpath(dest, ROOT)} exists")
        return dest
    print(f"  [get ] {url}\n         -> {os.path.relpath(dest, ROOT)}")
    if dry_run:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "tides-reimplementation"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"  [fail] {url}: {e}\n         Check the dataset page for the current link and pass it with --url.")
    total = int(resp.headers.get("Content-Length", 0)) + (have if resp.status == 206 else 0)
    mode = "ab" if resp.status == 206 else "wb"
    done = have if resp.status == 206 else 0
    with open(part, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r         {_human(done)} / {_human(total)} ({100 * done / total:.1f}%)")
                sys.stdout.flush()
    print()
    if sha256:
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != sha256:
            raise SystemExit(f"  [fail] checksum mismatch for {dest}")
    os.replace(part, dest)
    return dest


def extract(path: str, out_dir: str, remove: bool = False):
    if not os.path.exists(path):
        return
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, f".extracted_{os.path.basename(path)}")
    if os.path.exists(marker):
        return
    print(f"  [unz ] {os.path.relpath(path, ROOT)}")
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
    elif path.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(path) as t:
            t.extractall(out_dir, filter="data")
    else:
        return
    open(marker, "w").close()
    if remove:
        os.remove(path)


def get_weights(which: str, dry_run: bool):
    for name, w in WEIGHTS.items():
        if which not in ("all", name, w["group"]):
            continue
        print(f"[weights] {name}: {w['note']}")
        if "hf_repo" in w:
            if dry_run:
                print(f"  [get ] huggingface {w['hf_repo']} -> {w['dest']}")
                continue
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("  [skip] huggingface_hub not installed (pip install huggingface_hub)")
                continue
            snapshot_download(repo_id=w["hf_repo"], local_dir=os.path.join(ROOT, w["dest"]))
        else:
            fetch(w["url"], w["dest"], w.get("sha256"), dry_run)


def get_dataset(name: str, seqs: list[str] | None, url: str | None, dry_run: bool, keep_archives: bool):
    d = DATASETS[name]
    print(f"[dataset] {name}: {d['note']}\n          page: {d['page']}")
    out = os.path.join(ROOT, "data", name)
    if name in ("hsergb", "bsergb"):
        link = url or os.environ.get(d["env"])
        if not link:
            raise SystemExit(f"  {name} needs a download link: {d['note']}")
        arc = fetch(link, os.path.join("data", "_archives", f"{name}.zip"), dry_run=dry_run)
        if not dry_run:
            extract(arc, out, remove=not keep_archives)
        return
    seqs = seqs or d["default_seqs"]
    for seq in seqs:
        if name == "eds":
            u = url or d["url"].format(seq=seq)
            arc = fetch(u, os.path.join("data", "_archives", "eds", f"{seq}.tgz"), dry_run=dry_run)
            if not dry_run:
                extract(arc, out, remove=not keep_archives)
        elif name == "dsec":
            for part in DSEC_PARTS:
                u = DSEC_BASE.format(seq=seq, part=part)
                dst = os.path.join("data", "dsec", seq, f"{seq}_{part}")
                f = fetch(u, dst, dry_run=dry_run)
                if not dry_run and part.endswith(".zip"):
                    sub = {"events_left.zip": "events/left", "images_rectified_left.zip": "images/left",
                           "calibration.zip": "calibration"}[part]
                    extract(f, os.path.join(out, seq, sub), remove=not keep_archives)
    if d.get("verify"):
        print("  note: if a link above fails, the host may have moved files; see the dataset page.")


def setup_third_party(dry_run: bool):
    script = os.path.join(ROOT, "scripts", "setup_third_party.sh")
    print(f"[code] {script}")
    if not dry_run:
        subprocess.check_call(["bash", script])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show everything that can be downloaded")
    ap.add_argument("--weights", default=None, help="required | optional | downstream | all | <name>")
    ap.add_argument("--dataset", action="append", choices=list(DATASETS), help="repeatable")
    ap.add_argument("--seq", action="append", help="sequence name(s); default: the ones used in the paper's figures")
    ap.add_argument("--url", default=None, help="override the download link (single dataset/sequence)")
    ap.add_argument("--third-party", action="store_true", help="also clone 4DGaussians / Easi3R")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("Weights:")
        for k, w in WEIGHTS.items():
            print(f"  {k:10s} [{w['group']}] {w['note']}")
        print("Datasets:")
        for k, d in DATASETS.items():
            print(f"  {k:10s} {d['note']}")
        return
    if not (a.weights or a.dataset or a.third_party):
        a.weights = "required"
    if a.third_party:
        setup_third_party(a.dry_run)
    if a.weights:
        get_weights(a.weights, a.dry_run)
    for ds in a.dataset or []:
        get_dataset(ds, a.seq, a.url, a.dry_run, a.keep_archives)


if __name__ == "__main__":
    main()
