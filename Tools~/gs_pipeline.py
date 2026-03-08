#!/usr/bin/env python3
"""
QuestRoomScan → Gaussian Splat training pipeline.

Pulls captured keyframes and point cloud from a Quest device,
converts to COLMAP format, and trains a Gaussian Splat.

Full pipeline:  python gs_pipeline.py --package com.your.app
Pull only:      python gs_pipeline.py --pull --package com.your.app
Convert only:   python gs_pipeline.py --convert-only
Train only:     python gs_pipeline.py --train-only

Requirements:
  pip install numpy scipy plyfile
  pip install "msplat[cli]"    # Apple Silicon (Metal)
  OR pip install gsplat        # NVIDIA GPU (CUDA)
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_CAPTURE_DIR = "captures"


# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------

def find_adb():
    """Locate adb binary — PATH first, then common SDK install locations."""
    adb = shutil.which("adb")
    if adb:
        return adb
    candidates = [
        # macOS: Unity Hub bundled SDK (common versions)
        *sorted(Path("/Applications/Unity/Hub/Editor").glob("*/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"), reverse=True),
        # macOS: standalone Android SDK
        Path.home() / "Library/Android/sdk/platform-tools/adb",
        # Linux
        Path.home() / "Android/Sdk/platform-tools/adb",
        # Windows
        Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    print("ERROR: adb not found. Install Android SDK platform-tools or add to PATH.")
    sys.exit(1)


def pull(args):
    """Pull GSExport directory from Quest device via adb."""
    adb = find_adb()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = Path(args.output_dir) / timestamp
    latest_link = Path(args.output_dir) / "latest"

    device_path = f"/storage/emulated/0/Android/data/{args.package}/files/GSExport"
    print(f"Pulling from {device_path} ...")

    capture_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [adb, "pull", device_path + "/.", str(capture_dir)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        alt_path = f"/sdcard/Android/data/{args.package}/files/GSExport"
        print(f"Trying alternate path: {alt_path}")
        result = subprocess.run(
            [adb, "pull", alt_path + "/.", str(capture_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"adb pull failed:\n{result.stderr}")
            sys.exit(1)

    print(result.stdout)

    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(capture_dir.resolve())
    print(f"Capture saved to: {capture_dir}")
    print(f"Symlinked: {latest_link} -> {capture_dir}")
    return capture_dir


# ---------------------------------------------------------------------------
# COLMAP conversion
# ---------------------------------------------------------------------------

def parse_frames(capture_dir: Path):
    """Parse frames.jsonl, deduplicating by image ID (keep last pose per image)."""
    manifest = capture_dir / "frames.jsonl"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found")
        sys.exit(1)

    by_id = {}
    total = 0
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            by_id[frame["id"]] = frame
            total += 1

    img_dir = capture_dir / "images"
    frames = []
    for fid in sorted(by_id.keys()):
        if (img_dir / f"{fid:06d}.jpg").exists():
            frames.append(by_id[fid])

    print(f"Parsed {total} entries -> {len(frames)} unique keyframes (with images)")
    return frames


def unity_to_colmap_pose(px, py, pz, qx, qy, qz, qw):
    """
    Convert a Unity camera pose (left-handed Y-up, camera-to-world)
    to COLMAP convention (right-handed Y-down, world-to-camera).

    Returns (qw, qx, qy, qz, tx, ty, tz) in COLMAP order.
    """
    from scipy.spatial.transform import Rotation

    r_unity = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    # diag(1,-1,1): determinant -1 flips handedness (Y-up → Y-down)
    flip = np.diag([1.0, -1.0, 1.0])

    r_c2w = flip @ r_unity @ flip
    r_w2c = r_c2w.T

    t_world = np.array([px, -py, pz])
    t_w2c = -r_w2c @ t_world

    r_colmap = Rotation.from_matrix(r_w2c)
    quat = r_colmap.as_quat()  # scipy returns (x,y,z,w)
    qw_c, qx_c, qy_c, qz_c = quat[3], quat[0], quat[1], quat[2]

    return qw_c, qx_c, qy_c, qz_c, t_w2c[0], t_w2c[1], t_w2c[2]


def convert_to_colmap(capture_dir: Path, frames: list):
    """Convert parsed frames + images + points3d.ply → COLMAP binary format."""
    sparse_dir = capture_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    f0 = frames[0]
    fx, fy = f0["fx"], f0["fy"]
    cx_sensor, cy_sensor = f0["cx"], f0["cy"]
    sw, sh = f0["sw"], f0["sh"]
    w, h = f0["w"], f0["h"]

    crop_x = (sw - w) / 2.0
    crop_y = (sh - h) / 2.0
    cx = cx_sensor - crop_x
    cy = cy_sensor - crop_y

    print(f"Intrinsics: sensor={sw}x{sh}, image={w}x{h}, "
          f"crop=({crop_x:.0f},{crop_y:.0f}), "
          f"cx={cx_sensor:.1f}->{cx:.1f}, cy={cy_sensor:.1f}->{cy:.1f}")

    # cameras.bin — PINHOLE model (id=1)
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<Q", w))
        f.write(struct.pack("<Q", h))
        f.write(struct.pack("<4d", fx, fy, cx, cy))

    print(f"cameras.bin: PINHOLE {w}x{h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # images.bin
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for frame in frames:
            img_id = frame["id"] + 1
            img_name = f"{frame['id']:06d}.jpg"
            qw, qx, qy, qz, tx, ty, tz = unity_to_colmap_pose(
                frame["px"], frame["py"], frame["pz"],
                frame["qx"], frame["qy"], frame["qz"], frame["qw"],
            )
            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", tx, ty, tz))
            f.write(struct.pack("<I", 1))
            f.write(img_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))

    print(f"images.bin: {len(frames)} images")

    # points3D.bin
    ply_path = capture_dir / "points3d.ply"
    if ply_path.exists():
        points = read_ply_points(ply_path)
        write_colmap_points3d_bin(sparse_dir / "points3D.bin", points)
        print(f"points3D.bin: {len(points)} points from PLY")
    else:
        with open(sparse_dir / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", 0))
        print("points3D.bin: empty (no PLY found)")

    print(f"COLMAP binary files written to {sparse_dir}")
    return sparse_dir


def read_ply_points(ply_path: Path):
    """Read binary PLY (float xyz, float nxnynz, uchar rgb) → COLMAP-coord points."""
    points = []
    with open(ply_path, "rb") as f:
        vertex_count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break

        for _ in range(vertex_count):
            data = f.read(27)
            if len(data) < 27:
                break
            x, y, z, nx, ny, nz = struct.unpack("<6f", data[:24])
            r, g, b = struct.unpack("<3B", data[24:27])
            points.append({"x": x, "y": -y, "z": z, "r": r, "g": g, "b": b})

    return points


def write_colmap_points3d_bin(path: Path, points: list):
    """Write COLMAP-format points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, p in enumerate(points):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", p["x"], p["y"], p["z"]))
            f.write(struct.pack("<3B", p["r"], p["g"], p["b"]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Training backends
# ---------------------------------------------------------------------------

def train(capture_dir: Path, args):
    """Auto-detect and run the best available GS training backend."""
    output_dir = capture_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # msplat (Metal, Apple Silicon)
    try:
        import msplat  # noqa: F401
        print("Using msplat (Metal) for training...")
        train_msplat(capture_dir, output_dir, args)
        return output_dir
    except ImportError:
        pass

    # gsplat (CUDA)
    try:
        import gsplat  # noqa: F401
        print("Using gsplat for training...")
        train_gsplat(capture_dir, output_dir, args)
        return output_dir
    except ImportError:
        pass

    # Original 3DGS repo
    gs_repo = Path(args.gs_repo) if args.gs_repo else None
    if gs_repo and gs_repo.exists():
        print(f"Using 3DGS repo at {gs_repo}...")
        train_3dgs(capture_dir, output_dir, gs_repo, args)
        return output_dir

    print("ERROR: No Gaussian Splatting training backend found.")
    print("Install one of:")
    print("  pip install msplat[cli]   (Apple Silicon, Metal)")
    print("  pip install gsplat        (NVIDIA GPU, CUDA)")
    print("  --gs-repo /path/to/gaussian-splatting  (original 3DGS)")
    sys.exit(1)


def train_msplat(capture_dir: Path, output_dir: Path, args):
    cmd = [
        sys.executable, "-m", "msplat.cli",
        "--input", str(capture_dir),
        "--output", str(output_dir / "splat.ply"),
        "--num-iters", str(args.iterations),
        "--num-downscales", "0",
        "--eval",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        cmd[0:3] = ["msplat-train"]
        print(f"Retrying: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def train_gsplat(capture_dir: Path, output_dir: Path, args):
    cmd = [
        sys.executable, "-m", "gsplat", "fit",
        "--data_dir", str(capture_dir),
        "--result_dir", str(output_dir),
        "--data_factor", "1",
        "--max_steps", str(args.iterations),
        "--init_type", "sfm",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def train_3dgs(capture_dir: Path, output_dir: Path, gs_repo: Path, args):
    cmd = [
        sys.executable, str(gs_repo / "train.py"),
        "-s", str(capture_dir),
        "-m", str(output_dir),
        "--iterations", str(args.iterations),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuestRoomScan → Gaussian Splat pipeline",
    )
    parser.add_argument("--pull", action="store_true",
                        help="Pull data from Quest (included by default in full pipeline)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert to COLMAP (skip pull and train)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train (skip pull and convert)")
    parser.add_argument("--package",
                        help="Android package name (required for --pull)")
    parser.add_argument("--output-dir", default=DEFAULT_CAPTURE_DIR,
                        help=f"Local capture directory (default: {DEFAULT_CAPTURE_DIR})")
    parser.add_argument("--capture-dir",
                        help="Specific capture directory (default: latest)")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Training iterations (default: 7000)")
    parser.add_argument("--gs-repo",
                        help="Path to 3DGS repo (if not using pip-installed backend)")
    args = parser.parse_args()

    only_flags = args.pull or args.convert_only or args.train_only
    do_pull = args.pull or not only_flags
    do_convert = args.convert_only or not only_flags
    do_train = args.train_only or not only_flags

    if do_pull and not args.package:
        parser.error("--package is required when pulling from device")

    capture_dir = None

    if do_pull:
        capture_dir = pull(args)

    if capture_dir is None:
        if args.capture_dir:
            capture_dir = Path(args.capture_dir)
        else:
            latest = Path(args.output_dir) / "latest"
            if latest.is_symlink() or latest.exists():
                capture_dir = latest.resolve()
            else:
                print("ERROR: No capture directory found. Run with --pull first.")
                sys.exit(1)

    print(f"\nUsing capture directory: {capture_dir}")

    if not (capture_dir / "frames.jsonl").exists():
        print(f"ERROR: {capture_dir / 'frames.jsonl'} not found")
        sys.exit(1)

    if do_convert:
        frames = parse_frames(capture_dir)
        convert_to_colmap(capture_dir, frames)

    if do_train:
        output_dir = train(capture_dir, args)

        trained_ply = output_dir / "splat.ply"
        if not trained_ply.exists():
            candidates = list(output_dir.rglob("*.ply"))
            trained_ply = candidates[0] if candidates else None

        if trained_ply and trained_ply.exists():
            print(f"\nTrained PLY: {trained_ply}")
            print("Import into Unity: Tools > Gaussian Splats > Create GaussianSplatAsset")
        else:
            print("\nTraining complete. Check output directory for results.")


if __name__ == "__main__":
    main()
