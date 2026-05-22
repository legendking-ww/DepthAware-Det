"""
KITTI depth evaluation (sparse LiDAR projection as GT).

Expects:
  data/kitti/
    image_2/000000.png
    velodyne/000000.bin
    calib/000000.txt

For a minimal demo, place a few KITTI samples or symlink official subset.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import parse_kitti_focal, parse_kitti_intrinsic
from src.depth_estimator import create_depth_estimator
from src.detector import create_detector
from src.fusion import get_object_depth
from src.model_check import normalize_backend, require_backend
from src.pipeline import DepthAwarePipeline


def load_velodyne_points(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def project_lidar_to_image(
    points: np.ndarray, calib_path: Path, image_shape: tuple
) -> np.ndarray:
    """Sparse depth map from Velodyne (simplified KITTI chain)."""
    with open(calib_path) as f:
        lines = f.readlines()

    p2 = None
    tr = None
    r0 = np.eye(4)
    for line in lines:
        if line.startswith("P2"):
            p2 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
        if line.startswith("Tr"):
            tr = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
        if line.startswith("R0_rect"):
            r0 = np.eye(4)
            r0[:3, :3] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)

    h, w = image_shape[:2]
    sparse = np.zeros((h, w), dtype=np.float32)
    if p2 is None or tr is None:
        return sparse

    tr4 = np.eye(4)
    tr4[:3, :] = tr
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    cam = (r0 @ tr4 @ pts_h.T).T[:, :3]
    cam = cam[cam[:, 2] > 0.1]
    if cam.size == 0:
        return sparse

    uvz = (p2 @ np.hstack([cam, np.ones((cam.shape[0], 1))]).T).T
    u = (uvz[:, 0] / uvz[:, 2]).astype(int)
    v = (uvz[:, 1] / uvz[:, 2]).astype(int)
    z = uvz[:, 2]
    for ui, vi, zi in zip(u, v, z):
        if 0 <= ui < w and 0 <= vi < h:
            sparse[vi, ui] = zi
    return sparse


def metrics(pred: List[float], gt: List[float]) -> dict:
    pred = np.array(pred, dtype=np.float64)
    gt = np.array(gt, dtype=np.float64)
    mask = (gt > 0.1) & (pred > 0.1)
    if mask.sum() == 0:
        return {"abs_rel": np.nan, "rmse": np.nan, "n": 0}
    p, g = pred[mask], gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    return {"abs_rel": abs_rel, "rmse": rmse, "n": int(mask.sum())}


def eval_frame(
    image_path: Path,
    calib_path: Path,
    velo_path: Optional[Path],
    pipeline: DepthAwarePipeline,
    use_calib: bool,
) -> tuple:
    img = cv2.imread(str(image_path))
    sparse = np.zeros(img.shape[:2], dtype=np.float32)
    if velo_path and velo_path.is_file():
        pts = load_velodyne_points(velo_path)
        sparse = project_lidar_to_image(pts, calib_path, img.shape)

    focal = parse_kitti_focal(str(calib_path))
    k = parse_kitti_intrinsic(str(calib_path))
    if use_calib:
        pipeline.focal_length = focal
        pipeline.k_matrix = k
    else:
        pipeline.focal_length = None
        pipeline.k_matrix = None

    dets, depths, depth_map, scale, _ = pipeline.process_frame(img)

    preds, gts = [], []
    for det, depth in zip(dets, depths):
        if depth is None:
            continue
        gt_d = get_object_depth(sparse, det[:4], method="median")
        if gt_d is None or gt_d <= 0:
            continue
        preds.append(depth)
        gts.append(gt_d)
    return preds, gts


def build_pipeline(backend: str, device: str = "cuda") -> DepthAwarePipeline:
    backend = normalize_backend(backend)
    require_backend(backend)
    det_map = {"torch": "torch", "onnx": "onnx", "trt": "trt"}
    detector = create_detector(backend=det_map[backend], model_name="yolov8s", device=device)
    depth_est = create_depth_estimator(
        backend=backend, encoder="vits", device=device, input_size=518
    )
    return DepthAwarePipeline(
        detector, depth_est, class_names=detector.class_names, enable_bev=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=str, default=str(ROOT / "data" / "kitti"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx", "trt", "pytorch"],
        default="torch",
        help="推理后端（与 Web / run_realtime 一致）",
    )
    args = parser.parse_args()

    root = Path(args.kitti_root)
    image_dir = root / "image_2"
    if not image_dir.is_dir():
        print(f"请将 KITTI 样本放到 {root} (image_2, calib, velodyne)")
        return 1

    backend = normalize_backend(args.backend)
    print(f"后端: {backend}")
    pipeline_cal = build_pipeline(backend)
    pipeline_raw = build_pipeline(backend)
    pipeline_raw.focal_length = None

    all_p_cal, all_g_cal = [], []
    all_p_raw, all_g_raw = [], []

    images = sorted(image_dir.glob("*.png"))[: args.limit]
    for img_path in images:
        stem = img_path.stem
        calib = root / "calib" / f"{stem}.txt"
        velo = root / "velodyne" / f"{stem}.bin"
        if not calib.is_file():
            continue
        p, g = eval_frame(img_path, calib, velo, pipeline_cal, use_calib=True)
        all_p_cal.extend(p)
        all_g_cal.extend(g)
        p2, g2 = eval_frame(img_path, calib, velo, pipeline_raw, use_calib=False)
        all_p_raw.extend(p2)
        all_g_raw.extend(g2)

    m_cal = metrics(all_p_cal, all_g_cal)
    m_raw = metrics(all_p_raw, all_g_raw)
    print("With calibration:", m_cal)
    print("Without calibration:", m_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
