"""Metric scale calibration from known object heights."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Union

import numpy as np

# COCO class names used by YOLO — typical heights in meters
REAL_HEIGHTS: Dict[str, float] = {
    "car": 1.5,
    "truck": 2.5,
    "bus": 3.0,
    "motorcycle": 1.2,
    "bicycle": 1.2,
    "person": 1.7,
}


def estimate_scale_factor(
    depth_model_values: Sequence[float],
    boxes: Sequence[Sequence[float]],
    class_indices: Sequence[int],
    focal_length_px: float,
    class_names: Union[Mapping[int, str], List[str]],
    min_samples: int = 2,
) -> float:
    """
    Estimate scale s so that depth_metric = s * depth_relative.

    Uses pinhole: d_geo = (f * H_real) / h_pixel.
    """
    ratios: List[float] = []
    for d_model, box, cls_idx in zip(depth_model_values, boxes, class_indices):
        cls_name = class_names[cls_idx] if isinstance(class_names, dict) else class_names[int(cls_idx)]
        if cls_name not in REAL_HEIGHTS:
            continue
        real_h = REAL_HEIGHTS[cls_name]
        box_h_px = float(box[3] - box[1])
        if box_h_px <= 1.0 or d_model <= 1e-6:
            continue
        d_geo = (focal_length_px * real_h) / box_h_px
        if d_geo > 0:
            ratios.append(d_geo / d_model)

    if len(ratios) >= min_samples:
        return float(np.median(ratios))
    return 1.0


def parse_kitti_focal(calib_path: str, cam: int = 2) -> float:
    """Read focal length from KITTI calib (P2 matrix, fx = P[0,0])."""
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"P{cam}"):
                vals = [float(x) for x in line.split()[1:]]
                p = np.array(vals).reshape(3, 4)
                return float(p[0, 0])
    raise ValueError(f"Cannot find P{cam} in {calib_path}")


def parse_kitti_intrinsic(calib_path: str, cam: int = 2) -> np.ndarray:
    """Return 3x3 K from KITTI P matrix (upper 3x3 of rectified projection)."""
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"P{cam}"):
                vals = [float(x) for x in line.split()[1:]]
                p = np.array(vals).reshape(3, 4)
                k = p[:3, :3]
                return k
    raise ValueError(f"Cannot find P{cam} in {calib_path}")
