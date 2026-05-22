"""Bird's-eye view projection and rendering."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# BGR per class (fallback gray)
CLASS_COLORS: dict = {
    "person": (255, 128, 0),
    "car": (0, 180, 255),
    "bus": (0, 0, 255),
    "truck": (128, 0, 255),
    "motorcycle": (0, 255, 128),
    "bicycle": (0, 200, 100),
}


def project_to_bev(
    u: float,
    v: float,
    depth_z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float]:
    """
    Pinhole: lateral X = (u-cx)*Z/fx, forward Z = depth (metric after calibration).
    Uses bottom contact point (u,v) and calibrated depth as forward distance.
    """
    if depth_z <= 1e-3 or fx <= 1e-3:
        return 0.0, 0.0
    x_lateral = (u - cx) * depth_z / fx
    z_forward = depth_z
    return float(x_lateral), float(z_forward)


def bottom_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, float(y2)


def _auto_ranges(
    objects_3d: List[Tuple[float, float, str]],
    default_x: Tuple[float, float] = (-15, 15),
    default_z: Tuple[float, float] = (0, 60),
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not objects_3d:
        return default_x, default_z
    xs = [o[0] for o in objects_3d]
    zs = [o[1] for o in objects_3d]
    mx = max(3.0, (max(xs) - min(xs)) * 0.15)
    mz = max(5.0, (max(zs) - min(zs)) * 0.15)
    range_x = (min(xs) - mx, max(xs) + mx)
    range_z = (max(0.0, min(zs) - 2.0), max(zs) + mz)
    if range_x[1] - range_x[0] < 2.0:
        mid = (range_x[0] + range_x[1]) / 2
        range_x = (mid - 5.0, mid + 5.0)
    if range_z[1] - range_z[0] < 5.0:
        range_z = (0.0, max(range_z[1], 20.0))
    return range_x, range_z


def render_bev(
    objects_3d: List[Tuple[float, float, str]],
    size: int = 560,
    range_x: Optional[Tuple[float, float]] = None,
    range_z: Optional[Tuple[float, float]] = None,
    ego_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """BEV canvas: X = lateral (left/right), Z = forward (up). Ego at bottom center."""
    if range_x is None or range_z is None:
        auto_x, auto_z = _auto_ranges(objects_3d)
        range_x = range_x or auto_x
        range_z = range_z or auto_z

    bev = np.ones((size, size, 3), dtype=np.uint8) * 255
    rx = range_x[1] - range_x[0]
    rz = range_z[1] - range_z[0]
    dot_r = max(8, size // 70)
    font = max(0.45, size / 900.0)
    font_th = max(1, size // 280)

    for z_m in range(0, int(range_z[1]) + 1, 10):
        pz = int((z_m - range_z[0]) / rz * size)
        pz = size - 1 - pz
        if 0 <= pz < size:
            cv2.line(bev, (0, pz), (size, pz), (230, 230, 230), max(1, size // 400))

    for x, z, cls in objects_3d:
        px = int((x - range_x[0]) / rx * (size - 1))
        pz = int((z - range_z[0]) / rz * (size - 1))
        pz = size - 1 - pz
        color = CLASS_COLORS.get(cls, (0, 0, 200))
        if 0 <= px < size and 0 <= pz < size:
            cv2.circle(bev, (px, pz), dot_r, color, -1)
            cv2.circle(bev, (px, pz), dot_r, (0, 0, 0), 2)
            cv2.putText(
                bev, cls[:8], (px + dot_r + 4, pz + 6),
                cv2.FONT_HERSHEY_SIMPLEX, font, (30, 30, 30), font_th, cv2.LINE_AA,
            )

    ego_px, ego_pz = size // 2, size - max(24, size // 22)
    ego_r = max(10, size // 55)
    cv2.circle(bev, (ego_px, ego_pz), ego_r, ego_color, -1)
    cv2.putText(
        bev, "EGO", (ego_px - 22, ego_pz + ego_r + 22),
        cv2.FONT_HERSHEY_SIMPLEX, font * 1.1, (0, 0, 0), font_th + 1, cv2.LINE_AA,
    )
    cv2.putText(
        bev, f"X[{range_x[0]:.0f},{range_x[1]:.0f}]m  Z[{range_z[0]:.0f},{range_z[1]:.0f}]m",
        (8, int(28 * font + 10)), cv2.FONT_HERSHEY_SIMPLEX, font * 0.9, (80, 80, 80), font_th, cv2.LINE_AA,
    )
    return bev


# backward compat
def project_to_ground(u, v, depth, k_inv):
    """Legacy wrapper — prefer project_to_bev with fx,cx from image size."""
    fx = 1.0 / k_inv[0, 0]
    fy = 1.0 / k_inv[1, 1]
    cx = -k_inv[0, 2] / k_inv[0, 0] if abs(k_inv[0, 0]) > 1e-6 else 0.0
    cy = -k_inv[1, 2] / k_inv[1, 1] if abs(k_inv[1, 1]) > 1e-6 else 0.0
    return project_to_bev(u, v, depth, fx, fy, cx, cy)
