"""Fuse detection boxes with depth map."""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

DepthMethod = Literal["median", "median_center", "mean_center"]


def get_object_depth(
    depth_map: np.ndarray,
    box: Sequence[float],
    method: DepthMethod = "median_center",
    center_ratio: float = 0.5,
) -> Optional[float]:
    """
    Extract robust depth inside detection box.

    Args:
        depth_map: (H, W) float depth
        box: [x1, y1, x2, y2, ...]
        method: aggregation strategy
        center_ratio: fraction of box size for central ROI (0.5 = center 50%)
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth_map.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None

    if method in ("median_center", "mean_center"):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w = int((x2 - x1) * center_ratio / 2)
        half_h = int((y2 - y1) * center_ratio / 2)
        x1_c = max(cx - half_w, 0)
        x2_c = min(cx + half_w, w)
        y1_c = max(cy - half_h, 0)
        y2_c = min(cy + half_h, h)
        region = depth_map[y1_c:y2_c, x1_c:x2_c]
    else:
        region = depth_map[y1:y2, x1:x2]

    if region.size == 0:
        return None

    if method == "median_center":
        q1, q3 = np.percentile(region, [25, 75])
        iqr = q3 - q1
        if iqr > 1e-6:
            valid = region[(region >= q1 - 1.5 * iqr) & (region <= q3 + 1.5 * iqr)]
            if valid.size > 0:
                return float(np.median(valid))
        return float(np.median(region))

    if method == "mean_center":
        return float(np.mean(region))

    return float(np.median(region))
