"""Main inference pipeline: detect -> depth -> fuse -> calibrate -> BEV."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.bev import bottom_center, project_to_bev, render_bev
from src.calibration import estimate_scale_factor
from src.fusion import get_object_depth


class DepthAwarePipeline:
    def __init__(
        self,
        detector,
        depth_estimator,
        focal_length: Optional[float] = 800.0,
        class_names: Optional[Union[Mapping[int, str], list]] = None,
        enable_bev: bool = True,
        k_matrix: Optional[np.ndarray] = None,
        min_calib_objects: int = 2,
    ):
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.focal_length = focal_length
        self.class_names = class_names or getattr(detector, "class_names", {})
        self.enable_bev = enable_bev
        self.min_calib_objects = min_calib_objects
        self.k_matrix = k_matrix
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None

    def _update_intrinsics(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if self.k_matrix is not None:
            k = self.k_matrix
            self._fx = float(k[0, 0])
            self._fy = float(k[1, 1])
            self._cx = float(k[0, 2])
            self._cy = float(k[1, 2])
        elif self.focal_length is not None:
            self._fx = self._fy = float(self.focal_length)
            self._cx = w / 2.0
            self._cy = h / 2.0
        else:
            self._fx = self._fy = self._cx = self._cy = None

    def process_frame(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[
        List[List[float]],
        List[Optional[float]],
        np.ndarray,
        float,
        Optional[np.ndarray],
    ]:
        self._update_intrinsics(image)
        detections = self.detector.detect(image)
        if depth_map is None:
            depth_map = self.depth_estimator.infer(image)
        else:
            depth_map = depth_map.astype(np.float32)

        raw_depths: List[Optional[float]] = []
        for det in detections:
            raw_depths.append(get_object_depth(depth_map, det))

        scale = 1.0
        if self.focal_length is not None:
            valid = [(d, det) for d, det in zip(raw_depths, detections) if d is not None]
            if len(valid) >= self.min_calib_objects:
                valid_raw = [v[0] for v in valid]
                valid_boxes = [v[1][:4] for v in valid]
                valid_cls = [int(v[1][5]) for v in valid]
                scale = estimate_scale_factor(
                    valid_raw,
                    valid_boxes,
                    valid_cls,
                    self.focal_length,
                    self.class_names,
                    min_samples=self.min_calib_objects,
                )

        final_depths: List[Optional[float]] = []
        for d in raw_depths:
            final_depths.append(d * scale if d is not None else None)

        bev_img = None
        if self.enable_bev and self._fx is not None:
            objects_3d = []
            for det, depth in zip(detections, final_depths):
                if depth is None or depth <= 0:
                    continue
                u, v = bottom_center(det[:4])
                cls = int(det[5])
                name = self.class_names[cls]
                x_g, z_g = project_to_bev(
                    u, v, depth, self._fx, self._fy, self._cx, self._cy
                )
                objects_3d.append((x_g, z_g, name))
            bev_img = render_bev(objects_3d)

        return detections, final_depths, depth_map, scale, bev_img
