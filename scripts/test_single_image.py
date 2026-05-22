"""Quick test on one image."""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.depth_estimator import DepthEstimator
from src.detector import YOLODetector
from src.pipeline import DepthAwarePipeline
from src.paths import OUTPUTS_DIR
from src.utils import compose_static_export, draw_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--focal", type=float, default=800.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    detector = YOLODetector("yolov8s.pt")
    depth_est = DepthEstimator("vits")
    pipeline = DepthAwarePipeline(detector, depth_est, focal_length=args.focal)

    t0 = time.perf_counter()
    dets, depths, depth_map, scale, bev = pipeline.process_frame(img)
    elapsed = time.perf_counter() - t0
    fps = 1.0 / max(elapsed, 1e-6)

    vis = draw_detections(img, dets, depths, detector.class_names, ui_scale=1.2)
    out_img = compose_static_export(vis, depth_map, bev, scale=scale, fps=fps, ui_scale=1.2)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "test_result.jpg"
    cv2.imwrite(str(out_path), out_img)
    print(f"检测数: {len(dets)}, scale={scale:.3f}, FPS={fps:.1f}, 耗时={elapsed:.2f}s")
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
