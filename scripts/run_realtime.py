"""Realtime demo — use --onnx for ONNX Runtime acceleration."""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.depth_estimator import create_depth_estimator
from src.detector import create_detector
from src.pipeline import DepthAwarePipeline
from src.utils import compose_realtime_frame, draw_detections, upscale_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--yolo", default="yolov8s", help="model stem, e.g. yolov8s")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb"])
    parser.add_argument("--focal", type=float, default=800.0)
    parser.add_argument("--no-bev", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="YOLO+Depth ONNX Runtime",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="YOLO + Depth 均用 TensorRT engine (scripts/build_trt.py --all)",
    )
    parser.add_argument(
        "--depth-every",
        type=int,
        default=1,
        help="run depth every N frames (reuse last depth map, faster)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--depth-size", type=int, default=518, help="depth network input")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.35,
        help="HUD/标签/画中画放大倍数，默认 1.35",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="输出画面宽度（放大便于观看/录屏，0=不放大）",
    )
    parser.add_argument("--pip-bev", type=float, default=0.46, help="BEV 小窗相对屏幕比例")
    args = parser.parse_args()

    if args.trt and args.onnx:
        parser.error("use only one of --trt or --onnx")
    if args.trt:
        det_backend, depth_backend = "trt", "trt"
    elif args.onnx:
        det_backend, depth_backend = "onnx", "onnx"
    else:
        det_backend, depth_backend = "torch", "torch"
    device = "cuda"

    detector = create_detector(
        backend=det_backend,
        model_name=args.yolo,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    depth_est = create_depth_estimator(
        backend=depth_backend,
        encoder=args.encoder,
        device=device,
        input_size=args.depth_size,
    )
    pipeline = DepthAwarePipeline(
        detector,
        depth_est,
        focal_length=args.focal,
        class_names=detector.class_names,
        enable_bev=not args.no_bev,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w
    out_h = h
    if args.display_width > 0 and w < args.display_width:
        out_w = args.display_width
        out_h = int(h * out_w / w)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (out_w, out_h))

    cv2.namedWindow("DepthAware-Det", cv2.WINDOW_NORMAL)
    mode = "TensorRT" if args.trt else ("ONNX" if args.onnx else "PyTorch")
    print(f"模式: {mode} | depth每{args.depth_every}帧 | [d]深度 [q]退出")

    last_depth = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        if args.depth_every <= 1 or frame_idx % args.depth_every == 0 or last_depth is None:
            last_depth = depth_est.infer(frame)
        dets, depths, depth_map, scale, bev = pipeline.process_frame(frame, depth_map=last_depth)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        frame_idx += 1

        vis = draw_detections(
            frame, dets, depths, detector.class_names, ui_scale=args.ui_scale
        )
        display = compose_realtime_frame(
            vis,
            depth_map,
            bev=bev,
            show_depth=args.show_depth,
            scale=scale,
            fps=fps,
            ui_scale=args.ui_scale,
            pip_bev=args.pip_bev,
        )
        display = upscale_frame(display, args.display_width)

        cv2.imshow("DepthAware-Det", display)
        if writer:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            args.show_depth = not args.show_depth

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
