"""
DepthAware-Det — Gradio Web UI (image / video / webcam).

Run from project root:
  pip install gradio
  python app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.depth_estimator import create_depth_estimator
from src.detector import create_detector
from src.model_check import normalize_backend, require_backend
from src.pipeline import DepthAwarePipeline
from src.utils import compose_realtime_frame, compose_static_export, draw_detections

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND_MAP = {
    "torch": ("torch", "torch"),
    "onnx": ("onnx", "onnx"),
    "trt": ("trt", "trt"),
}

_state: dict = {
    "backend": None,
    "pipeline": None,
}


def _setup_trt_path() -> None:
    for root in [
        Path(os.environ.get("TRT_ROOT", "")),
        Path(r"D:\Program Files\TensorRT-8.6.1.6"),
    ]:
        if not root.exists():
            continue
        for sub in ("lib", "bin"):
            d = root / sub
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def load_models(backend: str) -> DepthAwarePipeline:
    backend = normalize_backend(backend)
    if backend not in BACKEND_MAP:
        raise ValueError(f"未知后端: {backend}")
    require_backend(backend)
    if _state["pipeline"] is not None and _state["backend"] == backend:
        return _state["pipeline"]

    if _state["pipeline"] is not None:
        _state["pipeline"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    det_b, depth_b = BACKEND_MAP[backend]
    print(f"[app] 加载后端: {backend} (det={det_b}, depth={depth_b})")

    detector = create_detector(
        backend=det_b,
        model_name="yolov8s",
        device=DEVICE,
        imgsz=640,
        conf=0.35,
    )
    depth_est = create_depth_estimator(
        backend=depth_b,
        encoder="vits",
        device=DEVICE,
        input_size=518,
    )
    pipeline = DepthAwarePipeline(
        detector,
        depth_est,
        focal_length=800.0,
        class_names=detector.class_names,
        enable_bev=True,
    )
    _state["backend"] = backend
    _state["pipeline"] = pipeline
    return pipeline


def _infer_frame(
    frame_bgr: np.ndarray,
    backend: str,
    focal: float,
    depth_map_cache: Optional[np.ndarray] = None,
) -> Tuple[list, list, np.ndarray, float, Optional[np.ndarray], float]:
    pipeline = load_models(backend)
    pipeline.focal_length = float(focal)
    t0 = time.perf_counter()
    if depth_map_cache is None:
        result = pipeline.process_frame(frame_bgr)
    else:
        result = pipeline.process_frame(frame_bgr, depth_map=depth_map_cache)
    elapsed = time.perf_counter() - t0
    dets, depths, depth_map, scale, bev = result
    fps = 1.0 / max(elapsed, 1e-6)
    return dets, depths, depth_map, scale, bev, fps


def process_image(
    image: Optional[np.ndarray],
    backend: str,
    focal: float,
    show_depth: bool,
) -> Tuple[Optional[np.ndarray], str]:
    if image is None:
        return None, "请上传图片"
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pipeline = load_models(backend)
    dets, depths, depth_map, scale, bev, fps = _infer_frame(
        frame_bgr, backend, focal
    )
    vis = draw_detections(frame_bgr, dets, depths, pipeline.class_names, ui_scale=1.2)
    display_bgr = compose_static_export(
        vis, depth_map, bev=bev, scale=scale, fps=fps, ui_scale=1.2
    )
    info = (
        f"检测: {len(dets)} | 尺度: {scale:.3f} | "
        f"{1000 / max(fps, 1e-6):.0f} ms ({fps:.1f} FPS) | 后端: {backend}"
    )
    return cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB), info


def process_video(
    video_path: Optional[str],
    backend: str,
    focal: float,
    show_depth: bool,
    depth_every: int,
    max_frames: int,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    if not video_path:
        return None, "请上传视频"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"无法打开视频: {video_path}"

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = max_frames if max_frames > 0 else (total if total > 0 else 10_000)

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
    )

    load_models(backend)
    frame_idx = 0
    processed = 0
    total_time = 0.0
    last_depth = None

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break
        progress(min(1.0, (frame_idx + 1) / limit), desc="处理视频")

        t0 = time.perf_counter()
        if depth_every <= 1 or frame_idx % depth_every == 0 or last_depth is None:
            dets, depths, depth_map, scale, bev, _ = _infer_frame(
                frame, backend, focal
            )
            last_depth = depth_map
        else:
            dets, depths, depth_map, scale, bev, _ = _infer_frame(
                frame, backend, focal, depth_map_cache=last_depth
            )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        vis = draw_detections(
            frame, dets, depths, _state["pipeline"].class_names, ui_scale=1.0
        )
        fps = 1.0 / max(elapsed, 1e-6)
        display_bgr = compose_realtime_frame(
            vis,
            depth_map,
            bev=bev,
            show_depth=show_depth,
            scale=scale,
            fps=fps,
            ui_scale=1.0,
            pip_bev=0.38,
        )
        writer.write(display_bgr)
        frame_idx += 1
        processed += 1

    cap.release()
    writer.release()

    if processed == 0:
        return None, "未读取到有效帧"
    avg_fps = processed / max(total_time, 1e-6)
    info = (
        f"帧数: {processed} | 平均 FPS: {avg_fps:.1f} | "
        f"深度每 {depth_every} 帧 | 后端: {backend}"
    )
    return out_path, info


def process_webcam(
    frame: Optional[np.ndarray],
    backend: str,
    focal: float,
    show_depth: bool,
) -> Tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, "等待摄像头…"
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    pipeline = load_models(backend)
    dets, depths, depth_map, scale, bev, fps = _infer_frame(frame_bgr, backend, focal)
    vis = draw_detections(frame_bgr, dets, depths, pipeline.class_names, ui_scale=1.0)
    display_bgr = compose_realtime_frame(
        vis,
        depth_map,
        bev=bev,
        show_depth=show_depth,
        scale=scale,
        fps=fps,
        ui_scale=1.0,
        pip_bev=0.40,
    )
    info = f"检测: {len(dets)} | 尺度: {scale:.3f} | {fps:.1f} FPS | {backend}"
    return cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB), info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DepthAware-Det", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# DepthAware-Det：单目 3D 感知\n"
            "YOLOv8 检测 + Depth Anything V2，支持 **PyTorch / ONNX / TensorRT**。"
        )

        with gr.Row():
            common_backend = gr.Radio(
                choices=[
                    ("TensorRT（最快）", "trt"),
                    ("ONNX Runtime", "onnx"),
                    ("PyTorch", "torch"),
                ],
                value="trt",
                label="推理后端",
            )
            common_focal = gr.Slider(300, 2000, value=800, step=10, label="焦距 (像素)")
            common_depth = gr.Checkbox(value=True, label="深度热力图（画中画）")

        with gr.Tabs():
            with gr.Tab("图片"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="上传图片", type="numpy")
                        btn_img = gr.Button("开始推理", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(label="结果（检测 + BEV + 深度条）")
                        img_info = gr.Textbox(label="统计", lines=2)
                btn_img.click(
                    process_image,
                    [img_in, common_backend, common_focal, common_depth],
                    [img_out, img_info],
                )

            with gr.Tab("视频"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="上传视频")
                        depth_every = gr.Slider(
                            1, 8, value=1, step=1, label="深度每 N 帧更新"
                        )
                        max_frames = gr.Slider(
                            0, 2000, value=300, step=50,
                            label="最大帧数（0=全部）",
                        )
                        btn_vid = gr.Button("开始处理", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="输出视频")
                        vid_info = gr.Textbox(label="统计", lines=2)
                btn_vid.click(
                    process_video,
                    [
                        vid_in,
                        common_backend,
                        common_focal,
                        common_depth,
                        depth_every,
                        max_frames,
                    ],
                    [vid_out, vid_info],
                )

            with gr.Tab("实时摄像头"):
                gr.Markdown("浏览器需授权摄像头；切换后端后首帧会重新加载模型。")
                cam_in = gr.Image(
                    label="摄像头",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                )
                cam_out = gr.Image(label="输出", type="numpy")
                cam_info = gr.Textbox(label="统计", lines=2)
                cam_in.stream(
                    process_webcam,
                    [cam_in, common_backend, common_focal, common_depth],
                    [cam_out, cam_info],
                    show_progress="hidden",
                )

        gr.Markdown(
            "---\n"
            "TRT → `build_trt.py --all`；ONNX → `export_onnx.py --all`；"
            " PyTorch → `checkpoints/depth_anything_v2_vits.pth`"
        )
    return demo


if __name__ == "__main__":
    _setup_trt_path()
    from src.model_check import format_status_report

    for preferred in ("trt", "onnx", "torch"):
        try:
            load_models(preferred)
            print(f"[app] 预加载: {preferred}")
            break
        except Exception as e:
            print(f"[app] 预加载 {preferred} 失败:\n{e}")
    else:
        print(format_status_report())
        print("[app] 未预加载模型；请在网页中选择可用后端或补齐文件。")

    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    build_ui().launch(server_name=host, server_port=port, share=False)
