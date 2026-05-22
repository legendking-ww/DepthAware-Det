"""YOLOv8 detector — PyTorch .pt or ONNX via Ultralytics."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from ultralytics import YOLO

from src.paths import MODELS_DIR

_BACKEND_HINTS = {
    "torch": "使用 yolov8s.pt 或运行 ultralytics 自动下载",
    "onnx": "python scripts/export_onnx.py --yolo",
    "trt": "python scripts/build_trt.py --yolo",
}


def _backend_label(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".onnx":
        return "ONNX Runtime"
    if ext == ".engine":
        return "TensorRT"
    return "PyTorch"


def _resolve_model_path(path: Path, backend: str) -> Path:
    if path.is_file():
        return path
    alt = MODELS_DIR / path.name
    if alt.is_file():
        return alt
    hint = _BACKEND_HINTS.get(backend, "")
    raise FileNotFoundError(
        f"检测模型不存在: {path}\n"
        + (f"（已尝试 {alt}）\n" if alt != path else "")
        + (f"请先: {hint}" if hint else "")
    )


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.35,
    ):
        path = Path(model_path)
        ext = path.suffix.lower()
        backend_key = "onnx" if ext == ".onnx" else ("trt" if ext == ".engine" else "torch")
        path = _resolve_model_path(path, backend_key)
        self.model = YOLO(str(path))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        print(f"YOLO: {path.name} | {_backend_label(path)}")

    def detect(self, image: Union[np.ndarray, str]) -> List[List[float]]:
        results = self.model.predict(
            image,
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: List[List[float]] = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2, float(box.conf.item()), int(box.cls.item())])
        return detections

    @property
    def class_names(self):
        return self.model.names


def create_detector(
    backend: str = "torch",
    model_name: str = "yolov8s",
    device: str = "cuda",
    imgsz: int = 640,
    conf: float = 0.35,
) -> YOLODetector:
    if backend == "trt":
        path = MODELS_DIR / f"{model_name}_fp16.engine"
        _resolve_model_path(path, backend)
    elif backend == "onnx":
        path = MODELS_DIR / f"{model_name}.onnx"
        _resolve_model_path(path, backend)
    else:
        path = Path(f"{model_name}.pt")
    return YOLODetector(str(path), device=device, imgsz=imgsz, conf=conf)
