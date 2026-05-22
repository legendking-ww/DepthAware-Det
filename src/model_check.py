"""Check required model files before inference."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.paths import CHECKPOINTS_DIR, MODELS_DIR

BACKEND_ALIASES = {
    "pytorch": "torch",
    "torch": "torch",
    "onnx": "onnx",
    "trt": "trt",
    "tensorrt": "trt",
}

# (label, path relative to MODELS_DIR or CHECKPOINTS_DIR, root key)
ASSETS: dict[str, list[tuple[str, str, str]]] = {
    "torch": [
        ("深度权重", "depth_anything_v2_vits.pth", "checkpoints"),
    ],
    "onnx": [
        ("YOLO ONNX", "yolov8s.onnx", "models"),
        ("深度 ONNX", "depth_anything_v2_vits.onnx", "models"),
    ],
    "trt": [
        ("YOLO TensorRT", "yolov8s_fp16.engine", "models"),
        ("深度 TensorRT", "depth_anything_v2_vits_fp16.engine", "models"),
    ],
}

HINTS: dict[str, str] = {
    "torch": "python scripts/download_weights.py && 首次运行会自动下载 yolov8s.pt",
    "onnx": "python scripts/export_onnx.py --all",
    "trt": "python scripts/build_trt.py --all  （或 build_trt.ps1）",
}


def normalize_backend(name: str) -> str:
    key = name.strip().lower()
    if key not in BACKEND_ALIASES:
        raise ValueError(f"未知后端: {name}，可选: torch / onnx / trt")
    return BACKEND_ALIASES[key]


def _resolve(rel: str, root_key: str) -> Path:
    base = MODELS_DIR if root_key == "models" else CHECKPOINTS_DIR
    return base / rel


def check_backend(backend: str) -> Tuple[bool, List[str]]:
    """Return (ok, list of error lines in Chinese)."""
    b = normalize_backend(backend)
    errors: List[str] = []
    for label, rel, root_key in ASSETS[b]:
        path = _resolve(rel, root_key)
        if not path.is_file():
            errors.append(f"缺少 {label}: {path}")
    if errors:
        errors.append(f"修复建议: {HINTS[b]}")
        return False, errors
    return True, []


def require_backend(backend: str) -> None:
    ok, errs = check_backend(backend)
    if not ok:
        raise FileNotFoundError("\n".join(errs))


def format_status_report() -> str:
    lines = ["DepthAware-Det 环境与模型检查", "=" * 40]
    for b in ("trt", "onnx", "torch"):
        ok, errs = check_backend(b)
        tag = "OK" if ok else "缺失"
        lines.append(f"[{tag}] {b}")
        for e in errs:
            if not e.startswith("修复"):
                lines.append(f"      - {e}")
    lines.append("=" * 40)
    return "\n".join(lines)
