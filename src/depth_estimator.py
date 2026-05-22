"""Depth Anything V2 — PyTorch, ONNX Runtime & TensorRT backends."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch

from src.paths import CHECKPOINTS_DIR, MODELS_DIR, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "缺少 depth_anything_v2 包。请运行: python scripts/setup_depth_module.py"
    ) from e

EncoderType = Literal["vits", "vitb"]

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
}


def _build_depth_transform(input_size: int):
    from torchvision.transforms import Compose

    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

    return Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )


def preprocess_depth_bgr(image_bgr: np.ndarray, input_size: int = 518) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Same preprocessing as DepthAnythingV2.infer_image (RGB normalized CHW)."""
    h, w = image_bgr.shape[:2]
    transform = _build_depth_transform(input_size)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    tensor = transform({"image": image})["image"]
    return tensor.astype(np.float32), (h, w)


def _fit_chw_to_size(tensor_chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize CHW tensor to fixed HxW (ONNX export uses static input shape)."""
    c, in_h, in_w = tensor_chw.shape
    if in_h == out_h and in_w == out_w:
        return tensor_chw
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(tensor_chw[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return out


def _onnx_input_hw(session) -> Tuple[int, int]:
    shape = session.get_inputs()[0].shape
    h = int(shape[2]) if isinstance(shape[2], int) else 518
    w = int(shape[3]) if isinstance(shape[3], int) else 518
    return h, w


_TRT_SEARCH = [
    Path(os.environ.get("TRT_ROOT", "")),
    Path(r"D:\Program Files\TensorRT-8.6.1.6"),
]


def _setup_trt_path() -> None:
    for root in _TRT_SEARCH:
        if not root.exists():
            continue
        for sub in ("lib", "bin"):
            d = root / sub
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class DepthEstimator:
    def __init__(
        self,
        encoder: EncoderType = "vits",
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        input_size: int = 518,
    ):
        self.device = device
        self.input_size = input_size
        ckpt_dir = checkpoint_dir or CHECKPOINTS_DIR
        weight_path = ckpt_dir / f"depth_anything_v2_{encoder}.pth"
        if not weight_path.is_file():
            raise FileNotFoundError(f"权重不存在: {weight_path}")

        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    @torch.no_grad()
    def infer(self, image: np.ndarray) -> np.ndarray:
        depth = self.model.infer_image(image, self.input_size)
        return depth.astype(np.float32)


class DepthEstimatorONNX:
    """ONNX Runtime GPU backend with official preprocessing."""

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        encoder: EncoderType = "vits",
        input_size: int = 518,
    ):
        import onnxruntime as ort

        path = Path(onnx_path) if onnx_path else MODELS_DIR / f"depth_anything_v2_{encoder}.onnx"
        if not path.is_file():
            raise FileNotFoundError(
                f"ONNX 不存在: {path}\n请先运行: python scripts/export_onnx.py --all"
            )

        self.input_size = input_size
        providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in providers
        self.session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.onnx_h, self.onnx_w = _onnx_input_hw(self.session)
        print(
            f"Depth ONNX: {path.name} | input {self.onnx_h}x{self.onnx_w} | "
            f"providers: {self.session.get_providers()}"
        )

    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tensor, _ = preprocess_depth_bgr(image, self.input_size)
        tensor = _fit_chw_to_size(tensor, self.onnx_h, self.onnx_w)
        depth = self.session.run(None, {self.input_name: tensor[np.newaxis, ...]})[0]
        depth = np.squeeze(depth).astype(np.float32)
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


class DepthEstimatorTRT:
    """TensorRT engine backend (same preprocessing as ONNX export)."""

    def __init__(
        self,
        engine_path: Optional[Path] = None,
        encoder: EncoderType = "vits",
        input_size: int = 518,
        device: str = "cuda",
    ):
        _setup_trt_path()
        import tensorrt as trt

        path = (
            Path(engine_path)
            if engine_path
            else MODELS_DIR / f"depth_anything_v2_{encoder}_fp16.engine"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine 不存在: {path}\n"
                "请先运行: python scripts/build_trt.py --depth --simplify"
            )

        self.input_size = input_size
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Depth TensorRT 需要 CUDA")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"无法加载 engine: {path}")

        self.context = engine.create_execution_context()
        self.input_name = None
        self.output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                in_shape = tuple(engine.get_tensor_shape(name))
            else:
                self.output_name = name
                out_shape = tuple(engine.get_tensor_shape(name))

        self.onnx_h, self.onnx_w = int(in_shape[2]), int(in_shape[3])
        self.input_buf = torch.empty(in_shape, dtype=torch.float32, device=self.device)
        self.output_buf = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(self.input_name, self.input_buf.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buf.data_ptr())
        self.stream = torch.cuda.Stream()
        print(
            f"Depth TRT: {path.name} | input {self.onnx_h}x{self.onnx_w} | "
            f"device {torch.cuda.get_device_name(self.device)}"
        )

    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tensor, _ = preprocess_depth_bgr(image, self.input_size)
        tensor = _fit_chw_to_size(tensor, self.onnx_h, self.onnx_w)
        self.input_buf.copy_(
            torch.from_numpy(tensor[np.newaxis, ...]).to(device=self.device, non_blocking=True)
        )
        ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self.stream.synchronize()
        depth = self.output_buf.squeeze().detach().cpu().numpy().astype(np.float32)
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def create_depth_estimator(
    backend: str = "torch",
    encoder: EncoderType = "vits",
    device: str = "cuda",
    input_size: int = 518,
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
):
    if backend == "trt":
        return DepthEstimatorTRT(
            engine_path=Path(engine_path) if engine_path else None,
            encoder=encoder,
            input_size=input_size,
            device=device,
        )
    if backend == "onnx":
        return DepthEstimatorONNX(
            onnx_path=Path(onnx_path) if onnx_path else None,
            encoder=encoder,
            input_size=input_size,
        )
    return DepthEstimator(encoder=encoder, device=device, input_size=input_size)
