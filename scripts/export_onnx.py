"""Export YOLOv8 and Depth Anything V2 to models/*.onnx"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.depth_estimator import MODEL_CONFIGS  # noqa: E402
from src.paths import CHECKPOINTS_DIR, MODELS_DIR  # noqa: E402

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def export_yolo(model_name: str = "yolov8s.pt", imgsz: int = 640):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pt = Path(model_name)
    if not pt.is_file():
        pt = ROOT / model_name
    model = YOLO(str(pt))
    out = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True))
    dest = MODELS_DIR / f"{pt.stem}.onnx"
    if out.resolve() != dest.resolve():
        shutil.copy2(out, dest)
    print(f"YOLO ONNX -> {dest}")


def export_depth(encoder: str = "vits", input_size: int = 518):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS_DIR / f"depth_anything_v2_{encoder}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少权重: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    out_path = MODELS_DIR / f"depth_anything_v2_{encoder}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["depth"],
        opset_version=12,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f"Depth ONNX -> {out_path} (input {input_size}x{input_size})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--depth-size", type=int, default=518)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.yolo:
        export_yolo(imgsz=args.imgsz)
    if args.all or args.depth:
        export_depth(args.encoder, args.depth_size)


if __name__ == "__main__":
    main()
