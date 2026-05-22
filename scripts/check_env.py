"""One-shot environment & model check. Run: python scripts/check_env.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _console_utf8() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    _console_utf8()
    print("=== DepthAware-Det 环境检查 ===\n")

    # Python / CUDA
    import torch

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA: {'可用' if cuda_ok else '不可用'}", end="")
    if cuda_ok:
        print(f" — {torch.cuda.get_device_name(0)}")
    else:
        print()

    # TensorRT
    trt_ok = False
    try:
        import tensorrt as trt

        print(f"TensorRT: {trt.__version__}")
        trt_ok = True
    except ImportError:
        print("TensorRT: 未安装 Python 包（仅 trtexec 也可构建 engine）")

    # ONNX Runtime
    try:
        import onnxruntime as ort

        print(f"ONNX Runtime: {ort.__version__} | {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime: 未安装")

    # Gradio
    try:
        import gradio as gr

        print(f"Gradio: {gr.__version__}")
    except ImportError:
        print("Gradio: 未安装 — pip install gradio")

    # depth module
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: F401

        print("depth_anything_v2: OK")
    except ImportError:
        print("depth_anything_v2: 缺失 — python scripts/setup_depth_module.py")

    print()
    from src.model_check import format_status_report

    print(format_status_report())

    from src.model_check import check_backend

    ready = [b for b in ("trt", "onnx", "torch") if check_backend(b)[0]]
    if ready:
        print(f"\n可立即使用的后端: {', '.join(ready)}")
        print("启动 Web: 双击 启动网页.bat  或  python app.py")
        return 0
    print("\n请先补齐模型文件（见上方缺失项）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
