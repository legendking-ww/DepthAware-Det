"""
Build TensorRT engines from models/*.onnx

TensorRT trtexec cannot handle non-ASCII paths on Windows — ONNX is copied to %TEMP% first.

Usage (depthaware env, PowerShell):
  $env:PATH = "D:\Program Files\TensorRT-8.6.1.6\lib;D:\Program Files\TensorRT-8.6.1.6\bin;" + $env:PATH
  python scripts/build_trt.py --all
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

TRT_SEARCH = [
    Path(os.environ.get("TRT_ROOT", "")),
    Path(r"D:\Program Files\TensorRT-8.6.1.6"),
]


def setup_trt_path() -> None:
    for root in TRT_SEARCH:
        if not root.exists():
            continue
        for sub in ("lib", "bin"):
            d = root / sub
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def find_trtexec() -> Path | None:
    for root in TRT_SEARCH:
        p = root / "bin" / "trtexec.exe"
        if p.is_file():
            return p
    w = shutil.which("trtexec")
    return Path(w) if w else None


def ascii_stage(onnx_path: Path, tag: str) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
    """Copy ONNX to ASCII-only temp dir for trtexec."""
    tmp = tempfile.TemporaryDirectory(prefix="depthaware_trt_")
    stage = Path(tmp.name)
    staged_onnx = stage / f"{tag}.onnx"
    staged_engine = stage / f"{tag}.engine"
    shutil.copy2(onnx_path, staged_onnx)
    return staged_onnx, staged_engine, tmp


def build_with_trtexec(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_mb: int = 4096,
    tag: str = "model",
) -> None:
    staged_onnx, staged_eng, tmp = ascii_stage(onnx_path, tag)
    try:
        cmd = [
            str(trtexec),
            f"--onnx={staged_onnx}",
            f"--saveEngine={staged_eng}",
            f"--memPoolSize=workspace:{workspace_mb}",
        ]
        if fp16:
            cmd.append("--fp16")
        print(">>> trtexec", tag, "(staging in TEMP)...")
        subprocess.run(cmd, check=True)
        shutil.copy2(staged_eng, engine_path)
        print("OK:", engine_path.name, "(in models/)")
    finally:
        tmp.cleanup()


def build_with_python_api(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_gb: int = 4,
    tag: str = "model",
) -> None:
    setup_trt_path()
    import tensorrt as trt

    staged_onnx, staged_eng, tmp = ascii_stage(onnx_path, tag)
    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(staged_onnx, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build failed")
        with open(staged_eng, "wb") as f:
            f.write(serialized)
        shutil.copy2(staged_eng, engine_path)
        print("OK:", engine_path.name)
    finally:
        tmp.cleanup()


def simplify_onnx(onnx_path: Path) -> Path:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        return onnx_path
    out = onnx_path.with_suffix(".sim.onnx")
    print("onnxsim:", onnx_path.name)
    model_simp, ok = simplify(onnx.load(str(onnx_path)))
    if ok:
        onnx.save(model_simp, str(out))
        return out
    return onnx_path


def build_one(name, onnx_name, engine_name, fp16, method, do_sim, workspace_mb):
    onnx_path = MODELS / onnx_name
    engine_path = MODELS / engine_name
    if not onnx_path.is_file():
        print("[skip]", name, "- missing", onnx_name)
        return False
    src = simplify_onnx(onnx_path) if do_sim else onnx_path
    tag = onnx_name.replace(".", "_")
    print("\n===", name, "===")
    try:
        if method == "python":
            build_with_python_api(src, engine_path, fp16, workspace_mb // 1024, tag)
        else:
            trtexec = find_trtexec()
            if trtexec is None:
                raise FileNotFoundError("trtexec not found")
            build_with_trtexec(trtexec, src, engine_path, fp16, workspace_mb, tag)
        return True
    except Exception as e:
        print("[FAIL]", name, ":", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--simplify", action="store_true", help="run onnxsim before build")
    parser.add_argument("--method", choices=["trtexec", "python"], default="trtexec")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    args = parser.parse_args()

    if not (args.yolo or args.depth or args.all):
        parser.error("use --yolo / --depth / --all")

    setup_trt_path()
    fp16 = not args.no_fp16
    jobs = []
    if args.all or args.yolo:
        jobs.append(("YOLOv8s", "yolov8s.onnx", "yolov8s_fp16.engine"))
    if args.all or args.depth:
        jobs.append(("Depth-V2", "depth_anything_v2_vits.onnx", "depth_anything_v2_vits_fp16.engine"))

    ok = 0
    for j in jobs:
        if build_one(j[0], j[1], j[2], fp16, args.method, args.simplify, args.workspace_mb):
            ok += 1
    print("\nDone:", ok, "/", len(jobs))
    if ok < len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
