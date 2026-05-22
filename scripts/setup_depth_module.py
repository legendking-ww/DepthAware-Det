"""Copy depth_anything_v2 from Depth-Anything-V2 clone into project root."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "depth_anything_v2"

SOURCES = [
    ROOT / "Depth-Anything-V2" / "depth_anything_v2",
    ROOT.parent / "Depth-Anything-V2" / "depth_anything_v2",
]


def main():
    if DEST.is_dir() and (DEST / "dpt.py").is_file():
        print(f"已存在: {DEST}")
        return

    for src in SOURCES:
        if (src / "dpt.py").is_file():
            if DEST.exists():
                shutil.rmtree(DEST)
            shutil.copytree(src, DEST)
            print(f"已复制: {src} -> {DEST}")
            return

    print("未找到 Depth-Anything-V2。请先执行:")
    print("  git clone https://github.com/DepthAnything/Depth-Anything-V2.git")
    print(f"  放到 {ROOT / 'Depth-Anything-V2'} 后重新运行本脚本")
    sys.exit(1)


if __name__ == "__main__":
    main()
