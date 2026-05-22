"""Download Depth Anything V2 Small weights to checkpoints/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINTS = ROOT / "checkpoints"


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub")
        raise

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINTS / "depth_anything_v2_vits.pth"
    if dest.is_file():
        print(f"已存在: {dest}")
        return

    print("正在从 Hugging Face 下载 depth_anything_v2_vits.pth ...")
    path = hf_hub_download(
        repo_id="depth-anything/Depth-Anything-V2-Small",
        filename="depth_anything_v2_vits.pth",
        local_dir=str(CHECKPOINTS),
    )
    print(f"完成: {path}")


if __name__ == "__main__":
    main()
