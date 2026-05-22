# DepthAware-Det

联合 **YOLOv8 目标检测** 与 **Depth Anything V2 单目深度估计** 的 3D 感知系统：检测框内深度提取、几何尺度校准、鸟瞰图（BEV）可视化，支持 ONNX / TensorRT 加速与实时摄像头 Demo。

## 项目结构

```
DepthAware-Det/
├── checkpoints/          # depth_anything_v2_vits.pth
├── models/               # 导出的 ONNX / TensorRT engine
├── depth_anything_v2/    # 从官方仓库复制（见下方）
├── src/
│   ├── detector.py
│   ├── depth_estimator.py
│   ├── fusion.py
│   ├── calibration.py
│   ├── bev.py
│   ├── pipeline.py
│   └── utils.py
├── scripts/
│   ├── download_weights.py
│   ├── test_single_image.py
│   ├── run_realtime.py
│   ├── export_onnx.py
│   ├── build_trt.py
│   └── eval_kitti.py
├── data/
├── outputs/
└── requirements.txt
```

## 1. 环境搭建（Windows + Anaconda）

```bash
conda create -n depthaware python=3.10 -y
conda activate depthaware

# RTX 4060 / CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install ultralytics opencv-python matplotlib onnx onnxruntime-gpu
```

### Depth Anything V2 代码

```bash
# 在任意目录克隆官方仓库
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

将仓库中的 **`depth_anything_v2` 文件夹** 复制到本项目根目录（与 `src/` 同级），最终路径为：

`DepthAware-Det/depth_anything_v2/dpt.py`

### 下载深度权重

```bash
cd "你的项目根目录"
python scripts/download_weights.py
```

权重保存为 `checkpoints/depth_anything_v2_vits.pth`。YOLOv8s 首次运行会自动下载。

## 2. 快速验证

```bash
# 单张图（将 test.jpg 放到 data/）
python scripts/test_single_image.py --image data/test.jpg

# 摄像头实时（画中画 BEV；d=深度热力图；q=退出）
python scripts/run_realtime.py --source 0 --focal 800

# 保存视频
python scripts/run_realtime.py --source 0 --save outputs/demo.mp4

# ONNX 加速（先导出，再运行；FPS 通常可从 ~5 提升到 ~10–20）
python scripts/export_onnx.py --all
python scripts/run_realtime.py --source "Depth-Anything-V2/assets/examples_video/basketball.mp4" --onnx --save outputs/demo_onnx.mp4

# 进一步提速：每 2 帧算一次深度 + 略小输入
python scripts/run_realtime.py --source 0 --onnx --depth-every 2 --imgsz 512 --depth-size 518
```

### Gradio Web 界面

**日常使用（推荐，无需每次 conda / PATH）：**

- 双击项目根目录 **`启动网页.bat`**
- 或在项目根目录：`.\launch_web.ps1`

**仅需配置一次：**

1. 若路径不同，编辑 `config/env.ps1`（`TRT_ROOT`、可选 `PYTHON_EXE`）。
2. （可选）一劳永逸写入系统用户环境变量，新终端也自带 TensorRT PATH：
   ```powershell
   .\scripts\setup_windows_once.ps1
   ```
   执行后**重新打开**终端；之后即使不用启动脚本，`python app.py` 也能找到 TRT。

浏览器打开 `http://127.0.0.1:7860`：支持**图片 / 视频 / 摄像头**，PyTorch / ONNX / TensorRT 可切换。

**环境与模型自检：**

```powershell
python scripts/check_env.py
```

## 项目现状（能力清单）

| 能力 | 状态 |
|------|------|
| YOLOv8 + Depth V2 融合管线 | 完成 |
| 尺度校准 + BEV 可视化 | 完成 |
| PyTorch / ONNX / TensorRT 三后端 | 完成 |
| ONNX 导出 + TRT engine 构建 | 完成（本机双 engine） |
| Gradio Web（图/视频/摄像头） | 完成 |
| 一键启动 `启动网页.bat` | 完成 |
| 启动前模型检查 `check_env.py` | 完成 |
| KITTI 评估 `--backend trt/onnx/torch` | 完成（需自备 `data/kitti`） |
| 自动化单元测试 / CI | 未做 |

## 3. 核心流程

| 步骤 | 模块 | 说明 |
|------|------|------|
| 检测 | `src/detector.py` | YOLOv8 → `[x1,y1,x2,y2,conf,cls]` |
| 深度 | `src/depth_estimator.py` | 官方 `infer_image`（ImageNet 归一化 + 518 输入） |
| 融合 | `src/fusion.py` | 框中心 50% 区域 + IQR 中值深度 |
| 校准 | `src/calibration.py` | 已知物体高度 + 焦距 → 米制尺度 |
| BEV | `src/bev.py` | 底边中心反投影到地面 |
| 管道 | `src/pipeline.py` | 串联上述模块 |

**焦距**：普通摄像头可先用 `--focal 800`；KITTI 从 `calib` 读取 `P2[0,0]`。

## 4. ONNX 加速（推荐）

```bash
# 1. 导出到 models/
python scripts/export_onnx.py --all
# 生成 models/yolov8s.onnx  models/depth_anything_v2_vits.onnx

# 2. 实时 ONNX 推理
python scripts/run_realtime.py --source 0 --onnx

# 3. 仍慢时可加（深度每 2 帧更新一次，检测每帧仍更新）
python scripts/run_realtime.py --source 0 --onnx --depth-every 2 --imgsz 512
```

| 参数 | 作用 |
|------|------|
| `--onnx` | YOLO + Depth 均走 ONNX Runtime GPU |
| `--trt` | YOLO + Depth 均走 TensorRT engine（需先 `build_trt.py --all`） |
| `--depth-every 2` | 隔帧复用深度图，显著提速 |
| `--imgsz 512` | 缩小 YOLO 输入 |
| `--depth-size 518` | 深度网络输入（改小需重新 export） |

### TensorRT 构建（Windows）

项目路径含中文时，需先把 PATH 指到 TensorRT 的 **lib+bin**（`nvinfer.dll` 在 `lib` 目录）：

```powershell
$env:PATH = "D:\Program Files\TensorRT-8.6.1.6\lib;D:\Program Files\TensorRT-8.6.1.6\bin;" + $env:PATH

# 或一键脚本
.\scripts\build_trt.ps1

# 手动：先 YOLO（约 4 分钟），再 Depth（可能 10–30 分钟或失败）
python scripts/build_trt.py --yolo
python scripts/build_trt.py --depth --simplify
```

生成：`models/yolov8s_fp16.engine`、`models/depth_anything_v2_vits_fp16.engine`

```powershell
# 全 TensorRT 实时推理（需先 build_trt --all）
python scripts/run_realtime.py --source 0 --trt
```

> Depth 含 `grid_sample` 等算子，TRT 8.6 可能构建失败；失败时继续用 `--onnx` 即可。

## 5. KITTI 评估

将样本放到 `data/kitti/`：

```
data/kitti/image_2/000000.png
data/kitti/calib/000000.txt
data/kitti/velodyne/000000.bin
```

```bash
python scripts/eval_kitti.py --kitti-root data/kitti --limit 50
```

示例输出：

```
With calibration:    {'abs_rel': 0.15, 'rmse': 2.3, 'n': 120}
Without calibration: {'abs_rel': 0.50, 'rmse': 7.8, 'n': 120}
```

## 6. 常见问题

| 问题 | 处理 |
|------|------|
| `No module named depth_anything_v2` | 复制官方 `depth_anything_v2` 到项目根 |
| 权重找不到 | `python scripts/download_weights.py` |
| CUDA OOM | 使用 `vits`、减小输入或 FP16 |
| 深度全白/异常 | 必须使用 `infer_image`，不要仅 `/255` |
| FPS 低 | `export_onnx.py` + ONNX Runtime；或 TensorRT |

## 7. 简历亮点

- 单目深度 + 2D 检测融合，框级相对深度 → 几何校准米制深度  
- 多目标在线尺度估计（中值/RANSAC 可扩展）  
- BEV 鸟瞰可视化  
- 实时 Demo（4060 上 vits + yolov8s 目标 ~20–30 FPS）

## 参考

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
