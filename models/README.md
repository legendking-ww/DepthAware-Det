# 模型文件（不纳入 Git）

本目录用于存放导出/构建的模型，体积较大，请在本机生成：

```powershell
python scripts/export_onnx.py --all
python scripts/build_trt.py --all
```

| 文件 | 说明 |
|------|------|
| `yolov8s.onnx` | YOLO 导出 |
| `depth_anything_v2_vits.onnx` | 深度 ONNX |
| `yolov8s_fp16.engine` | YOLO TensorRT |
| `depth_anything_v2_vits_fp16.engine` | 深度 TensorRT |

权重：`checkpoints/depth_anything_v2_vits.pth` → `python scripts/download_weights.py`
