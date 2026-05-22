# TensorRT engine build helper (Windows)
# Run in depthaware conda env from project root

$TRT = "D:\Program Files\TensorRT-8.6.1.6"
$env:PATH = "$TRT\lib;$TRT\bin;" + $env:PATH
$env:TRT_ROOT = $TRT

Write-Host "TRT PATH set. Building engines (YOLO first, then Depth)..."
python scripts/build_trt.py --yolo
if ($LASTEXITCODE -eq 0) {
    Write-Host "YOLO engine OK. Building Depth (may take 10-30 min or fail on grid_sample)..."
    python scripts/build_trt.py --depth --simplify
}
Write-Host "Check models/*.engine"
