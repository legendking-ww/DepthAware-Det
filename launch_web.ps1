# 一键启动 Gradio Web 界面（自动设置 TRT 路径 + conda 环境）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$EnvFile = Join-Path $Root "config\env.ps1"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Root "config\env.example.ps1") $EnvFile
    Write-Host "已生成 config\env.ps1 ，请按需修改路径后再次运行。" -ForegroundColor Yellow
}
. $EnvFile

if ($TRT_ROOT -and (Test-Path $TRT_ROOT)) {
    $env:TRT_ROOT = $TRT_ROOT
    $env:PATH = "$TRT_ROOT\lib;$TRT_ROOT\bin;" + $env:PATH
}
$env:GRADIO_HOST = $GRADIO_HOST
$env:GRADIO_PORT = "$GRADIO_PORT"

function Find-Python {
    if ($PYTHON_EXE -and (Test-Path $PYTHON_EXE)) { return $PYTHON_EXE }
    $candidates = @(
        "$env:USERPROFILE\anaconda3\envs\$CONDA_ENV\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$CONDA_ENV\python.exe",
        "D:\App\anaconda3\envs\$CONDA_ENV\python.exe",
        "C:\ProgramData\anaconda3\envs\$CONDA_ENV\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        $py = & conda run -n $CONDA_ENV python -c "import sys; print(sys.executable)" 2>$null
        if ($py -and (Test-Path $py.Trim())) { return $py.Trim() }
    }
    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Host "找不到 conda 环境 '$CONDA_ENV' 的 Python。请在 config\env.ps1 中设置 PYTHON_EXE。" -ForegroundColor Red
    exit 1
}

Write-Host "Python: $Python"
& $Python scripts/check_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "环境/模型未就绪，请按上方提示补齐后再启动。" -ForegroundColor Yellow
    pause
    exit $LASTEXITCODE
}

Write-Host "启动 Web 界面 http://${GRADIO_HOST}:${GRADIO_PORT} ..."
Write-Host "关闭本窗口即停止服务。" -ForegroundColor Cyan
& $Python app.py
