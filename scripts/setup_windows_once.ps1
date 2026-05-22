# 一次性：把 TensorRT 写入用户环境变量（重启终端后全局生效，无需每次 $env:PATH）
# 以管理员身份运行不是必须；修改的是「用户」级变量。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvFile = Join-Path $Root "config\env.ps1"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Root "config\env.example.ps1") $EnvFile
}
. $EnvFile

if (-not (Test-Path $TRT_ROOT)) {
    Write-Host "TRT 目录不存在: $TRT_ROOT" -ForegroundColor Red
    Write-Host "请先编辑 config\env.ps1 中的 TRT_ROOT。"
    exit 1
}

[Environment]::SetEnvironmentVariable("TRT_ROOT", $TRT_ROOT, "User")

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$add = @("$TRT_ROOT\lib", "$TRT_ROOT\bin")
$changed = $false
foreach ($dir in $add) {
    if ($userPath -notlike "*$dir*") {
        $userPath = "$dir;$userPath"
        $changed = $true
        Write-Host "已加入用户 PATH: $dir"
    }
}
if ($changed) {
    [Environment]::SetEnvironmentVariable("Path", $userPath, "User")
} else {
    Write-Host "PATH 中已包含 TensorRT，无需修改。"
}

Write-Host ""
Write-Host "完成。请「关闭并重新打开」PowerShell / 终端后，PATH 才会在新窗口生效。" -ForegroundColor Green
Write-Host "日常使用：双击 启动网页.bat  或运行  .\launch_web.ps1" -ForegroundColor Green
