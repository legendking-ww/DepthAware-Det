# 首次推送到 GitHub（需先在网页创建空仓库 DepthAware-Det，不要勾选 README）
# 用法: .\scripts\push_github.ps1 -UserName 你的GitHub用户名

param(
    [Parameter(Mandatory = $true)]
    [string]$UserName,
    [string]$RepoName = "DepthAware-Det"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$remote = "https://github.com/$UserName/$RepoName.git"
Write-Host "Remote: $remote"

$existing = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "已存在 origin: $existing"
} else {
    git remote add origin $remote
}

git branch -M main
git push -u origin main
Write-Host "完成: https://github.com/$UserName/$RepoName"
