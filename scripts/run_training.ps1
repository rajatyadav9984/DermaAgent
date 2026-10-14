# DermaAgent - PowerShell Training Runner
# Usage: .\scripts\run_training.ps1 [-Epochs 10] [-BatchSize 32] [-LR 0.0001]

param (
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [float]$LR = 0.0001,
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "DermaAgent - ISIC 2019 Training Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment Python not found at: $PythonExe" -ForegroundColor Red
    exit 1
}

# 1. Ensure dataset manifests exist
$TrainManifest = Join-Path $ProjectRoot "data_raw\ISIC2019\isic2019_train_split.csv"
if (!(Test-Path $TrainManifest)) {
    Write-Host "[INFO] Generating ISIC 2019 split manifests..." -ForegroundColor Yellow
    & $PythonExe (Join-Path $ProjectRoot "scripts\prepare_isic2019.py")
}

# 2. Run training script
Write-Host "[INFO] Starting training pipeline with $Epochs epochs..." -ForegroundColor Green
& $PythonExe (Join-Path $ProjectRoot "ml\train_isic2019.py") --epochs $Epochs --batch-size $BatchSize --lr $LR --workers $Workers

Write-Host "`n[SUCCESS] Pipeline script completed." -ForegroundColor Cyan
