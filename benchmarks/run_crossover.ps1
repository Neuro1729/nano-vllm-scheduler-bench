# Crossover helper for Windows / PowerShell.
# Usage:
#   $env:MODEL = "C:\path\to\model"
#   .\benchmarks\run_crossover.ps1
param(
    [string]$Model = $env:MODEL,
    [string]$Workload = "mixed",
    [int]$NumRequests = 256,
    [int]$Seed = 42,
    [int]$Runs = 5
)

if (-not $Model) {
    throw "Set -Model or `$env:MODEL to a local HF model directory"
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "results" | Out-Null

function Invoke-Bench {
    param($Branch, $Gpu, $Label, $Out)
    git checkout $Branch
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    python benchmarks/scheduler_bench.py `
        --workload $Workload `
        --model $Model `
        --num-requests $NumRequests `
        --seed $Seed `
        --runs $Runs `
        --output $Out `
        --experiment-label $Label
}

Write-Host "Sequential crossover runs. For parallel GPU runs, use two terminals instead."

Invoke-Bench real 0 A_real_gpu0 "results/A_real_mixed.json"
Invoke-Bench dev 1 A_dev_gpu1 "results/A_dev_mixed.json"
Invoke-Bench real 1 B_real_gpu1 "results/B_real_mixed.json"
Invoke-Bench dev 0 B_dev_gpu0 "results/B_dev_mixed.json"

Write-Host "Compare matched seeds, e.g.:"
Write-Host "  python benchmarks/compare_results.py results/A_real_mixed_run0_seed$Seed.json results/A_dev_mixed_run0_seed$Seed.json"
