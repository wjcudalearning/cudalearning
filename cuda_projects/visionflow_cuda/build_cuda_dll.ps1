param(
    [string]$Architecture = "sm_86",
    [string]$OutputDir = "",
    [switch]$BuildSmoke,
    [switch]$RunTests,
    [switch]$AllowNoGpu
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectName = Split-Path -Leaf $PSScriptRoot
if ($ProjectName -ne "visionflow_cuda") {
    throw "This folder must be named 'visionflow_cuda' so the workflow output is visionflow_cuda.dll. Current: $ProjectName"
}
if ($Architecture -notmatch '^sm_\d{2,3}$') {
    throw "Invalid -Architecture '$Architecture'. Example: sm_86"
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
$cl = Get-Command cl -ErrorAction SilentlyContinue
$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $nvcc) { throw "nvcc not found. Install CUDA Toolkit and reopen an x64 Native Tools PowerShell." }
if (-not $cl) { throw "cl.exe not found. Use an x64 Native Tools PowerShell." }
if (-not $dumpbin) { throw "dumpbin.exe not found. Install Visual Studio C++ Build Tools." }
if (-not $python) { throw "python not found. Python is required for the preflight check." }

$source = Join-Path $PSScriptRoot "visionflow_cuda.cu"
$smokeSource = Join-Path $PSScriptRoot "tests\test_cuda_api.cu"
$dllSources = @($source)
$smokeSources = @($smokeSource)

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "build"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Get-Location) $OutputDir
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$dllPath = Join-Path $OutputDir "$ProjectName.dll"
$libPath = Join-Path $OutputDir "$ProjectName.lib"
$smokePath = Join-Path $OutputDir "test_cuda_api.exe"
$manifestPath = Join-Path $OutputDir "preflight.json"

& $python.Source (Join-Path $PSScriptRoot "preflight_cuda_build.py") --output $manifestPath
if ($LASTEXITCODE -ne 0) { throw "CUDA project preflight failed with exit code $LASTEXITCODE" }

Write-Host "Project: $ProjectName"
Write-Host "Architecture: $Architecture"
Write-Host "DLL source: $source"
Write-Host "Output: $dllPath"

& $nvcc.Source `
    "--std=c++17" `
    "-O3" `
    "--shared" `
    "--cudart=static" `
    "-arch=$Architecture" `
    "-I$PSScriptRoot" `
    "-Xcompiler=/MD /EHsc" `
    "-Xlinker" "/IMPLIB:$libPath" `
    "-o" $dllPath `
    $dllSources
if ($LASTEXITCODE -ne 0) { throw "CUDA DLL build failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $dllPath)) { throw "DLL was not created: $dllPath" }

$expectedExports = @((Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json).exports)
$exportsText = (& $dumpbin.Source /exports $dllPath | Out-String)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /exports failed with exit code $LASTEXITCODE" }
foreach ($expectedExport in $expectedExports) {
    if ($exportsText -notmatch "(?m)\b$([regex]::Escape($expectedExport))\b") {
        throw "DLL is missing expected export: $expectedExport"
    }
}

if ($BuildSmoke -or $RunTests) {
    & $nvcc.Source `
        "--std=c++17" `
        "-O2" `
        "-arch=$Architecture" `
        "-I$PSScriptRoot" `
        "-Xcompiler=/MD /EHsc" `
        "-o" $smokePath `
        $smokeSources `
        $libPath
    if ($LASTEXITCODE -ne 0) { throw "Smoke executable build failed with exit code $LASTEXITCODE" }
}

Write-Host "Built: $dllPath"
Write-Host "Built: $libPath"
if (Test-Path -LiteralPath $smokePath) { Write-Host "Built: $smokePath" }
Write-Host "Verified $($expectedExports.Count) exports."

if ($RunTests) {
    if ($AllowNoGpu) {
        & $python.Source (Join-Path $PSScriptRoot "test.py") --dll $dllPath --allow-no-gpu
    } else {
        & $python.Source (Join-Path $PSScriptRoot "test.py") --dll $dllPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Python DLL smoke test failed with exit code $LASTEXITCODE" }
}
