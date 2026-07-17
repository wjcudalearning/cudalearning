param(
    [string]$Architecture = "sm_86",
    [string]$OutputDirectory = "",
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Architecture -notmatch '^sm_\d{2,3}$') {
    throw "Invalid -Architecture '$Architecture'. Expected a value such as sm_86."
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "nvcc not found. Install CUDA Toolkit and reopen an x64 Native Tools PowerShell."
}
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    throw "MSVC cl.exe not found. Run this script from an x64 Native Tools PowerShell."
}
$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
if (-not $dumpbin) {
    throw "dumpbin.exe not found. Install VS 2022 C++ Build Tools and use an x64 Native Tools PowerShell."
}
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python not found. Python is required for the static build preflight."
}

$projectRoot = $PSScriptRoot
$includeDirectory = Join-Path $projectRoot "include"
$dllSource = Join-Path $projectRoot "visionflow_cuda.cu"
$smokeSource = Join-Path $projectRoot "test_cuda_api.cu"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $outputRoot = Join-Path $projectRoot "build"
} else {
    $outputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
}

$stageDirectory = Join-Path $outputRoot ".stage"
$outputDll = Join-Path $outputRoot "visionflow_cuda.dll"
$outputLib = Join-Path $outputRoot "visionflow_cuda.lib"
$outputSmokeExe = Join-Path $outputRoot "test_cuda_api.exe"
$preflightManifest = Join-Path $outputRoot "cuda_build_preflight.json"
$stageDll = Join-Path $stageDirectory "visionflow_cuda.dll"
$stageLib = Join-Path $stageDirectory "visionflow_cuda.lib"
$stageSmokeExe = Join-Path $stageDirectory "test_cuda_api.exe"

# Explicit manifests are intentional. test_cuda_api.cu must never be linked into the DLL.
$dllSources = @($dllSource)
$smokeSources = @($smokeSource)

New-Item -ItemType Directory -Force -Path $outputRoot, $stageDirectory | Out-Null
foreach ($artifact in @($stageDll, $stageLib, $stageSmokeExe)) {
    if (Test-Path -LiteralPath $artifact) {
        Remove-Item -LiteralPath $artifact -Force
    }
}

& $pythonCommand.Source (Join-Path $projectRoot "preflight_cuda_build.py") --output $preflightManifest
if ($LASTEXITCODE -ne 0) {
    throw "CUDA source/API preflight failed with exit code $LASTEXITCODE"
}

Write-Host "Project root: $projectRoot"
Write-Host "Output root: $outputRoot"
Write-Host "nvcc: $($nvcc.Source)"
Write-Host "cl: $($cl.Source)"
Write-Host "architecture: $Architecture"

& $nvcc.Source `
    "--std=c++17" `
    "-O3" `
    "--shared" `
    "--cudart=static" `
    "-arch=$Architecture" `
    "-I$includeDirectory" `
    "-Xcompiler=/MD,/EHsc" `
    "-Xlinker" "/IMPLIB:$stageLib" `
    "-o" $stageDll `
    $dllSources
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DLL build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $stageDll)) {
    throw "nvcc returned success but DLL was not created: $stageDll"
}
if (-not (Test-Path -LiteralPath $stageLib)) {
    throw "DLL import library was not created: $stageLib"
}

& $nvcc.Source `
    "--std=c++17" `
    "-O2" `
    "-arch=$Architecture" `
    "-I$includeDirectory" `
    "-Xcompiler=/MD,/EHsc" `
    "-o" $stageSmokeExe `
    $smokeSources `
    $stageLib
if ($LASTEXITCODE -ne 0) {
    throw "CUDA C ABI smoke executable build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $stageSmokeExe)) {
    throw "nvcc returned success but smoke executable was not created: $stageSmokeExe"
}

$preflight = Get-Content -LiteralPath $preflightManifest -Raw | ConvertFrom-Json
$expectedExports = @($preflight.exports)
$exports = (& $dumpbin.Source /exports $stageDll | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /exports failed with exit code $LASTEXITCODE"
}
foreach ($expectedExport in $expectedExports) {
    if ($exports -notmatch "(?m)\b$([regex]::Escape($expectedExport))\b") {
        throw "CUDA DLL is missing expected export: $expectedExport"
    }
}

$dependents = (& $dumpbin.Source /dependents $stageDll | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /dependents failed with exit code $LASTEXITCODE"
}

Move-Item -LiteralPath $stageDll -Destination $outputDll -Force
Move-Item -LiteralPath $stageLib -Destination $outputLib -Force
Move-Item -LiteralPath $stageSmokeExe -Destination $outputSmokeExe -Force
Remove-Item -LiteralPath $stageDirectory -Recurse -Force

Write-Host "Built CUDA DLL: $outputDll"
Write-Host "Built import library: $outputLib"
Write-Host "Built native smoke executable: $outputSmokeExe"
Write-Host "Verified exports: $($expectedExports -join ', ')"
Write-Host "DLL dependencies:"
Write-Host $dependents.Trim()

if ($RunTests) {
    Write-Host "Running GPU smoke test. This requires an actual NVIDIA GPU and compatible driver."
    & $outputSmokeExe
    if ($LASTEXITCODE -ne 0) {
        throw "C ABI smoke test failed with exit code $LASTEXITCODE"
    }
}
