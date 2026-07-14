param(
    [string]$Architecture = "sm_86",
    [switch]$RunNativeTest,
    [switch]$RunPythonTest
)

$ErrorActionPreference = "Stop"
$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildPath = Join-Path $projectPath "build"
$includePath = Join-Path $projectPath "include"
$dllPath = Join-Path $buildPath "visionflow_cuda.dll"
$libPath = Join-Path $buildPath "visionflow_cuda.lib"
$testPath = Join-Path $buildPath "test_cuda_api.exe"

New-Item -ItemType Directory -Force -Path $buildPath | Out-Null

$common = @(
    "-std=c++17",
    "-O2",
    "--cudart", "static",
    "-arch=$Architecture",
    "-Xcompiler=/MD,/EHsc",
    "-I$includePath"
)

& nvcc @common --shared `
    (Join-Path $projectPath "visionflow_cuda.cu") `
    -o $dllPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path $libPath)) {
    throw "Import library was not generated: $libPath"
}

& nvcc @common `
    (Join-Path $projectPath "test_cuda_api.cu") `
    $libPath `
    -o $testPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built: $dllPath"
Write-Host "Built: $libPath"
Write-Host "Built: $testPath"

if ($RunNativeTest) {
    & $testPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($RunPythonTest) {
    python -m pip install -r (Join-Path $projectPath "requirements-test.txt")
    python (Join-Path $projectPath "validate_cuda_dll.py") --dll $dllPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
