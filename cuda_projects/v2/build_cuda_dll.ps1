param(
    [string]$Architecture = "sm_86",
    [switch]$RunTests,
    [string]$Image = "",
    [string]$Recipe = "",
    [int]$Benchmark = 20
)

$ErrorActionPreference = "Stop"

if ([bool]$Image -ne [bool]$Recipe) {
    throw "-Image and -Recipe must be provided together."
}
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

$root = Split-Path -Parent $PSScriptRoot
$include = Join-Path $PSScriptRoot "include"
$source = Join-Path $PSScriptRoot "visionflow_cuda.cu"
$output = Join-Path $PSScriptRoot "visionflow_cuda.dll"
$importLibrary = Join-Path $PSScriptRoot "visionflow_cuda.lib"
$smokeSource = Join-Path $PSScriptRoot "test_cuda_api.cu"
$smokeExe = Join-Path $PSScriptRoot "test_cuda_api.exe"
$dllSources = @($source)
$smokeSources = @($smokeSource)
$stageDirectory = Join-Path $root "outputs_validation\cuda_build_stage"
$stageOutput = Join-Path $stageDirectory "visionflow_cuda.dll"
$stageImportLibrary = Join-Path $stageDirectory "visionflow_cuda.lib"
$stageSmokeExe = Join-Path $stageDirectory "test_cuda_api.exe"
$preflightManifest = Join-Path $root "outputs_validation\cuda_build_preflight.json"
$python = Join-Path $root "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python not found. Create the project env or add Python to PATH."
    }
    $python = $pythonCommand.Source
}

& $python (Join-Path $PSScriptRoot "preflight_cuda_build.py") --output $preflightManifest
if ($LASTEXITCODE -ne 0) {
    throw "CUDA source/API preflight failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path $stageDirectory | Out-Null
foreach ($stageArtifact in @($stageOutput, $stageImportLibrary, $stageSmokeExe)) {
    if (Test-Path -LiteralPath $stageArtifact) {
        Remove-Item -LiteralPath $stageArtifact -Force
    }
}

Write-Host "nvcc: $($nvcc.Source)"
Write-Host "cl: $($cl.Source)"
Write-Host "architecture: $Architecture"
Write-Host "preflight manifest: $preflightManifest"

& $nvcc.Source `
    "--std=c++17" `
    "-O3" `
    "--shared" `
    "--cudart=static" `
    "-arch=$Architecture" `
    "-I$include" `
    "-Xcompiler=/MD" `
    "-Xlinker" "/IMPLIB:$stageImportLibrary" `
    "-o" $stageOutput `
    $dllSources
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DLL build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $stageOutput)) {
    throw "nvcc returned success but DLL was not created: $stageOutput"
}
if (-not (Test-Path $stageImportLibrary)) {
    throw "DLL import library was not created: $stageImportLibrary"
}

& $nvcc.Source `
    "--std=c++17" `
    "-O2" `
    "-arch=$Architecture" `
    "-I$include" `
    "-Xcompiler=/MD" `
    "-o" $stageSmokeExe `
    $smokeSources `
    $stageImportLibrary
if ($LASTEXITCODE -ne 0) {
    throw "CUDA C ABI smoke executable build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $stageSmokeExe)) {
    throw "nvcc returned success but smoke executable was not created: $stageSmokeExe"
}

$expectedExports = @((Get-Content -LiteralPath $preflightManifest -Raw | ConvertFrom-Json).exports)
$exports = (& $dumpbin.Source /exports $stageOutput | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /exports failed with exit code $LASTEXITCODE"
}
foreach ($expectedExport in $expectedExports) {
    if ($exports -notmatch "(?m)\b$([regex]::Escape($expectedExport))\b") {
        throw "CUDA DLL is missing expected export: $expectedExport"
    }
}
$dependents = (& $dumpbin.Source /dependents $stageOutput | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /dependents failed with exit code $LASTEXITCODE"
}

Move-Item -LiteralPath $stageOutput -Destination $output -Force
Move-Item -LiteralPath $stageImportLibrary -Destination $importLibrary -Force
Move-Item -LiteralPath $stageSmokeExe -Destination $smokeExe -Force

Write-Host "Built CUDA DLL: $output"
Write-Host "Built import library: $importLibrary"
Write-Host "Built C ABI smoke executable: $smokeExe"
Write-Host "Verified exports: $($expectedExports -join ', ')"
Write-Host "DLL dependencies:"
Write-Host $dependents.Trim()

if ($RunTests) {
    & $smokeExe
    if ($LASTEXITCODE -ne 0) {
        throw "C ABI smoke test failed with exit code $LASTEXITCODE"
    }

    $validationArgs = @(
        (Join-Path $PSScriptRoot "validate_cuda_dll.py"),
        "--dll", $output,
        "--benchmark", $Benchmark
    )
    if ($Image -and $Recipe) {
        $validationArgs += @("--image", $Image, "--recipe", $Recipe)
    }
    & $python @validationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python CUDA validation failed with exit code $LASTEXITCODE"
    }
}
