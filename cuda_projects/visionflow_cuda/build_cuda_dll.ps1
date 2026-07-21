param(
    [string]$Architecture = "sm_86",
    [string]$BuildDirectory = ""
)

$ErrorActionPreference = "Stop"

$projectDirectory = $PSScriptRoot
$projectName = Split-Path -Leaf $projectDirectory
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $projectDirectory)
$builder = Join-Path $repositoryRoot ".github\scripts\build_cuda_project.ps1"
if (-not (Test-Path -LiteralPath $builder -PathType Leaf)) {
    throw "Common Action builder not found: $builder"
}
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $repositoryRoot "build\$projectName"
}

& $builder `
    -ProjectName $projectName `
    -ProjectDirectory $projectDirectory `
    -BuildDirectory $BuildDirectory `
    -Architecture $Architecture
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
