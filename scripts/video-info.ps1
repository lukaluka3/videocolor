# Inspect a video using the installed VideoColor environment.
param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    Write-Error 'Input video does not exist.'
    exit 1
}
& python -m videocolor info $Path
exit $LASTEXITCODE
