# Cut a local preview without replacing existing files.
# Example: .\scripts\cut-clip.ps1 -Path .\input.mp4 -Start '00:01:00' -Duration 10 -Exact
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [string]$Start = '00:00:00',
    [ValidateRange(1, 86400)][int]$Duration = 10,
    [string]$Output = '',
    [switch]$Exact
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'Install FFmpeg and reopen the terminal.' }
if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'Input video does not exist.' }
if ($Start -notmatch '^\d+(?::\d{1,2}){0,2}(?:\.\d+)?$') { throw 'Use seconds or HH:MM:SS for Start.' }
$inputFile = (Resolve-Path -LiteralPath $Path).Path
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path ([IO.Path]::GetDirectoryName($inputFile)) ([IO.Path]::GetFileNameWithoutExtension($inputFile) + '_preview_' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.mp4')
}
$outputFile = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputFile) { throw 'Output already exists. Choose a different filename.' }
if ([IO.Path]::GetExtension($outputFile) -ne '.mp4') { throw 'Choose an .mp4 output filename.' }
$arguments = @('-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-ss', $Start, '-i', $inputFile, '-t', "$Duration", '-map', '0:v:0', '-map', '0:a:0?', '-map_metadata', '-1', '-map_chapters', '-1')
if ($Exact) { $arguments += @('-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'aac') }
else {
    Write-Host 'Fast copy cuts near keyframes; use -Exact for precise timing.'
    $arguments += @('-c', 'copy')
}
& ffmpeg @arguments $outputFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path -LiteralPath $outputFile -PathType Leaf)) { throw 'FFmpeg did not create an output.' }
Write-Host "Saved: $outputFile"
