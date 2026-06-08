# Build the On The Record Windows installer with Inno Setup.
#
# Prerequisites:
#   1. Inno Setup installed, with `iscc.exe` on PATH
#      (https://jrsoftware.org/isdl.php).
#   2. dist\On The Record.exe already built:
#         uv run python scripts/build_windows_tray.py
#
# Usage:
#   powershell -File scripts\build_windows_installer.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot 'dist\On The Record.exe'
$script = Join-Path $repoRoot 'windows\installer\on-the-record.iss'

if (-not (Test-Path $exe)) {
    throw "Missing $exe. Run: uv run python scripts/build_windows_tray.py"
}

# Prefer iscc on PATH, then fall back to common (incl. per-user) install locations.
$isccPath = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $isccPath) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $isccPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $isccPath) {
    throw "ISCC.exe not found on PATH or in standard install locations. Install Inno Setup from https://jrsoftware.org/isdl.php"
}

& $isccPath $script
Write-Host "Installer written to $(Join-Path $repoRoot 'dist\installer')"
