# One-shot Windows setup: GStreamer (devel + runtime) + Rust build.
# Requires an elevated PowerShell for the MSI installs.
$ErrorActionPreference = "Stop"
$version = "1.24.2"
$base = "https://gstreamer.freedesktop.org/data/pkg/windows/$version/msvc"
$work = Join-Path $env:TEMP "gstreamer-setup"
New-Item -ItemType Directory -Force -Path $work | Out-Null

foreach ($pkg in @("devel", "runtime")) {
    $msi = Join-Path $work "gstreamer-1.0-$pkg-msvc-x86_64-$version.msi"
    if (-not (Test-Path $msi)) {
        Write-Host "Downloading $pkg ..."
        Invoke-WebRequest "$base/gstreamer-1.0-$pkg-msvc-x86_64-$version.msi" -OutFile $msi
    }
    Write-Host "Installing $pkg ..."
    Start-Process msiexec.exe -ArgumentList "/i", "`"$msi`"", "/quiet", "/norestart" -Wait
}

$root = "C:\gstreamer\1.0\msvc_x86_64"
$env:GSTREAMER_1_0_ROOT_MSVC_X86_64 = $root
$env:PKG_CONFIG_PATH = "$root\lib\pkgconfig"
$env:PATH = "$root\bin;$env:PATH"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Install Rust from https://rustup.rs first (or run: winget install Rustlang.Rustup)."
    exit 1
}
cargo build --release
cargo test
Write-Host "Build OK. Run: .\target\release\sfm2mesh.exe --help"
