#!/usr/bin/env bash
# One-shot Linux setup: GStreamer dev packages + Rust build.
set -euo pipefail
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential pkg-config \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-good gstreamer1.0-libav
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y gcc gcc-c++ make pkgconf-pkg-config \
        gstreamer1-devel gstreamer1-plugins-base-devel \
        gstreamer1-plugins-good gstreamer1-libav
else
    echo "Install GStreamer 1.20+ dev packages manually for your distro." >&2
fi
cargo build --release
cargo test
echo "Build OK. Run: ./target/release/sfm2mesh --help"
