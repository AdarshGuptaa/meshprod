#!/usr/bin/env bash
# Trustworthy-AI reproducibility check: run the full pipeline twice on the
# same input and verify that every output is byte-identical.
set -euo pipefail

VIDEO="${1:?usage: tai_reproduce.sh <video.mp4> [out_dir]}"
OUT="${2:-out/tai}"
mkdir -p "$OUT"

run() {
    local tag="$1"
    ./target/release/sfm2mesh "$VIDEO" "$OUT/$tag.ply" \
        --fx 895 --fy 895 --cx 640 --cy 360 \
        --detector sift --n-features 500 --frame-step 10 \
        --geo-verify --sprt
}

echo "[1/3] run A"
run a
echo "[2/3] run B"
run b

echo "[3/3] compare"
status=0
for suffix in "" "_sparse" "_cloud"; do
    a=$(sha256sum "$OUT/a$suffix.ply" | cut -d' ' -f1)
    b=$(sha256sum "$OUT/b$suffix.ply" | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then
        echo "  $suffix: identical ($a)"
    else
        echo "  $suffix: DIFFER ($a != $b)"
        status=1
    fi
done
exit "$status"
