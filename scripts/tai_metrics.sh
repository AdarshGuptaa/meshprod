#!/usr/bin/env bash
# Robustness suite: how do the measured TAI metrics move under controlled
# perturbations (resolution, blur, noise, frame drop)?
#
#   scripts/tai_metrics.sh <video.mp4> [out_dir]
#
# Intrinsics default to the 1280x720 phone values and are halved for 360p.
set -euo pipefail

VIDEO="${1:?usage: tai_metrics.sh <video.mp4> [out_dir]}"
OUT="${2:-out/tai_suite}"
FX="${FX:-895}"; FY="${FY:-895}"; CX="${CX:-640}"; CY="${CY:-360}"
BIN=./target/release/sfm2mesh
mkdir -p "$OUT"

perturb() {  # name, video-filter
    local name=$1 filter=$2
    echo "[suite] encoding $name"
    ffmpeg -y -v error -i "$VIDEO" -vf "$filter" -c:v libx264 -crf 20 "$OUT/$name.mp4"
}

run() {  # name, file, fx, fy, cx, cy
    local name=$1 file=$2 fx=$3 fy=$4 cx=$5 cy=$6
    echo "[suite] running $name"
    "$BIN" "$file" --output "$OUT/$name.ply" \
        --fx "$fx" --fy "$fy" --cx "$cx" --cy "$cy" \
        --detector sift --n-features 500 --frame-step 10 \
        --geo-verify --sprt --auto-config --remove-plane \
        > "$OUT/$name.log" 2>&1 || echo "  (failed: see $OUT/$name.log)"
}

cp "$VIDEO" "$OUT/baseline.mp4"
perturb p360 "scale=640:360"
perturb blur "gblur=sigma=2"
perturb noise "noise=alls=12:allf=t"
perturb drop "select='not(mod(n,3))',setpts=N/FRAME_RATE/TB"

run baseline "$OUT/baseline.mp4" "$FX" "$FY" "$CX" "$CY"
run p360 "$OUT/p360.mp4" "$(bc -l <<<"$FX/2")" "$(bc -l <<<"$FY/2")" 320 180
run blur "$OUT/blur.mp4" "$FX" "$FY" "$CX" "$CY"
run noise "$OUT/noise.mp4" "$FX" "$FY" "$CX" "$CY"
run drop "$OUT/drop.mp4" "$FX" "$FY" "$CX" "$CY"

python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "*_report.json"))):
    name = os.path.basename(path).replace("_report.json", "")
    with open(path) as f:
        report = json.load(f)
    s = report.get("stats", {})
    q = report.get("mesh_quality", {})
    rows.append((name, s.get("views"), s.get("rmse_px"), s.get("depth_median_pct"),
                 q.get("faces"), q.get("patches"), q.get("euler"), q.get("closed"),
                 report.get("self_assessment", {}).get("confidence")))
print(f"{'variant':10} {'views':>5} {'rmse':>6} {'depth%':>7} {'faces':>7} {'patch':>5} {'euler':>6} {'closed':>6} {'trust':>5}")
for r in rows:
    print(f"{r[0]:10} {r[1] or 0:5} {r[2] or 0:6.2f} {r[3] or 0:7.1f} {r[4] or 0:7} {r[5] or 0:5} {r[6] or 0:6} {str(bool(r[7])):>6} {(r[8] or 0)*100:5.0f}")
PY
