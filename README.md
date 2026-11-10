# sfm2mesh

Semester project: turn **one monocular phone video** of an object into a
**simplified, coloured, normal-bearing triangle mesh** (`.ply`).

```
phone MP4 ──► frames ──► sparse SfM ──► dense depth ──► TSDF fusion ──► marching cubes ──► mesh ──► .ply
             (done)       (done)        (next)          (next)          (next)          (next)
```

## Status

| Stage | State |
|---|---|
| Video decode (GStreamer) | ✅ ported from `kornia-rs` PR #1136 |
| SIFT/ORB features + matching + geometric verification | ✅ ported |
| Sparse reconstruction (`kornia-calib::reconstruct`) | ✅ working |
| Sparse PLY (XYZ + RGB + normals) | ✅ working |
| Dense depth (plane sweep) | ⬜ planned |
| TSDF fusion (uses `kornia-3d::voxel`) | ⬜ planned |
| Marching cubes + simplification + mesh PLY | ⬜ planned |
| Trustworthy-AI (EU ALTAI) analysis | ⬜ planned |

The sparse half is the code from
[PR #1136](https://github.com/kornia/kornia-rs/pull/1136) (author: this repo's
owner), copied out of the workspace so it can evolve independently.

## Quick start

Requirements: Rust 1.89+, GStreamer 1.20+ with the `libav` or `nvcodec` plugin
for HEVC/H.265 phone video.

```bash
# path dependencies point at the local kornia-rs checkout (branch semproj/sfm2mesh)
cargo build --release

./target/release/sfm2mesh <video.mp4> out/sparse.ply \
    --fx 895 --fy 895 --cx 640 --cy 360 \
    --detector sift --n-features 500 --frame-step 10 \
    --geo-verify --sprt
```

Open `out/sparse.ply` in MeshLab, CloudCompare or Blender.

### Reproduced result (`swiss_knife.mp4`, 628 frames, 1280×720 HEVC)

| Config | Points | Views | RMSE (px) | Time |
|---|---|---|---|---|
| SIFT, frame-step 10, geo-verify, SPRT | 1175 | 22/63 | 2.086 | ~44 s CPU |

Identical to the PR's champion CPU run — the port is faithful and the pipeline
is deterministic.

## CLI

| Flag | Meaning |
|---|---|
| `--fx/--fy/--cx/--cy` | pinhole intrinsics in pixels (required) |
| `--detector orb\|sift` | feature detector (default `orb`) |
| `--n-features N` | max keypoints per frame |
| `--frame-step N` | keep every Nth frame |
| `--match-window N` | match each frame against the next N frames |
| `--geo-verify` | RANSAC epipolar verification of matches |
| `--geo-min-inliers N` | min inliers per verified pair (floor 15) |
| `--wide-baseline K` | extra matches at distance K, 2K, … |
| `--sprt` | Wald SPRT early rejection during PnP registration |
| `--refine-intrinsics` | refine focal scale + distortion against the solve |
| `--cuda` | CUDA SIFT (**SIFT only**) |

Intrinsics from a phone: `f_px = f_35mm / 36 * frame_width`, principal point at
the image centre. Portrait clips are often decoded *unrotated* — take the
intrinsics for the decoded frame, not the phone's sensor orientation.

## Layout

```
src/video.rs          decode MP4 → RGB + gray frames
src/features.rs       ORB / SIFT extraction (CPU + CUDA SIFT)
src/matching.rs       descriptor matching, ratio test, epipolar RANSAC
src/reconstruction.rs thin wrapper over kornia-calib::reconstruct
src/ply_writer.rs     binary PLY (XYZ + RGB + normals)
src/main.rs           CLI + pipeline orchestration
docs/pipeline.md      ← how every stage works (start here)
```

## Dependency note

Dependencies are **paths** into `../kornia-rs` for fast local iteration. For a
reproducible hand-in, switch them to the git branch:

```toml
kornia-3d = { git = "https://github.com/AdarshGuptaa/kornia-rs.git", branch = "semproj/sfm2mesh" }
```

## Roadmap

1. Dense depth maps per registered view (plane sweep, NCC, left-right check).
2. TSDF fusion over the sparse voxel grid (`kornia-3d::voxel`, merged upstream
   in #1242) — this is where the point cloud becomes a surface.
3. Marching cubes → triangle mesh; vertex normals from the TSDF gradient.
4. Quadric edge-collapse simplification + colour projection from source frames.
5. Mesh PLY writer (vertices, faces, normals, colours).
6. TAI evaluation (EU ALTAI): robustness, accuracy, repeatability, privacy,
   transparency, human oversight, accountability.
