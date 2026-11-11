# sfm2mesh

Semester project: turn **one monocular phone video** of an object into a
**simplified, coloured, normal-bearing triangle mesh** (`.ply`).

```
phone MP4 ─► frames ─► sparse SfM ─► dense depth ─► TSDF fusion ─► marching tets ─► mesh ─► .ply
             (done)       (done)        (done)          (done)          (done)         (done)
```

## Status

| Stage | State |
|---|---|
| Video decode (GStreamer) | ✅ |
| SIFT/ORB features + matching + geometric verification | ✅ |
| Sparse reconstruction (`kornia-calib::reconstruct`) | ✅ |
| Sparse PLY (XYZ + RGB + normals) | ✅ |
| Dense depth (plane sweep, NCC, left-right check) | ✅ |
| TSDF fusion (`kornia-3d::voxel`-style dense grid) | ✅ |
| Marching-tetrahedra mesh extraction | ✅ |
| Quadric-error-metric simplification | ✅ |
| Mesh PLY (vertices, faces, normals, colours) | ✅ |
| Trustworthy-AI (EU ALTAI) assessment | ✅ `docs/tai/ALTAI-assessment.md` |

## Quick start

Requirements: Rust 1.89+, GStreamer 1.20+ with `libav` or `nvcodec` for
HEVC/H.265 phone video.

```bash
# path dependencies point at the local kornia-rs worktree (branch semproj/sfm2mesh)
cargo build --release

./target/release/sfm2mesh <video.mp4> out/mesh.ply \
    --fx 895 --fy 895 --cx 640 --cy 360 \
    --detector sift --n-features 500 --frame-step 10 \
    --geo-verify --sprt
```

Outputs next to `out/mesh.ply`:

| File | Content |
|---|---|
| `mesh.ply` | simplified triangle mesh with normals + vertex colours |
| `mesh_sparse.ply` | sparse SfM point cloud |
| `mesh_cloud.ply` | voxel-downsampled dense point cloud |

Open them in MeshLab, CloudCompare, Blender or Open3D.

### Result on `swiss_knife.mp4` (628 frames, 1280×720 HEVC)

| Metric | Value |
|---|---|
| Registered views | 22 / 63 |
| Sparse points / RMSE | 1,175 / 2.086 px |
| Dense depth vs sparse depth | median 4.8 % error |
| Raw mesh | 39,521 faces |
| Final mesh | 19,999 faces, 18 patches, ~55 s CPU |
| Determinism | byte-identical reruns |

The object is a polished Swiss knife; its specular blade is the main reason the
mesh is open and split into patches. See the TAI report for the failure modes.

## Pipeline (short)

1. **Decode** every Nth frame to RGB + gray.
2. **SIFT** keypoints/descriptors per frame (CUDA optional).
3. **Match** within a sliding window: mutual nearest neighbours, Lowe ratio,
   epipolar-RANSAC verification.
4. **Track** matches across frames (`kornia-calib`).
5. **Sparse SfM**: bootstrap, PnP registration (SPRT), triangulation, bundle
   adjustment (up to scale).
6. **Dense depth**: per registered view, sweep inverse-depth planes and keep
   the depth with the best NCC; cross-check with neighbours.
7. **TSDF**: average "in front / behind" votes into a signed distance field.
8. **Mesh**: marching tetrahedra, drop small patches, quadric-error-metric
   simplification.
9. **Attributes**: normals from the TSDF gradient, colours by projecting each
   vertex into the source frames.

`docs/pipeline.md` explains the theory of every stage.

## CLI (main flags)

| Flag | Meaning |
|---|---|
| `--fx/--fy/--cx/--cy` | pinhole intrinsics in pixels (required) |
| `--detector orb\|sift` | feature detector |
| `--frame-step N` | keep every Nth frame (10 works well for orbit clips) |
| `--geo-verify` | RANSAC epipolar verification |
| `--sprt` | SPRT early rejection during PnP |
| `--sparse-only` | stop after the sparse cloud |
| `--dense-scale` | depth-map resolution relative to source (0.5 default) |
| `--dense-planes` / `--dense-neighbors` / `--ncc-min` | plane-sweep quality knobs |
| `--voxel-resolution` / `--trunc-voxels` | TSDF resolution and band width |
| `--target-faces` | simplify the mesh to this budget (0 = keep) |
| `--min-component-faces` | drop patches smaller than this (0 = keep) |

Intrinsics from a phone: `f_px = f_35mm / 36 * frame_width`, principal point at
the image centre. Portrait clips are often decoded *unrotated* — use the
intrinsics of the decoded frame.

## Layout

```
src/video.rs          decode MP4 → RGB + gray frames
src/features.rs       ORB / SIFT extraction (CPU + CUDA SIFT)
src/matching.rs       descriptor matching, ratio test, epipolar RANSAC
src/reconstruction.rs thin wrapper over kornia-calib::reconstruct
src/dense.rs          plane-sweep dense depth + dense cloud
src/tsdf.rs           signed-distance fusion + trilinear sampling
src/mesh.rs           marching tetrahedra, normals, colours, patch cleanup
src/simplify.rs       quadric-error-metric edge collapse
src/mesh_ply.rs       binary mesh PLY writer
src/ply_writer.rs     binary point-cloud PLY writer
docs/pipeline.md      how every stage works (start here)
docs/tai/             EU ALTAI trustworthiness assessment
scripts/              reproducibility check for the TAI experiments
```

## Dependency note

Dependencies are **paths** into `../kornia-rs-semproj` (a detached worktree of
`semproj/sfm2mesh`) for fast local iteration. For a reproducible hand-in,
switch them to the git branch:

```toml
kornia-3d = { git = "https://github.com/AdarshGuptaa/kornia-rs.git", branch = "semproj/sfm2mesh" }
```

## Known limitations

- Up to scale: no metric anchor (AprilTag) yet.
- Specular, textureless and thin objects reconstruct poorly.
- The mesh can be open (partial view coverage); `closed=` is logged honestly.
- The dense stage ignores lens distortion (a warning is printed if the solve
  fitted a large one).
- Background inside the reconstruction volume is meshed too (no segmentation).
