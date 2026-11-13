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
| Dense depth (plane sweep; NCC, Census or SGM cost) | ✅ |
| Auto-config from measured texture | ✅ |
| Auto-crop (largest voxel cluster) + recentering | ✅ |
| TSDF fusion (confidence- and view-angle-weighted votes) | ✅ |
| Field smoothing + Taubin mesh smoothing | ✅ |
| Marching-tetrahedra mesh extraction | ✅ |
| Quadric-error-metric simplification | ✅ |
| Volumetric hole filling (off by default; see TAI) | ✅ |
| Mesh PLY (vertices, faces, normals, colours) | ✅ |
| Poisson backend comparison (Open3D, screened Poisson) | ✅ `scripts/open3d_poisson.py` |
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

### Poisson comparison backend (Open3D)

```bash
python3 -m venv ~/Dev/tools/venv-o3d
~/Dev/tools/venv-o3d/bin/pip install open3d
~/Dev/tools/venv-o3d/bin/python scripts/open3d_poisson.py \
    out/mesh_oriented.ply out/mesh_poisson.ply --target-faces 20000 \
    --reference out/mesh.ply
```

### Blender demo

```bash
~/Dev/tools/blender-4.5.3-linux-x64/blender -b \
    --python scripts/blender_demo.py -- out/mesh.ply out/demo
```

### Result on `swiss_knife.mp4` (628 frames, 1280×720 HEVC)

| Metric | Value |
|---|---|
| Registered views | 22 / 63 |
| Sparse points / RMSE | 1,175 / 2.086 px |
| Dense depth vs sparse depth | median 4.8 % error |
| Raw mesh | 109,933 faces |
| Final mesh | 20,000 faces, 36 patches, ~60 s CPU |
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
   the depth with the best NCC; require agreement from `--lr-required`
   neighbours.
7. **TSDF**: weighted "in front / behind" votes (confidence × view angle) into
   a signed distance field; optional box-filter smoothing.
8. **Mesh**: marching tetrahedra, drop small patches, Taubin smoothing,
   quadric-error-metric simplification.
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
| `--remove-plane` | RANSAC-remove the dominant plane (table/floor) before fusion |
| `--dense-scale` | depth-map resolution relative to source (0.5 default) |
| `--dense-planes` / `--dense-neighbors` / `--ncc-min` | plane-sweep quality knobs |
| `--cost ncc\|census\|sgm` | photometric cost (SGM regularizes low-texture regions) |
| `--sgm-paths` | cardinal SGM paths (default 4) |
| `--auto-config` | derive cost/window/NCC/planes and mesh budget from frame texture |
| `--auto-crop` | keep only the largest 26-connected voxel cluster |
| `--recenter` | translate outputs so their centre is the origin |
| `--voxel-resolution` / `--trunc-voxels` | TSDF resolution and band width |
| `--lr-required` | neighbours that must agree on a depth (default 1) |
| `--volume-smoothing` | TSDF box-filter iterations before meshing (default 2) |
| `--smooth-iterations` | Taubin smoothing iterations (default 8) |
| `--hole-fill-iterations` | TSDF diffusion hole filling (default 0; risky) |
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
- **Textureless objects are hard even with Census/SGM.** `lip_balm.mp4`
  (66/68 views registered, 2.05 px RMSE) goes from 16.8 % to 19–20 % dense
  depth error; SGM improves the fragmentation from 61 to 43 patches, but the
  white tube still lacks the signal. Sparse works; dense needs texture or a
  stronger prior (depth network / projected pattern).
- **`--auto-crop` needs a clean cloud.** On the noisy knife cloud the largest
  voxel cluster is a sub-region, shrinking the scene; use it after
  `--remove-plane` on well-behaved captures.
- On the supplied specular knife clip the mesh remains fragmented: the ceiling
  is the capture, not the solver (see the TAI report's Path-1/Path-2 results).
