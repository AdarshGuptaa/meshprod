# How the sparse SfM pipeline works

Start here before changing anything. Every stage below exists in `src/`, and
most of the heavy maths lives in the `kornia-rs` crates the project depends on.

---

## 0. The camera model (the foundation of everything)

A pinhole camera maps a 3D point `X = (X, Y, Z)` **in camera coordinates** to a
pixel:

```
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

`fx, fy` are focal lengths in pixels, `(cx, cy)` the principal point (image
centre for a phone). The same relation in homogeneous form:

```
p ~ K [R | t] X_world        K = [[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]]
```

`[R | t]` is the **world→camera** transform. Internally the solver projects
points with it; the pipeline's output stores the inverse **camera→world**
(`T_world_cam`), because a camera *centre* is then simply the translation.

**Normalized coordinates** `x = (X/Z, Y/Z) = ((u-cx)/fx, (v-cy)/fy)` are
unitless and independent of resolution. Most geometric thresholds in
`kornia-calib` are in normalized units; RANSAC pixel thresholds are in pixels.
Mixing them up is the classic bug — e.g. `max_reprojection_error = 0.01`
normalized ≈ 9 px at `fx = 895`.

Monocular SfM cannot recover metric scale: scaling the whole scene by `s` leaves
every projection identical. The map is therefore **up to scale** unless an
object of known size (AprilTag, ruler) anchors it. Shape is correct; units are
arbitrary.

---

## 1. Frames — `src/video.rs`

`VideoReader` (GStreamer) decodes to RGB. We keep every `frame_step`-th frame
and convert each to grayscale (`gray_from_rgb_u8`) for feature detection:

- **RGB** is kept only until colours are sampled from it.
- **Gray** is kept until features are extracted.

Two known GStreamer quirks are handled here:

1. `grab_rgb8()` returns `None` for "no frame yet", end-of-stream *and* pipeline
   error alike (kornia-io #1158). A file that never delivers a frame would spin
   forever, so there is a startup timeout. Low-fps truncation is a separate
   bug (#1158) left open.
2. RGB readers ignore row stride for widths not divisible by 4 (#1160, fixed
   upstream); 1280 is safe.

`frame_step` is the first big quality/CPU trade-off. Adjacent frames have almost
no parallax, so every 10th frame is both cheaper and better conditioned.

---

## 2. Features — `src/features.rs`

We need repeatable, distinctive image locations with a descriptor that can be
matched across viewpoints.

**ORB** — FAST corner detector + oriented BRIEF descriptor.
- FAST: compare a circle of 16 pixels around a candidate; many consecutive
  brighter/darker pixels ⇒ corner. Very fast, no scale invariance by itself.
- ORB adds an image pyramid and orientation (intensity centroid), and rotates
  the BRIEF sampling pattern. Descriptor = **256 bits** (32 bytes), matched by
  **Hamming distance** (XOR + popcount).
- Cheap, good for small baselines; weaker under large viewpoint/scale change.

**SIFT** — Difference-of-Gaussians keypoints + gradient-histogram descriptor.
- Build a Gaussian scale space, take DoG, find 3D (x, y, σ) extrema, refine to
  sub-pixel, assign orientation from gradient histograms.
- Descriptor = **128 floats** (4×4 spatial cells × 8 orientation bins),
  normalized, matched by **L2 distance**.
- Slower, markedly more robust to scale/rotation; the right default for a
  phone orbit around an object.

Both are exposed behind one `FeatureExtractor` trait; extraction runs on rayon
(CPU) or sequentially on one CUDA stream (CUDA SIFT shares a stream, so it
cannot be parallel).

---

## 3. Pairwise matching — `src/matching.rs`

Matching proposes correspondences; geometry then throws the lies away.

### 3a. Descriptor distance + Lowe's ratio test

For keypoint `a` in frame *i*, find its best and second-best descriptor
distances `d1, d2` among frame *j*. Accept only if

```
d1 / d2 < ratio        (default 0.8)
```

A distinctive feature is much closer to its true match than to the runner-up.
Low-contrast/repetitive texture (a blank wall) produces `d1 ≈ d2` and is
rejected. Lower ratio = stricter.

### 3b. Mutual nearest neighbours (cross-check)

Accept `a → b` only if `b`'s nearest neighbour is also `a`. Why this matters
here: `build_tracks` requires **at most one observation per camera per track**.
Forward-only matching lets two nearby keypoints in frame *i* both claim the same
keypoint in *j*; the track then visits camera *j* twice and is dropped
entirely. In the PR benchmark, restoring the cross-check recovered the SIFT
view count from 9/63 to 22/63.

### 3c. Epipolar verification (optional, `--geo-verify`)

Two views of a static scene satisfy

```
x2ᵀ F x1 = 0
```

for every true correspondence, where `F` is the **fundamental matrix** (pixel
coordinates). `F` is estimated with RANSAC:

- minimal sample: **7 points** (the 7-point algorithm: `det(F) = 0` gives a
  cubic, up to 3 solutions) or 8 points (linear, Hartley-normalized);
- score: count points whose **Sampson distance** (first-order geometric
  distance to the epipolar line) is below the pixel threshold;
- keep the best model, refit it on all inliers.

Pairs need enough matches to be trustworthy: fewer than 7 cannot fit the model,
and exactly 7 is the minimal sample explaining itself — hence the floor of 15
inliers (`MIN_SAFE_GEO_INLIERS`). Determinism: pairs are processed per camera
pair and results emitted in **input order**, so a hash-map iteration order can
never leak into the output.

---

## 4. Multi-view tracks — `kornia-calib::build_tracks`

Each `TrackEdge` is an observation `(frame i, keypoint a) ↔ (frame j, keypoint
b)`. `build_tracks` runs **union-find** over `(camera, keypoint)` nodes: a
connected component is one physical 3D point seen in several frames. Components
that contain two observations from the same camera are invalid (one point cannot
appear twice in one image) and are discarded. Result: `FeatureTrack { obs:
[(cam, pixel), …] }`.

More/better edges ⇒ longer tracks ⇒ better triangulation and more registrable
views. This is why matching quality dominates the whole pipeline.

---

## 5. Incremental reconstruction — `kornia-calib::reconstruct`

`src/reconstruction.rs` is a thin wrapper: one `PinholeCamera` per frame,
`ReconstructionConfig::new(0.0).sequential()`, then `reconstruct(&cameras, &[],
&tracks, &config, None)`. Inside:

### 5a. Bootstrap

Pick the best-initialized pair of views, estimate the **essential matrix**

```
E = [t]ₓ R          x2ᵀ E x1 = 0   (normalized coordinates)
```

(five-point RANSAC; `F` and `E` are related by `E = Kᵀ F K`), decompose
`E → (R, t)` (four solutions; cheirality picks the one with points in front of
both cameras), triangulate the inlier tracks, and run a first bundle
adjustment. This fixes the world frame and the unknown scale.

### 5b. Registration of new views (PnP)

For each remaining view, find 2D↔3D correspondences (track observations ↔
already-triangulated points) and solve **Perspective-n-Point** — AP3P/EPnP —
inside RANSAC. If inliers ≥ `min_registration_inliers` (30), the view is
registered.

`--sprt` enables **Wald's sequential probability ratio test**: instead of
scoring all points to decide whether a pose is good, accumulate evidence
inlier-by-inlier and stop early once the pose is clearly bad (or clearly
good). Same decision, far fewer evaluations — this is why it lifts the
registered-view count at equal runtime.

### 5c. Triangulation

Registers create new points from track observations in ≥ 2 views. Linear
multi-view triangulation (`x × P X = 0`, SVD) or the optimal two-view midpoint
method; points with high reprojection error or negative depth are rejected.

### 5d. Bundle adjustment

After each batch of registrations, minimize the total **reprojection error**

```
min Σ ‖ π(R_i X_j + t_i) − x_ij ‖²
```

over camera poses and 3D points (Levenberg–Marquardt, Schur complement to
exploit sparsity). The reported **RMSE** is the root-mean-square of that error
in pixels: `2.086 px` on the sample clip. It is a *consistency* measure, not a
truth measure — a wrong but self-consistent map can have low RMSE.

`--refine-intrinsics` additionally fits a global focal scale plus radial/
tangential distortion (COLMAP-style), useful when the phone's intrinsics are
guessed.

### 5e. Why only 22/63 views?

Views register only with ≥ 30 verified 2D↔3D inliers against a well-conditioned
map. Motion blur, rolling shutter, textureless or repetitive regions, and
accumulated drift all break that chain. 22/63 is expected for hand-held
monocular orbit footage and is enough for the sparse stage; the dense stage will
use exactly these registered views.

---

## 6. PLY export — `src/ply_writer.rs`

Binary little-endian PLY, 27 bytes/vertex: `x y z` (f32), `red green blue`
(u8), `nx ny nz` (f32). Readable by MeshLab, CloudCompare, Blender, Open3D.

- **Colour**: sampled from the RGB frame at each track observation *before*
  reconstruction (so frames can be freed). A point takes the colour of an
  observation that survived the solve, with the raw track as fallback.
- **Normals**: for each point, k-nearest neighbours (k=20, kd-tree) →
  covariance → SVD; the eigenvector of the smallest singular value is the local
  plane normal. The sign is arbitrary, so it is flipped toward the observer that
  sees the point most head-on (`max |n·dir|`) — this is what makes the normals
  point "outward" on a scanned object instead of randomly in/out.

---

## 7. Dense depth → solid mesh

The sparse cloud only has points where SIFT fired. The dense stage reuses the
recovered poses and adds per-pixel depth, then turns depth into a surface.

### 7a. Plane sweep — `src/dense.rs`

For a reference view `r` and a neighbour `n`, the relative motion is

```
X_n = R · X_r + t        R = R_nᵀ R_r,  t = R_nᵀ (c_r − c_n)
```

with `c` the camera centres (`T_world_cam` translations). A fronto-parallel
plane at depth `d` in `r` induces the homography

```
H(d) = K_n (R + (t e₃ᵀ) / d) K_r⁻¹          (e₃ = (0,0,1))
```

We sweep **inverse depth** `ρ = 1/d` (uniform sampling in disparity, like a
focus ring) over a range taken from the sparse points' depths. For every pixel
and every plane, warp the neighbour and score a 5×5 normalized
cross-correlation; winner-takes-all gives the depth, `NCC < 0.4` rejects flat
or ambiguous patches.

**Left-right check**: backproject the winning depth, project it into the
neighbour, and require the neighbour's own estimate to agree within 5 %.
This removes occluded/grazing pixels that photo-consistency alone accepts.

### 7b. TSDF fusion — `src/tsdf.rs`

Each depth pixel votes on the ray: in front of the measured surface is empty
(`+`), behind it is solid (`−`), with

```
sdf(v) = weighted mean of  (depth − v_z) / trunc
```

truncated to ±1 and only accumulated within `trunc` of a measurement. The
grid covers the dense cloud's bounding box at `--voxel-resolution` points along
its longest axis. Unobserved voxels keep weight 0.

Two practical lessons learned the hard way:

- the truncation band must be wider than the depth noise, or views disagree and
  the surface averages away (the default is 3 voxels at 256 resolution);
- meshing must only use cells whose **eight corners are observed** — treating
  unobserved corners as "empty" invents a surface at the inner edge of the band.

### 7c. Marching tetrahedra — `src/mesh.rs`

Each cube is split into six tetrahedra sharing body diagonal 0–7. A tetrahedron
has only three crossing patterns (1-in/3-out, 2-in/2-out, 3-in/1-out), so no
256-entry table is needed. The 2-in/2-out case produces a quad whose vertices
**must be traversed cyclically** — in edge-list order it bowties and leaves the
mesh open. Vertices are welded by quantized position; each triangle's winding
is fixed against the TSDF gradient so normals point outward.

### 7d. Cleanup and simplification — `src/mesh.rs`, `src/simplify.rs`

Photogrammetry produces small floating fragments; connected patches below
`--min-component-faces` are removed. The remaining mesh is simplified with
**quadric error metrics**: each vertex accumulates `Q = Σ plane quadrics`, an
edge collapse costs `xᵀA x + 2bᵀx + c` at the optimal merged position, and a
heap removes the cheapest shape change first. The link condition and a
normal-flip check keep the result manifold and fold-free.

### 7e. Attributes — `src/mesh.rs`

- **Normals** = normalized TSDF gradient at the vertex.
- **Colours** = weighted average of the source frames that see the vertex,
  weight `cos²` of the angle between the normal and the viewing ray, so
  grazing views cannot smear the colour.

## 8. Code map

| Concept | Where |
|---|---|
| Decode + gray | `src/video.rs` |
| ORB / SIFT | `src/features.rs`, `crates/kornia-imgproc/src/features/` |
| Ratio + mutual NN | `crates/kornia-imgproc/src/features/match.rs` |
| Epipolar RANSAC | `crates/kornia-3d/src/pose/twoview/ransac.rs` (`ransac_fundamental`) |
| 7-point / 5-point solvers | `crates/kornia-3d/src/pose/fundamental_7pt.rs`, `essential_5pt.rs` |
| Tracks | `crates/kornia-calib/src/tracks.rs` |
| Reconstruction (BA, PnP, triangulation) | `crates/kornia-calib/src/sfm.rs` |
| PnP | `crates/kornia-3d/src/pnp/` |
| Bundle adjustment | `crates/kornia-3d/src/ba.rs`, `ba_schur.rs` |
| Plane sweep | `src/dense.rs` |
| TSDF | `src/tsdf.rs` |
| Surface extraction, normals, colours, cleanup | `src/mesh.rs` |
| Mesh simplification | `src/simplify.rs` |
| PLY writers | `src/ply_writer.rs` (points), `src/mesh_ply.rs` (mesh) |
| Voxel grid primitives | `crates/kornia-3d/src/voxel/` |
