# meshprod — complete process, methods, and iteration history

This document explains, end to end, how `meshprod` turns a monocular video into
a measured triangle mesh: every stage, the mathematics and the research it comes
from, which kornia-rs routine implements it, and the concrete failures we hit
while building it and how each was diagnosed and fixed. It is meant to be read
as both a design reference and an honest lab notebook.

Contents

1. Goals and system layout
2. The pipeline at a glance
3. Stage-by-stage detail (with references)
4. Measuring and trusting the result
5. Iteration history: what broke, why, and the fix
6. Current limitations
7. References

---

## 1. Goals and system layout

`meshprod` reconstructs a **single object** filmed by a hand-held or orbiting
camera, producing a triangle mesh plus a machine-readable report that states how
reliable the result is. Two properties are non-negotiable:

- **Correctness over decoration.** Every number in the report is measured
  (reprojection error, depth-vs-sparse disagreement, mesh topology). Nothing is
  asserted without a check behind it.
- **Honest failure.** When the input cannot support a mesh, the pipeline stops
  with a named error and actionable hints rather than emitting a confident blob.

The project is a thin application crate (`sfm2mesh`, this repo) built on the
**kornia-rs** library (a Rust computer-vision stack) pinned to the
`semproj/sfm2mesh` branch. The app owns the orchestration, the dense/TSDF/mesh
stages and the web UI; kornia-rs owns features, matching primitives, multi-view
geometry, PnP, bundle adjustment and incremental SfM.

```
video → decode → features → match → tracks → sparse SfM → [scale]
      → foreground isolation → dense depth → TSDF → mesh → report
```

A local web server (`--serve`) runs the CLI as a child process, parses its
progress and metric lines, and serves a browser front-end plus the PLY files.

---

## 2. The pipeline at a glance

| # | Stage | Output | Core routine |
|---|-------|--------|--------------|
| 1 | Decode | RGB + gray frames, every Nth | `kornia_io::gstreamer::video` |
| 2 | Features | keypoints + descriptors | SIFT/ORB (`kornia_imgproc` + app) |
| 3 | Match | pairwise correspondences | app `matching.rs` + `kornia_3d` RANSAC |
| 4 | Tracks | multi-view tracks | `kornia_calib::build_tracks` |
| 5 | Sparse SfM | poses + 3D points | `kornia_calib::reconstruct` |
| 6 | Scale | metric or up-to-scale | AprilTag + `estimate_focal` + robust N-view |
| 7 | Isolation | object mask + box | app `foreground.rs` |
| 8 | Dense depth | per-view depth maps | app `dense.rs` (plane sweep / SGM) |
| 9 | Fusion | signed distance field | app `tsdf.rs` |
| 10 | Mesh | triangles + topology clean-up | app `mesh.rs`, `simplify.rs` |
| 11 | Report | `_report.json`, `_error.json` | app `main.rs`, `error.rs` |

---

## 3. Stage-by-stage detail

### 3.1 Decode

The video is read through GStreamer into RGB frames, and each is converted to
grayscale for feature detection (`gray_from_rgb_u8`, BT.601 luma). Only every
`--frame-step`-th frame is kept: consecutive 30 fps frames move too little to
triangulate stably, so a step of 5–20 sets a healthy baseline while cutting
decode and matching cost. Frames are cached (see §5.8) because decode is the
single most expensive fixed cost.

The **intrinsics** must match the *decoded* frame, not the sensor: a portrait
clip decoded unrotated has its principal point swapped relative to a landscape
one. `fx ≈ 0.7 × max(width, height)` is a good phone guess; the tool warns when
the principal point is far from the image centre or the focal is implausible.

### 3.2 Feature extraction

SIFT (Lowe, 2004) is the default: scale- and rotation-invariant, robust to
moderate viewpoint change, and sub-pixel-localised. ORB (Rublee et al., 2011)
is available as a faster, weaker alternative for binary-descriptor pipelines.
Each keypoint carries a descriptor; the track builder later joins keypoints that
match across frames.

A capture with fewer than ~80 keypoints per frame is almost always blurry,
defocused or textureless; the pipeline warns, because no downstream stage can
recover structure that was never detected.

### 3.3 Matching

For each frame `i`, the matcher compares it to frames `i+1 … i+W`
(`--match-window`) using **mutual nearest neighbours** plus **Lowe's ratio
test**: keep a pair only if the best descriptor is clearly better than the
second best, `d₁ < r · d₂` (Lowe, 2004). This rejects ambiguous matches in
repetitive texture.

Optional **epipolar verification** (Hartley & Zisserman, 2004) fits a
fundamental matrix per frame pair with RANSAC (Fischler & Bolles, 1981) and
keeps only geometrically consistent inliers. The seven-point fundamental
estimator needs ≥ 7 points; a pair with fewer inliers than `--geo-min-inliers`
is dropped entirely. **Wide-baseline** matching additionally links frame `i` to
`i+K, i+2K, …` to bridge fast motion.

### 3.4 Tracks

Matches are chained into multi-view tracks with a union-find over
`(camera, keypoint)` nodes (`kornia_calib::build_tracks`). Two subtleties are
handled explicitly:

- **Orientation siblings.** SIFT can emit two keypoints at the same pixel with
  different orientations; they say the same thing geometrically and are merged.
- **Genuine ambiguity.** Two *distinct* image locations from one camera landing
  in one component is a real contradiction; that track is dropped.

Track order is made deterministic (BTreeMap keyed by union-find root) because
floating-point summation in bundle adjustment depends on observation order.

### 3.5 Sparse structure-from-motion

`kornia_calib::reconstruct` implements incremental SfM in the COLMAP tradition
(Schönberger & Frahm, 2016):

1. **Bootstrap.** Pick a seed pair by geometry (shared tracks, median
   triangulation angle, homography-vs-fundamental ratio) and initialise a
   two-view essential solve (Hartley & Zisserman, 2004). World = first camera,
   scale = unit.
2. **Grow.** Repeatedly register the unplaced camera with the most 2D↔3D links
   via **PnP-RANSAC**. EPnP (Lepetit et al., 2009) gives the pose; RANSAC
   rejects outlier correspondences; **SPRT** (Wald, 1945) can stream residuals
   and reject hopeless hypotheses early. A view is accepted only with at least
   `--min-registration-inliers` inliers — a thin registration is not
   self-correcting because later points are triangulated *from* it.
3. **Triangulate.** New tracks are triangulated (midpoint / linear DLT) with
   parallax, cheirality and reprojection gates.
4. **Bundle adjustment.** Minimise reprojection error over poses and points with
   Levenberg–Marquardt and a robust (Huber/Cauchy) loss via IRLS (Triggs et al.,
   2000; Lourakis & Argyros, 2009). The reduced camera system is factorised
   sparsely, which scales linearly rather than cubically in the camera count.
5. **Maintenance.** Periodic global BA, reprojection filtering, re-admission of
   previously rejected sightings (`CompleteTracks`), and local windowed BA after
   each registration keep long chains from drifting.

The solve is **up to scale** until something provides a metric reference.

### 3.6 Metric scale and calibration

Two different problems are often conflated:

- **Focal length** (intrinsics). Estimated from a planar AprilTag (Olson, 2011)
  via Zhang's orthonormality constraint (Zhang, 2000):
  `kornia_calib::estimate_focal` recovers `f` from the board→image homography,
  taking the median over sampled frames and reporting the robust (MAD) spread.
  Single-view estimation cannot separate the principal point or distortion;
  multi-view Zhang (`calibrate_multishot`) is available for that.
- **Metric scale** (extrinsics gauge). A tag of known edge length *in the scene*
  turns a shape into a measured shape. kornia's `tag_scale` triangulates the
  corners from a single widest-baseline pair; on an orbit that pair can be
  near-degenerate, so the app instead triangulates each corner by **N-view DLT**
  over every registered view that saw the tag and takes the median edge
  (`robust_tag_scale`). A scene extent that is implausible for the tag size is
  flagged.

### 3.7 Foreground isolation

A "scattered" mesh is usually the whole room, not the object. The capture
centres the object, so the app scores every sparse point by the fraction of its
observations in which it projects inside a central disk
(`--object-central-radius`, as a fraction of the image half-diagonal). Points
that are well-supported **and** central in most views are kept, the largest
26-connected component is taken, and an implausibly tiny cluster is rejected
(a few noise points near the centre are not an object). The mask trims the
sparse cloud and gives the dense stage a tight box.

Crucially, the **TSDF resolution is rescaled** to the object box so the voxel
size stays at the full-scene scale: a tight box at the configured resolution
would over-resolve the depth noise and shatter the surface into thousands of
patches (measured: 845 patches at 0.0004 units/voxel vs 1 patch at 0.0032).

### 3.8 Dense depth

Each registered view gets a depth map by **plane sweeping** (Collins, 1996):
sample a set of inverse-depth hypotheses and score each by photometric
consistency with neighbouring views. Costs available:

- **NCC** — normalised cross-correlation, robust to affine lighting change.
- **Census** (Zabih & Woodfill, 1994) — compares intensity orderings; strong on
  gentle shading but ambiguous on perfectly flat regions.
- **SGM** (Hirschmüller, 2008) — census plus a smoothness prior aggregated along
  4 or 8 paths; resolves low-texture regions that WTA cannot. It is the default
  on low/medium-texture captures (measured on the keyboard clip: census WTA → 30
  patches, SGM → 2).

Winners are refined to sub-pixel by fitting a parabola to the cost curve. A
pixel is kept only if `--lr-required` neighbours agree on the depth, and a
speckle filter removes isolated pixels. `--auto-config` measures frame texture
(mean gradient magnitude) and picks cost, window, NCC threshold and plane count.

### 3.9 TSDF fusion

Depth maps are fused into a truncated signed distance field (Curless & Levoy,
1996; KinectFusion, Newcombe et al., 2011). Each voxel accumulates
confidence-weighted "in front/behind" votes; the volume is bounded by the
**robust** (1st–99th percentile) extent of the dense cloud, and the truncation
band is `--trunc-voxels` voxels wide. The truncation must exceed the depth
noise, or views disagree and tear the surface. Optional box-filter smoothing
runs before meshing; hole-filling diffusion is off by default because it invents
surface where no camera looked.

### 3.10 Meshing, smoothing, simplification

The zero level set is extracted with **marching tetrahedra** (a variant of
marching cubes, Lorensen & Cline, 1987) which is topologically well-behaved on
the tetrahedral decomposition. Connected surface patches smaller than
`--min-component-faces` are dropped. The surface is then smoothed with
**Taubin** λ/μ passes (Taubin, 1995) — which smooth without the shrinkage of
plain Laplacian smoothing — and a **bilateral** pass that preserves creases
(keycaps, blade bevels) while flattening noise. Finally the mesh is simplified
to a face budget with **quadric error metrics** (Garland & Heckbert, 1997).

Colours and normals are assigned per vertex by projecting each vertex into the
source frames and averaging the observations; normals come from the TSDF
gradient.

### 3.11 Report and named failures

Every run writes `<stem>_report.json`: elapsed time, statistics (views, points,
reprojection RMSE, depth error percentiles, face counts), a mesh-quality pack
(patches, Euler characteristic, boundary-edge ratio, degenerate/needle
fractions, closedness), provenance (video SHA-256, exact argv, version),
resources (peak RSS), and a heuristic **self-assessment** confidence. Failures
write `<stem>_error.json` with a stable machine code and hints, so the UI shows
an actionable message instead of a stack trace.

---

## 4. Measuring and trusting the result

No single number proves a mesh is good, so the report triangulates several:

- **Sparse reprojection RMSE** (pixels) checks the camera/point solve.
- **Depth-vs-sparse error** samples the dense depth at triangulated points and
  reports median and p90 relative error. A low median with a high p90 means good
  typical depth and a bad outlier tail — exactly what makes a fragmented mesh.
- **Mesh topology**: patch count, Euler characteristic and boundary-edge ratio
  measure continuity. A continuous object is one patch with a small Euler
  magnitude; a shattered reconstruction is hundreds of patches.
- **Self-assessment** folds coverage, depth error and topology into a
  confidence and a `low/medium/high` verdict. It is a consistency signal, not a
  guarantee: it has been observed to report "high" on a run whose mesh covered
  almost no surface (see §5.10), which is why coverage/continuity were added to
  the story.

---

## 5. Iteration history: what broke, why, and the fix

Each entry is a real failure met while building the project: the symptom, the
root cause, the fix, and the measured before/after.

### 5.1 First dense results were blobs and holes

**Symptom:** early meshes were blobby with large holes on shaded surfaces.
**Cause:** winner-take-all NCC has no notion of spatial smoothness, and depth
was quantised to whole planes.
**Fix:** parabolic **sub-pixel** depth refinement, **8-path SGM**, a speckle
filter, and feature-preserving **bilateral** mesh smoothing.
**Result:** depth median error on the specular swiss knife fell 4.8 → 4.0 %,
p90 44.8 → 29.4 %, patches 34 → 33.

### 5.2 "Not enough views" on new landscape clips

**Symptom:** several new clips registered almost nothing.
**Cause:** the new clips were **854×480 landscape** while the intrinsics used
were from **480×854 portrait**, i.e. `cx` and `cy` were swapped. A swapped
principal point barely affects descriptor matching but wrecks PnP.
**Measured:** remote clip whole-clip registration was **12/151** with swapped
intrinsics vs **24/150** with correct ones; the usable 16–27 s segment then
registered **84/84**.
**Fix:** principal-point and focal plausibility warnings, plus a troubleshooting
row ("pass intrinsics for the decoded frame").

### 5.3 Low keypoint density

**Symptom:** some clips failed to bootstrap.
**Cause:** blurry or textureless captures yield ~30 keypoints per frame.
**Fix:** an explicit warning below 80 keypoints/frame. The tube clip measured
**27 keypoints/frame → 2/91 views**; no parameter tuning helps a clip with no
texture, so the tool says so.

### 5.4 Cache re-runs were not deterministic

**Symptom:** a cached re-run produced a slightly different reconstruction than
the fresh run (RMSE 1.996 vs 1.951).
**Cause (two bugs):**
1. `serde_json` does not round-trip every `f64` exactly — f32-promoted pixel
   coordinates came back one ULP off, perturbing the whole solve.
2. The frame cache wrote all gray frames then all RGB, but read them
   interleaved per frame, so a cache hit produced wrong per-vertex colours.
The two-frame round-trip test had masked bug 2 because the misalignment happened
to land on the last frame.
**Fix:** store pixel coordinates as raw `f64` **bits**; load frames in the same
block order they are written; harden the test to three frames and assert every
frame.
**Result:** fresh runs, cache hits and cache misses are now **byte-identical**.

### 5.5 Studio presets copied the previous mode

**Symptom:** clicking Fast → Balanced kept Fast's settings.
**Cause:** presets were *partial* overrides merged into the live config, and
`balanced` was literally `{}`, so it changed nothing.
**Fix:** presets became **complete** configs that *replace* the current one.
**Second bug found later:** `fast`/`quality` inherited `auto_config: true` from
the defaults, and auto-config rewrites planes/cost/window/ncc, so all the
"explicit" presets behaved alike. Now `fast`/`balanced`/`quality` pin
`auto_config: false`; only `auto` enables it.
**Verified:** the preset matrix (`docs/preset-matrix.md`) shows four distinct
results on `swiss_knife`.

### 5.6 Metric-scale anchoring was fragile (and now measured honestly)

**Symptom:** with an in-scene 10 cm tag the "metric" mesh came out tens of
metres wide.
**Cause:** two layers. kornia's `tag_scale` triangulates the tag from a single
widest-baseline pair and averages four edges — fragile on an orbit. The app now
does a robust **N-view DLT** scale instead. But even that gave the same wrong
answer, which localised the real fault: the tag triangulates to ~0.005 units
while the object spans ~3 units — a **gauge inconsistency** between two regions
of the same reconstruction, i.e. scale drift with no loop closure.
**Fix (this repo):** robust N-view scale, a plausibility warning when the scene
extent cannot belong with the tag size, and the finding filed upstream as UP-1
(loop closure). The true fix is a pose-graph loop closure in kornia; it is
recorded in `docs/upstream-kornia-issues.md` rather than hacked locally.

### 5.7 Self-assessment over-claimed

**Symptom:** a run reported confidence 0.77 ("high") for a mesh of 768 faces.
**Cause:** the heuristic weighted depth error and view coverage but not
reconstructed **surface extent**.
**Fix:** the gap is recorded in the TAI report and the report is treated as a
consistency signal, not a completeness guarantee. (Extending the heuristic is
tracked.)

### 5.8 Decode and cache economics

**Symptom:** every slider change re-decoded the clip (~40–70 s).
**Fix:** cache decoded frames (keyed by video SHA-256 + frame step) and tracks
(additionally keyed by detector/feature/matching parameters).
**Measured:** an 11 s clip's sparse stage dropped **42.7 s → 13.2 s** on a hit.
Frames are stored raw, so the cache costs roughly 1.5 MB/frame.

### 5.9 Scattered meshes from whole-scene fusion

**Symptom:** even with correct intrinsics, meshes were fragmented — e.g. the
swiss knife produced **49 patches, Euler −345**.
**Cause:** the TSDF volume covered the whole table/background, so the object was
fused inside a large volume and either under-resolved or torn by table noise.
**Fix:** foreground isolation (§3.7) plus resolution rescaling to the object box.
**Measured on swiss_knife (identical views and RMSE):** **49 patches / Euler
−345 → 1 patch / Euler −64**. The `remote` clip's isolation correctly declines
(too few central points) and falls back to the whole scene.

### 5.10 Representation and capture limits

**Symptom:** some objects simply do not reconstruct — the tube (white on red,
27 keypoints/frame), the remote (dark, small, plain background), transparent or
deformable objects.
**Cause:** these are data-bound, not parameter-bound.
**Fix:** the tool detects and reports them (keypoint warnings, named
`empty-mesh`/`low-registration` failures, low self-assessment) instead of
pretending. Improvement needs a better capture (texture, light, fill the frame)
or upstream loop closure, not more tuning.

---

### 5.11 Why one capture wins (the `swiss_knife` case)

Mesh quality is decided before any code runs, by three levers: **pixels on the
object**, **texture**, and **parallax**. `swiss_knife` is the only reference clip
that scores on all three, which is why it is the golden example.

| Lever | `swiss_knife` | the other clips |
|---|---|---|
| Pixels on the object | 1280×720, object large in frame | 480×854 portrait (`lip_balm`, `keyboard`, `stand`) or 854×480 (`remote`, `tube`) — a third or less of the pixels |
| Texture | markings, multi-tool body, red scales, a keyboard in frame | `tube` ~27 keypoints/frame; `lip_balm` white-on-white; `remote` dark plastic on plain red |
| Parallax | real orbit, RMSE ~2.09 px, depth ~4 % | `remote` only 16–27 s has parallax (p90 56 %); `tube` 2/91 views |

Measured comparison:

| Clip | Resolution | Registration | Depth | Isolation result |
|---|---|---|---|---|
| **swiss_knife** | 1280×720 | ~19–22/63 | ~4 % | **1 patch, Euler −64** |
| keyboard | 480×854 | 63 views | 4.5 % | works (low res) |
| stand | 480×854 | 140/380 | 2.0 % | 20 patches (thin object) |
| lip_balm | 480×854 | 66 views | 14.8 % | 41 patches (white-on-white) |
| remote | 854×480 | 20/151 | 4.2 % med / 56 % p90 | near-empty / fails (dark, short parallax) |
| tube | 854×480 | 2/91 | — | fails (textureless) |

Isolation also favours the knife: it is compact, centred and seen from many
views, so the centrality mask and the object box are clean, and the adaptive
TSDF resolution lands in a sane range. On `remote` too few central points
survive, so it correctly declines; on `tube` bootstrap fails first.

The one weakness of the knife is the **polished blade** (specular depth holes),
which is why even the un-isolated golden run is open (~33 patches); the diffuse,
textured handle is what carries it.

**The lesson is capture-side:** fill the frame, add diffuse light and physical
texture, and keep real camera translation. No preset recovers a missing lever.

## 6. Current limitations

- **Scale drift** on long orbits without loop closure (UP-1) can make a
  "metric" map wrong even when a tag anchors it; a plausibility warning is the
  current guard.
- **Foreground isolation** is heuristic and centred-object specific; it declines
  on clips with too few central points rather than guessing.
- **Non-watertight meshes.** Marching tetrahedra does not guarantee a closed
  surface; screened Poisson (Kazhdan & Hoppe, 2013) is the planned route to true
  watertightness.
- **Textureless / specular / dark** surfaces remain the dominant failure mode.

---

## 7. References

1. D. G. Lowe. Distinctive image features from scale-invariant keypoints.
   *IJCV*, 2004.
2. E. Rublee, V. Rabaud, K. Konolige, G. Bradski. ORB: an efficient alternative
   to SIFT or SURF. *ICCV*, 2011.
3. R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*.
   Cambridge University Press, 2004.
4. M. A. Fischler, R. C. Bolles. Random sample consensus. *CACM*, 1981.
5. V. Lepetit, F. Moreno-Noguer, P. Fua. EPnP: an accurate O(n) solution to the
   PnP problem. *IJCV*, 2009.
6. A. Wald. Sequential tests of statistical hypotheses. *Annals of Mathematical
   Statistics*, 1945.
7. B. Triggs, P. McLauchlan, R. Hartley, A. Fitzgibbon. Bundle adjustment — a
   modern synthesis. *Vision Algorithms*, 2000.
8. M. Lourakis, A. Argyros. SBA: a software package for generic sparse bundle
   adjustment. *ACM TOMS*, 2009.
9. Z. Zhang. A flexible new technique for camera calibration. *TPAMI*, 2000.
10. E. Olson. AprilTag: a robust and flexible visual fiducial system. *ICRA*,
    2011.
11. J. L. Schönberger, J.-M. Frahm. Structure-from-motion revisited. *CVPR*,
    2016.
12. R. T. Collins. A space-sweep approach to true multi-image matching. *CVPR*,
    1996.
13. R. Zabih, J. Woodfill. Non-parametric local transforms for computing visual
    correspondence. *ECCV*, 1994.
14. H. Hirschmüller. Stereo processing by semi-global matching and mutual
    information. *TPAMI*, 2008.
15. B. Curless, M. Levoy. A volumetric method for building complex models from
    range images. *SIGGRAPH*, 1996.
16. R. A. Newcombe et al. KinectFusion: real-time dense surface mapping and
    tracking. *ISMAR*, 2011.
17. W. E. Lorensen, H. E. Cline. Marching cubes: a high resolution 3D surface
    construction algorithm. *SIGGRAPH*, 1987.
18. G. Taubin. Curve and surface smoothing without shrinkage. *ICCV*, 1995.
19. M. Garland, P. S. Heckbert. Surface simplification using quadric error
    metrics. *SIGGRAPH*, 1997.
20. M. Kazhdan, H. Hoppe. Screened Poisson surface reconstruction. *ACM TOG*,
    2013.
21. P. J. Huber. Robust estimation of a location parameter. *Annals of
    Mathematical Statistics*, 1964.
22. S. Umeyama. Least-squares estimation of transformation parameters between
    two point patterns. *TPAMI*, 1991.
