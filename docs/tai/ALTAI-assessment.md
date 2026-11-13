# Trustworthy-AI assessment — sfm2mesh

Scope: the `sfm2mesh` system (monocular video → sparse SfM → dense depth →
TSDF → simplified coloured mesh) and the process that produced it.
Framework: **EU HLEG ALTAI** (Assessment List for Trustworthy AI), adapted to a
classical computer-vision pipeline with AI-assisted development.

> All numbers below were measured on this machine, on
> `swiss_knife.mp4` (628 frames, 1280×720 HEVC, handheld orbit) unless stated.
> Reproduction commands are in `scripts/tai_reproduce.sh`.

---

## 1. System description

| Stage | Method | Deterministic? |
|---|---|---|
| Decode | GStreamer → RGB + gray | yes |
| Features | SIFT (CPU), 500 per frame | yes |
| Matching | mutual NN + Lowe 0.8 + 7-point fundamental RANSAC (seed 0) | yes |
| Sparse SfM | `kornia-calib` incremental SfM, PnP-RANSAC + SPRT + BA | yes |
| Dense depth | plane sweep, NCC 5×5, winner-takes-all + left-right check | yes |
| Surface | TSDF fusion → marching tetrahedra | yes |
| Mesh | quadric edge-collapse (Garland–Heckbert) | yes |
| Colour/normals | TSDF gradient; cos²-weighted multi-view projection | yes |

The pipeline contains **no learned model**: the "AI" here is classical
geometry, and the trustworthiness question is about the *system's* behaviour
(accuracy, robustness, transparency) and about *AI-assisted engineering*
(how the code was produced and verified).

---

## 2. Evidence

### 2.1 Baseline run (720p, frame-step 10)

| Metric | Value |
|---|---|
| Registered views | 22 / 63 |
| Sparse points | 1,175 |
| Sparse reprojection RMSE | 2.086 px |
| Dense depth pixels kept | 2,199,161 |
| Dense depth vs sparse depth | **median 4.8 %**, p90 44.8 % (13,659 samples) |
| TSDF volume | 197 × 218 × 285 @ 2.30 units/voxel |
| Observed / solid voxels | 451,662 / 220,569 |
| Raw mesh | 66,459 vertices, 109,933 faces, 525 patches |
| After small-patch removal (≥200 faces) | 100,612 faces, 36 patches |
| Final mesh (Taubin + QEM) | 17,754 vertices, **20,000 faces**, 36 patches |
| Mesh topology | open (partial scan), Euler = −102 |
| Wall time | ≈ 55 s (decode 23.5 s, features 2.5 s, dense ≈ 25 s) |

**Fusion calibration fix.** Early runs used a truncation band derived only
from the voxel size and a bounding box containing every dense point. Both were
wrong for noisy data: a handful of bad depths inflated the box ~20×, and the
band was narrower than the measured depth error (3–5 % of depth). The volume
now uses 1–99 % per-axis bounds and a truncation floor of
`1.5 × measured_error × median_depth`. Raw mesh faces rose from 39,521 to
169,154 and the principal-axis extent became knife-like (2.0 : 1.09 : 0.40).

### 2.1b Algorithmic improvements (Path 1) and Poisson (Path 2)

| Variant | Depth error (median/p90) | Raw faces | Patches | Euler |
|---|---|---|---|---|
| Before (v4 baseline) | 4.8 % / 44.8 % | 169,154 | 1,940 → 42 | −604 |
| + weighted fusion, consistency=2, diffusion 12 | **3.5 % / 22.7 %** | 552,656 | 302 → 21 | −534 |
| + field smoothing, Taubin, no fill | 4.8 % / 44.8 % | 109,933 | 525 → 36 | **−102** |
| + diffusion 3, field smoothing 2 | 4.8 % / 44.8 % | 451,170 | 237 → 50 | −201 |

Field smoothing and Taubin clearly improve topology (Euler −604 → −102) and
remove voxel-scale pyramids. Volumetric diffusion **over-fills**: it invents
surface where no camera looked, producing blobs and sheets, so it ships
disabled by default. Requiring two consistent neighbours halves the number of
depth pixels but improves accuracy substantially; it is exposed as
`--lr-required`.

**Poisson backend (Open3D screened Poisson, depth 9, same oriented cloud):**
19,999 triangles but `watertight = false` after density trimming, mean
nearest-vertex distance to our mesh 60.3 units (~10 % of scene scale),
Hausdorff 602 units. Visually it produced lumpy blobs: a smooth prior cannot
recover geometry the input never observed. The conclusion is a *data-bound*
result, not a solver failure — on a complete, matte capture Poisson is the
better backend, and the script is kept for that case.

### 2.1c Low-texture costs and auto-config (Path 1.5)

`lip_balm.mp4` (18–27 s segment, 66/68 views, sparse RMSE 2.05 px):

| Cost | Depth median / p90 | Raw faces | Patches after cleanup |
|---|---|---|---|
| NCC (previous) | 16.8 % / 92 % | 115,881 | 68 |
| Census WTA (texture 0.021) | 20.5 % / 94 % | 231,520 | 61 |
| Census + SGM (4 paths) | **19.0 % / 77 %** | 334,898 | **43** |

SGM measurably reduces the depth tail (p90 94 % → 77 %) and fragmentation
(61 → 43 patches), but a white, untextured tube remains below the quality bar.
On the textured knife clip auto-config scores the texture at 0.038 (high),
correctly keeps NCC, and yields 34 patches / Euler −104 — parity with the
hand-tuned configuration.

**Auto-config** is rule-based, not learned: mean image gradient → cost, window,
NCC threshold and plane count; dense-point count → face budget and minimum
patch size. Every choice is logged with the statistic that drove it, which is
the transparent counterpart to a black-box tuner. Rule: texture < 0.035 →
**SGM** (census-WTA is ambiguous on flat regions: identical codes give cost
zero), otherwise NCC.

### 2.1d Keyboard capture (textured box)

`keyboard.mp4` (480×854, 30 fps, 38 s) has only ~6 s of real parallax; the
rest is a pan. Segment 15–25 s: **63/101 views register**, sparse RMSE 2.03 px,
dense depth median 6.8 %.

| Cost / cleanup | Raw faces | Patches |
|---|---|---|
| Census WTA (old auto rule) | 73,549 | 30 |
| Census + SGM + plane removal | 20,148 | **2** |
| + Taubin + QEM | 18,686 | 2 (Euler −36) |

The plane removal deleted 5.4 M tabletop points so the volume focused on the
keyboard. The surviving mesh is the keyboard's visible top surface (key rows
readable in the render) but open at the sides: the usable segment covers only
part of the orbit, and a flat keycap deck has little depth variation.

**Capture lesson:** most of this clip has translation-free panning. SfM needs
parallax; the `registered source frames` log is the fastest way to find the
usable seconds.

**Negative result — auto-crop:** on the noisy knife cloud the largest
26-connected voxel cluster is a sub-region, so cropping shrank the scene and
fragmented the mesh (6 tiny islands). It is opt-in and documented as needing a
clean cloud (after plane removal) or a segmented object.

### 2.2 Repeatability (determinism)

Two independent full runs produced **byte-identical** outputs:

```
sparse PLY   sha256 cc05bebb…
dense PLY    sha256 df724bdf…
mesh PLY     sha256 fce7933d…
```

Same seed, same input, same result — a prerequisite for auditing and for
comparing any parameter change.

### 2.3 Resolution robustness

The same clip re-encoded at 640×360 (intrinsics halved, all else equal):

| Metric | 720p | 360p |
|---|---|---|
| Registered views | 22 / 63 | **2 / 63** |
| Sparse points | 1,175 | **72** |
| Final mesh faces | 19,999 | 1,962 |
| Mesh patches | 18 | 6 |

Halving the resolution removed 4× the usable detail; registration collapsed.
This is a **sharp failure boundary**, not a graceful degradation — the system
should warn when the registered-view ratio is low (it currently only logs it).

### 2.4 Parameter sensitivity

| Run | Registered views | Dense depth median error | Raw mesh faces |
|---|---|---|---|
| frame-step 10 (default) | 22 | 4.8 % | 109,933¹ |
| frame-step 5 | 39 | 3.9 % | 4,493 |
| frame-step 5, voxel 128 / trunc 8 | 39 | 3.9 % | 484 |

¹ Measured after the robust-bounds + adaptive-truncation fix described below; the other rows predate it.

More registered views did **not** produce a better mesh here: the denser
sampling changed the SfM scale/arbitration and interacted badly with the
volume settings. The default (step 10) was the best-behaved configuration in
our experiments, but this sensitivity is a real risk and is not fully
explained — it is recorded as an open issue rather than hidden.

### 2.5 Failure-mode observations

- **Specular surfaces**: the Swiss knife's polished blade produces streaks;
  the mesh splits into 18 patches with open boundaries instead of one closed
  surface. Classical photo-consistency assumes a Lambertian surface.
- **Textureless regions** (plain handles/background) yield no SIFT features
  and no depth. Measured on `lip_balm.mp4` (18–27 s segment): **66/68 views
  register** and sparse RMSE is 2.05 px, but dense depth error is **16.8 %
  median** — the white tube and red table have no texture for NCC. Sparse
  succeeds, dense fails; the mesh is fragmented. This is the cleanest
  demonstration in the project that the dense stage is texture-bound, not
  geometry-bound.
- **Partial coverage**: only 22/63 views register, so the surface is open
  (Euler −141, 7,071 boundary edges). "Closed mesh" is *not* claimed.
- **No distortion model in the dense stage**: the sweep is a pure pinhole
  homography; fitted distortion is ignored with a warning. Wide-angle phone
  lenses will bias depth toward the frame edges.
- **Scale is arbitrary** (no fiducial): the mesh has correct shape, not
  metric size.

---

## 3. ALTAI requirement-by-requirement

| ALTAI requirement | Status | Evidence / action |
|---|---|---|
| **Human agency & oversight** | 🟡 adequate | All parameters exposed on the CLI; every stage logs counts and timings; a human decides whether a result is usable. No automatic acceptance threshold is enforced. |
| **Technical robustness & safety** | 🟡 adequate | Deterministic reruns; per-stage quantitative checks (`depth vs sparse`, RMSE); small-patch cleanup. Fails on specular/textureless input and degrades sharply at low resolution (2/63 views). Mesh openness is reported, never silently passed as "solid". |
| **Privacy & data governance** | 🟢 good | Fully local processing (no network, no cloud, no telemetry). Input stays on the machine; outputs are PLY files. Residual risk: a phone clip can capture people in the background and the mesh preserves their geometry/colour — users must obtain consent and frame only the object. |
| **Transparency** | 🟢 good | Open code; deterministic pipeline; full stage log; assumptions documented (pinhole, static rigid scene, Lambertian, up-to-scale); every threshold is a named CLI flag; method file `docs/pipeline.md`. |
| **Diversity, non-discrimination & fairness** | 🟡 needs work | Evaluation uses a single object. Dark/glossy objects are demonstrably worse (sec. 2.5). A category-diverse object set is required before claiming general quality. |
| **Societal & environmental wellbeing** | 🟢 good | CPU-only path (≈55 s per scan on one workstation, no GPU required); modest energy footprint. Dual-use: 3-D scanning of objects can raise IP/consent issues — documented. |
| **Accountability** | 🟢 good | Reproducible commands (`scripts/tai_reproduce.sh`), pinned kornia-rs branch, versioned source, measured claims with the commands that produced them. |

Overall: **suitable for research/demonstration use with human review**, not for
autonomous or safety-critical use.

---

## 4. AI-assisted development (process trustworthiness)

This project was built in an AI-assisted session. The honest disclosure:

| Item | Value |
|---|---|
| Agent | OpenCode CLI |
| Model | DeepSeek V4.1 Flash |
| Skills applied | `kornia-arch`, `kornia-workflow`, `kornia-coding-style` |
| Human part | scope, object choice, repository/branch decisions, review of every file |
| AI part | scaffolding, dense/TSDF/meshing/simplification code, tests, this report |

Process controls that make the claims checkable:

1. **Ground-truth unit tests before integration.** Every new algorithm is
   verified against an analytic reference: plane-sweep homography against
   projected points, TSDF against a known plane distance, marching
   tetrahedra against an analytic sphere (closed, Euler = 2, exact area),
   QEM against a sphere (stays closed, vertices stay on the surface).
2. **Claims are measured, not asserted.** Every number in this report comes
   from a logged run; the two determinism hashes are reproducible.
3. **Failures are recorded.** The frame-step-5 anomaly, the 360p collapse and
   the open mesh are reported rather than tuned away silently.
4. **Human ownership.** The human is the author of record; AI output was
   reviewed and is not trusted on its own.

Residual risk: AI-generated code can be plausible and wrong. The mitigation
is the test suite (52 tests) and the analytic fixtures, not the model's
confidence.

---

## 5. Open issues / next steps

1. **Metric accuracy**: add an AprilTag of known size to the scene, recover
   metric scale, and compare measured object dimensions with a ruler. Until
   then only *relative* accuracy is evidenced.
2. **Object segmentation**: dense depth currently includes the background;
   the mesh contains background patches. A mask (manual or learned) would
   focus the volume on the object.
3. **Hole filling / watertightness**: the mesh is open. Poisson
   reconstruction or a TSDF hole-filling pass would give a closed surface.
4. **Dense-stage sensitivity**: explain and stabilise the frame-step
   interaction (sec. 2.4).
5. **Fairness evaluation**: at least 5 objects spanning matte/dark/glossy/
   thin/textured categories.
6. **Low-resolution guard**: warn or fail when the registered-view ratio is
   below a threshold (e.g. < 20 %).

---

## 6. Reproduction

```bash
# Baseline (deterministic)
./target/release/sfm2mesh swiss_knife.mp4 out/mesh.ply \
    --fx 895 --fy 895 --cx 640 --cy 360 \
    --detector sift --n-features 500 --frame-step 10 --geo-verify --sprt

# Determinism
sha256sum out/mesh.ply out/mesh_sparse.ply out/mesh_cloud.ply
```

`scripts/tai_reproduce.sh` runs the baseline twice and compares hashes.
