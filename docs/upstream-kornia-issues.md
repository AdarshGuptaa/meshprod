# Upstream kornia-rs issues (to upstream later)

This file collects every improvement that requires changing **kornia-rs itself**
(the `kornia-calib` / `kornia-3d` crates on the `semproj/sfm2mesh` fork branch),
as opposed to `meshprod` (this repo). We keep them here so the upstream PR stays
clean while we keep working on `meshprod` locally.

**Branch note:** these live on the `semproj/sfm2mesh` fork branch. **Never push to
the upstream PR #1136 branch** — land these as their own focused PRs, one issue
at a time, once `meshprod` is in shape.

Each entry has: what's wrong, why it matters (with measured evidence where we
have it), the proposed change, and an acceptance bar.

---

## UP-1 · Loop closure + pose-graph optimization

- **Area:** `kornia-calib/src/sfm.rs` (`reconstruct`), using `kornia-3d/src/pgo.rs`.
- **Problem:** the incremental grow loop registers cameras greedily (best
  2D↔3D link via PnP) and runs periodic global BA, but there is **no revisit
  detection**. A 360° orbit never "closes": the last views that look back at the
  region the first views saw are registered against a drifted map, so scale and
  orientation accumulate error at the seam. `kornia-3d` already ships a full
  SE(3) pose-graph solver (`pose_graph_optimize` with `RelPoseFactor`), but
  `reconstruct` never calls it.
- **Evidence:** this is the dominant accuracy gap for the orbit-around-an-object
  captures `meshprod` is built for.
- **Proposed change:** after the incremental grow + terminal BA, detect
  covisible non-consecutive view pairs (frame i matching frame j far outside the
  match window), add them as `RelPoseFactor` constraints, and run a similarity-
  constrained pose-graph solve before the final BA.
- **Acceptance:** on a full orbit, the first and last views reconnect with no
  visible seam, and drift (scale/rotation) is measurably lower than the
  incremental-only result.

## UP-2 · Track merging

- **Area:** `kornia-calib/src/tracks.rs` (`build_tracks`).
- **Problem:** a track that is split by one broken match becomes two tracks for
  the same 3D point, and they are never merged back. This fragments points,
  shortens tracks, and reduces PnP support and BA constraint density. The
  existing dedup handles orientation siblings *within* a component but not
  duplicates *across* components.
- **Evidence:** `build_tracks`' own comment records median track length ~2 with
  66% two-view points, versus COLMAP's median ~5 with 2.6% — partly from the
  ambiguity drop, partly from unmerged tracks.
- **Proposed change:** after union-find, merge components that are geometrically
  consistent (overlapping observations that triangulate to the same 3D point),
  following COLMAP's `MergeTracks`.
- **Acceptance:** median track length rises; the number of duplicate
  near-coincident 3D points in a published map drops; registration and BA RMSE
  improve on the same input.

## UP-3 · Publish feature-path covariance

- **Area:** `kornia-calib/src/covariance.rs`, `sfm.rs::feature_stats`, `types.rs::Reconstruction`.
- **Problem:** `Reconstruction::per_view` covariance is computed only for the
  rigid-board/tag path. The feature path leaves it `None` (the code comment says
  so explicitly). Without per-point/per-pose uncertainty, downstream stages
  (dense depth weighting, TSDF truncation) treat every view as equally
  trustworthy, which is exactly what produces the outlier depth tail we measured
  (`p90` far above `p50`).
- **Proposed change:** compute the pose/point Hessian inverse (or at least a
  per-view reprojection-RMS spread and a per-point covariance) in the feature
  path and publish it on `per_view` / a new field.
- **Acceptance:** `per_view[c].covariance` is `Some` for registered views; a
  view with few/high-residual observations reports higher uncertainty.

## UP-4 · Robust tag scale anchor

- **Area:** `kornia-calib/src/sfm.rs` (`tag_scale`).
- **Problem:** scale anchoring is the least robust step in an otherwise robust
  pipeline. It:
  - uses **only the first tag** (`tags.first()`) — extra tags are ignored;
  - triangulates the 4 corners with **only the single widest-baseline pair**;
  - averages the up-to-4 edge lengths with a **plain arithmetic mean** — no
    median, no MAD, no IRLS, no Umeyama (confirmed against the current code).
  One bad corner or one poor pair drags the whole scale.
- **Proposed change:** (a) collect tag corners from **every registered view**
  that sees the tag, (b) fit a single **Umeyama similarity** (or a robust
  median of edge lengths as a first pass) over all corners, (c) combine multiple
  tags (e.g. median of per-tag scales). This reuses the MAD/IRLS/Umeyama
  machinery already in the tree instead of leaving `tag_scale` on a raw mean.
- **Acceptance:** with a deliberately noisy tag view injected, the recovered
  scale moves less than with the current mean-of-4-edges.

## UP-5 · Triangulation: midpoint → optimal / multi-view

- **Area:** `kornia-3d/src/pose/triangulation.rs` (`triangulate_matched_points`).
- **Problem:** points are triangulated by the **midpoint** method
  (`triangulate_midpoint_known_pose`), which is biased for slanted rays and
  suboptimal under noise. The standard is optimal two-view (Lindstrom) or a
  linear DLT followed by a non-linear refine.
- **Evidence:** triangulation error flows straight into the points that seed the
  next PnP, so it compounds along the chain (the code itself documents this
  compounding for the pose side).
- **Proposed change:** add an optimal two-view triangulation (and, where cheap,
  an N-view refine using all observing cameras of a track).
- **Acceptance:** lower triangulation residual on synthetic scenes with slanted
  geometry; no regression in runtime.

## UP-6 · Adaptive sparse thresholds

- **Area:** `kornia-calib/src/types.rs` (`ReconstructionConfig`), `sfm.rs`.
- **Problem:** `min_parallax_deg`, `robust_scale_sq`, and
  `max_reprojection_error` are fixed defaults, not tuned to the scene. Low-
  parallax phone orbits and wide-baseline rigs need different values, and the
  caller has to guess.
- **Proposed change:** derive sensible defaults from the measured feature scale /
  parallax distribution, or expose a small number of named presets (the
  `sequential()` preset already exists; add e.g. `orbital()` and `rig()`).
- **Acceptance:** the same capture registers more views with the tuned preset
  than with the one-size defaults, without user intervention.

---

## Notes (not issues — already upstream, need app wiring)

- **Monocular depth priors** (`reconstruct`'s `obs_depth` argument,
  `depth_prior_rel_sigma`) already exist upstream and make a map metric without
  a fiducial. `meshprod` just doesn't pass a depth model yet — that's a
  `meshprod` task, not an upstream change.
- **Multi-view Zhang intrinsics** (`calibrate_multishot`) already exist upstream.
  Wiring it into `meshprod`'s calibration step is a `meshprod` task.
- **Determinism** is already handled upstream (seeded PnP, `BTreeMap` track
  ordering, sorted point order). No upstream change needed.
