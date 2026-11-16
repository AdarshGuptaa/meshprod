//! Structure-from-Motion reconstruction wrapper.
//!
//! Builds a [`PinholeCamera`] per video frame from user-supplied intrinsics
//! and drives `kornia_calib`'s incremental SfM (`reconstruct`) over the
//! feature tracks. Because no AprilTag anchors the scale, the recovered map is
//! [`ScaleSource::UpToScale`]: correct shape, arbitrary units.

use std::error::Error;
use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{triangulate_n_view_linear, Pose3d};
use kornia_3d::ransac::SPRTConfig;
use kornia_calib::{
    reconstruct, FeatureTrack, Reconstruction, ReconstructionConfig, ScaleSource, TagObservation,
};

use crate::calibrate::median;

/// Tunable overrides for [`ReconstructionConfig`].
///
/// All fields are optional; `None` keeps the default set by
/// [`ReconstructionConfig::new(0.0).sequential()`]. This lets the CLI expose
/// only the knobs users care about while tests and other callers stay on the
/// defaults.
#[derive(Debug, Clone, Default)]
pub struct ReconstructionOverrides {
    /// Maximum bundle-adjustment LM iterations.
    pub max_iterations: Option<usize>,
    /// Minimum PnP inliers required to register a view.
    pub min_registration_inliers: Option<usize>,
    /// Constant-velocity motion-prior sigma (`0.0` disables).
    pub motion_prior_sigma: Option<f64>,
    /// Camera-up prior sigma (`0.0` disables).
    pub up_prior_sigma: Option<f64>,
    /// Maximum reprojection error (normalized units) for triangulation.
    pub max_reprojection_error: Option<f64>,
    /// Enable Wald's SPRT for the per-view PnP-RANSAC registration, with the
    /// given expected-inlier ratio and Type-I error. `None` keeps it disabled.
    pub sprt: Option<SPRTConfig>,
    /// Fit a global focal scale plus radial/tangential distortion against the
    /// reconstruction (COLMAP-style alternating intrinsics refinement).
    pub refine_intrinsics: Option<bool>,
}

/// Build a pinhole camera from intrinsics, with zero distortion.
pub fn make_camera(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        ..PinholeCamera::IDENTITY
    }
}

/// Run incremental SfM over the given feature tracks.
///
/// Each video frame is treated as one camera with the given intrinsics; the
/// `tracks[i].obs` camera indices index into that per-frame camera list. No
/// tag is supplied, so the reconstruction is up to scale.
///
/// # Arguments
///
/// * `tracks` - Multi-view feature tracks produced by `kornia_calib::build_tracks`.
/// * `fx` / `fy` - Focal lengths in pixels.
/// * `cx` / `cy` - Principal point in pixels.
/// * `n_frames` - Number of video frames (one camera per frame).
///
/// # Returns
///
/// The [`Reconstruction`] with per-frame poses and reconstructed 3D points.
///
/// # Errors
///
/// Returns an error if the reconstruction cannot bootstrap (e.g. no usable
/// tracks, or a degenerate camera configuration).
#[allow(clippy::too_many_arguments)] // thin wrapper over reconstruct's (also 8-arg) signature
pub fn run_sfm(
    tracks: &[FeatureTrack],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    n_frames: usize,
    tags_for_scale: &[TagObservation],
    tag_size_m: f64,
    progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    overrides: ReconstructionOverrides,
) -> Result<Reconstruction, Box<dyn Error>> {
    let cameras = vec![make_camera(fx, fy, cx, cy); n_frames];
    // `.sequential()` tunes the config for video walkthroughs (smaller
    // parallax threshold, more BA iterations). We always pass `tag_size_m = 0`
    // so kornia leaves the map up-to-scale; metric scale is applied below with
    // a robust N-view estimate (kornia's single widest-baseline pair collapses
    // on orbit captures).
    let mut config = ReconstructionConfig::new(0.0).sequential();
    config.progress = progress;
    if let Some(v) = overrides.max_iterations {
        config.max_iterations = v;
    }
    if let Some(v) = overrides.min_registration_inliers {
        config.min_registration_inliers = v;
    }
    if let Some(v) = overrides.motion_prior_sigma {
        config.motion_prior_sigma = v;
    }
    if let Some(v) = overrides.up_prior_sigma {
        config.up_prior_sigma = v;
    }
    if let Some(v) = overrides.max_reprojection_error {
        config.max_reprojection_error = v;
    }
    if let Some(s) = overrides.sprt {
        config.sprt = Some(s);
    }
    if let Some(v) = overrides.refine_intrinsics {
        config.refine_intrinsics = v;
    }
    let mut recon = reconstruct(&cameras, &[], tracks, &config, None)?;
    if tag_size_m > 0.0 {
        if let Some((tag_id, scale)) =
            robust_tag_scale(&recon, tags_for_scale, &cameras, tag_size_m)
        {
            for pose in recon.views.iter_mut().flatten() {
                pose.translation *= scale;
            }
            for point in &mut recon.points {
                point.position *= scale;
            }
            recon.scale = ScaleSource::Tag {
                id: tag_id,
                size_m: tag_size_m,
            };
        }
    }
    Ok(recon)
}

/// Estimate a metric scale factor from a planar tag, robustly.
///
/// kornia's own `tag_scale` triangulates the four corners from the single
/// widest-baseline registered pair and averages the edges — on an orbit that
/// pair can be nearly fronto-parallel and collapse the tag to a point, giving a
/// wildly wrong scale. Here each corner is triangulated by **N-view DLT over
/// every registered view that saw the tag**, and the scale is the ratio of the
/// known side length to the **median** of the four recovered edges.
fn robust_tag_scale(
    recon: &Reconstruction,
    tags: &[TagObservation],
    cameras: &[PinholeCamera],
    tag_size_m: f64,
) -> Option<(u16, f64)> {
    let tag = tags.first()?;
    let seers: Vec<usize> = tag
        .per_camera
        .iter()
        .filter(|(c, _)| recon.views[*c].is_some())
        .map(|(c, _)| *c)
        .collect();
    if seers.len() < 2 {
        return None;
    }

    let mut world = Vec::with_capacity(4);
    for k in 0..4 {
        let mut pixels = Vec::with_capacity(seers.len());
        let mut projections = Vec::with_capacity(seers.len());
        for &c in &seers {
            let corners = tag.per_camera.iter().find(|(cc, _)| *cc == c)?.1;
            pixels.push(corners[k]);
            let world_to_cam = recon.views[c].as_ref()?.inverse();
            projections.push(projection_3x4(&cameras[c], &world_to_cam));
        }
        world.push(triangulate_n_view_linear(&pixels, &projections)?);
    }

    let mut edges = [
        (world[0] - world[1]).length(),
        (world[1] - world[2]).length(),
        (world[2] - world[3]).length(),
        (world[3] - world[0]).length(),
    ];
    let side = median(&mut edges);
    if side <= 1e-9 {
        return None;
    }
    Some((tag.tag_id, tag_size_m / side))
}

/// Flatten the `3x4` camera matrix `K [R | t]` in row-major order for
/// [`triangulate_n_view_linear`], where `(R, t)` is a world-to-camera pose.
fn projection_3x4(cam: &PinholeCamera, world_to_cam: &Pose3d) -> [f64; 12] {
    let r = world_to_cam.rotation;
    let t = world_to_cam.translation;
    let (fx, fy, cx, cy) = (cam.fx, cam.fy, cam.cx, cam.cy);
    let (r0, r1, r2) = (r.row(0), r.row(1), r.row(2));
    [
        fx * r0.x + cx * r2.x,
        fx * r0.y + cx * r2.y,
        fx * r0.z + cx * r2.z,
        fx * t.x + cx * t.z,
        fy * r1.x + cy * r2.x,
        fy * r1.y + cy * r2.y,
        fy * r1.z + cy * r2.z,
        fy * t.y + cy * t.z,
        r2.x,
        r2.y,
        r2.z,
        t.z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
    use kornia_calib::ScaleSource;

    #[test]
    fn make_camera_constructs_pinhole_camera() {
        let cam = make_camera(500.0, 520.0, 320.0, 240.0);
        assert_eq!(cam.fx, 500.0);
        assert_eq!(cam.fy, 520.0);
        assert_eq!(cam.cx, 320.0);
        assert_eq!(cam.cy, 240.0);
        assert_eq!(cam.k1, 0.0);
        assert_eq!(cam.k2, 0.0);
        assert_eq!(cam.p1, 0.0);
        assert_eq!(cam.p2, 0.0);
    }

    #[test]
    fn run_sfm_recovers_points_and_poses_from_synthetic_scene() {
        use kornia_algebra::linalg::rigid::umeyama_f64;
        use kornia_algebra::Vec3F64;

        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            &[],
            0.0,
            None,
            ReconstructionOverrides::default(),
        )
        .expect("synthetic scene must reconstruct");

        assert!(
            !recon.points.is_empty(),
            "reconstruction should recover 3D points"
        );

        // The bootstrap pair alone satisfies ">= 2"; require all three cameras,
        // which is what exercising PnP registration actually means.
        let recovered: Vec<Vec3F64> = recon
            .views
            .iter()
            .map(|v| v.as_ref().map(|p| p.translation))
            .collect::<Option<Vec<_>>>()
            .expect("all three cameras must register");

        // The map is up to scale and world frame; align the recovered camera
        // centres to ground truth with a similarity before comparing.
        let gt: Vec<Vec3F64> = test_util::synthetic_poses()
            .iter()
            .map(|p| p.inverse().translation)
            .collect();
        let (r, t, s) = umeyama_f64(&recovered, &gt, None, true).expect("similarity alignment");
        let max_err = recovered
            .iter()
            .zip(&gt)
            .map(|(p, q)| (r * *p * s + t - *q).length())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-6,
            "similarity-aligned camera centres must match ground truth (max err {max_err:.3e})"
        );
    }

    #[test]
    fn run_sfm_anchors_metric_scale_from_a_tag() {
        use kornia_algebra::Vec3F64;
        use kornia_calib::TagObservation;

        let tracks = test_util::synthetic_tracks();
        let cam = || make_camera(test_util::FX, test_util::FY, test_util::CX, test_util::CY);
        let cams = [cam(), cam(), cam()];
        let gt = test_util::synthetic_poses();
        let tag_size = 0.2;

        // A planar 0.2 m tag in front of the rig, in aruco winding (TL,TR,BR,BL).
        let corners = [
            Vec3F64::new(-0.1, 0.1, 1.5),
            Vec3F64::new(0.1, 0.1, 1.5),
            Vec3F64::new(0.1, -0.1, 1.5),
            Vec3F64::new(-0.1, -0.1, 1.5),
        ];
        let per_camera: Vec<(usize, [kornia_algebra::Vec2F64; 4])> = (0..test_util::N_FRAMES)
            .map(|c| {
                let obs = [
                    test_util::project(corners[0], &gt[c], &cams[c]),
                    test_util::project(corners[1], &gt[c], &cams[c]),
                    test_util::project(corners[2], &gt[c], &cams[c]),
                    test_util::project(corners[3], &gt[c], &cams[c]),
                ];
                (c, obs)
            })
            .collect();
        let tag = TagObservation {
            tag_id: 42,
            per_camera,
        };

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            &[tag],
            tag_size,
            None,
            ReconstructionOverrides::default(),
        )
        .expect("synthetic scene must reconstruct");

        assert_eq!(
            recon.scale,
            ScaleSource::Tag {
                id: 42,
                size_m: tag_size
            }
        );

        // Metric scale is now fixed by the tag, so the recovered camera baseline
        // must equal the ground-truth baseline in metres (no similarity freedom).
        let centres: Vec<Vec3F64> = recon
            .views
            .iter()
            .map(|v| v.as_ref().expect("camera must register").translation)
            .collect();
        let gt_centres: Vec<Vec3F64> = gt.iter().map(|p| p.inverse().translation).collect();
        let recovered = (centres[1] - centres[0]).length();
        let truth = (gt_centres[1] - gt_centres[0]).length();
        assert!(
            (recovered - truth).abs() < 1e-3,
            "metric baseline {recovered:.6} must match ground truth {truth:.6}"
        );
    }

    #[test]
    fn run_sfm_is_up_to_scale_without_tag() {
        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            &[],
            0.0,
            None,
            ReconstructionOverrides::default(),
        )
        .expect("synthetic scene must reconstruct");
        assert_eq!(recon.scale, ScaleSource::UpToScale);
    }
}
