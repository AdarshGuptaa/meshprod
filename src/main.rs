//! End-to-end monocular reconstruction: video → sparse SfM → dense depth →
//! TSDF volume → simplified, coloured triangle mesh.
//!
//! Pipeline: read the MP4 into RGB + grayscale frames, extract features
//! (ORB or SIFT), match frames in a sliding window, chain the matches into
//! tracks via `kornia_calib::build_tracks`, reconstruct the scene with
//! `kornia_calib::reconstruct`, estimate dense depth by plane sweeping, fuse
//! the depth maps into a TSDF, extract a mesh with marching tetrahedra,
//! simplify it, and export a binary PLY with normals and vertex colours.
//!
//! Every pipeline stage logs its start, progress, and completion time to
//! stderr so slow stages can be identified.

mod dense;
mod dense_sgm;
mod features;
mod matching;
mod mesh;
mod mesh_ply;
mod ply_writer;
mod reconstruction;
mod simplify;
mod tsdf;
mod video;

#[cfg(test)]
mod test_util;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use argh::FromArgs;
use kornia_algebra::Vec3F64;
use kornia_calib::build_tracks;
use kornia_image::Image;

#[derive(FromArgs)]
/// Structure-from-Motion from a video file to a simplified triangle mesh.
struct Args {
    /// path to the input video file
    #[argh(positional)]
    video: PathBuf,

    /// path to the output mesh PLY (sparse/dense clouds are written beside it
    /// as <stem>_sparse.ply and <stem>_cloud.ply)
    #[argh(positional)]
    output: PathBuf,

    /// feature detector: "orb" or "sift"
    #[argh(option, default = "features::DetectorKind::Orb")]
    detector: features::DetectorKind,

    /// focal length X in pixels
    #[argh(option)]
    fx: f64,

    /// focal length Y in pixels
    #[argh(option)]
    fy: f64,

    /// principal point X in pixels
    #[argh(option)]
    cx: f64,

    /// principal point Y in pixels
    #[argh(option)]
    cy: f64,

    /// max features per frame
    #[argh(option, default = "2000")]
    n_features: usize,

    /// match each frame against this many following frames
    #[argh(option, default = "5")]
    match_window: usize,

    /// lowe's ratio test threshold (lower = stricter)
    #[argh(option, default = "0.8")]
    ratio: f32,

    /// process every Nth frame (1 = all frames)
    #[argh(option, default = "1")]
    frame_step: usize,

    /// worker threads for parallel feature extraction/matching (0 = auto)
    #[argh(option, default = "0")]
    threads: usize,

    /// max bundle adjustment iterations (default: 100)
    #[argh(option, default = "100")]
    max_ba_iterations: usize,

    /// min PnP inliers to register a view (default: 30)
    #[argh(option, default = "30")]
    min_registration_inliers: usize,

    /// motion prior sigma (0.0 = disabled)
    #[argh(option, default = "0.0")]
    motion_prior_sigma: f64,

    /// up prior sigma (0.0 = disabled)
    #[argh(option, default = "0.0")]
    up_prior_sigma: f64,

    /// max reprojection error in normalized units (default: 0.01)
    #[argh(option, default = "0.01")]
    max_reprojection_error: f64,

    /// verify matches with epipolar RANSAC (rejects false matches)
    #[argh(switch)]
    geo_verify: bool,

    /// epipolar RANSAC inlier threshold in pixels (default: 3.0)
    #[argh(option, default = "3.0")]
    geo_threshold: f64,

    /// min inliers for a pair's fundamental matrix to be trusted (default: 15;
    /// values below 15 warn — a seven-point model validates its own minimal sample)
    #[argh(option, default = "15")]
    geo_min_inliers: usize,

    /// use CUDA for SIFT extraction (requires an NVIDIA GPU)
    #[argh(switch)]
    cuda: bool,

    /// enable Wald's SPRT for PnP registration (rejects bad poses early)
    #[argh(switch)]
    sprt: bool,

    /// SPRT expected inlier ratio (default: 0.5)
    #[argh(option, default = "0.5")]
    sprt_epsilon: f64,

    /// SPRT Type-I error delta, probability of rejecting a good pose (default: 0.05)
    #[argh(option, default = "0.05")]
    sprt_delta: f64,

    /// refine focal length + radial/tangential distortion against the
    /// reconstruction (helps when intrinsics are guessed)
    #[argh(switch)]
    refine_intrinsics: bool,

    /// also match frame i against i+K, i+2K, ... (wide baseline, longer
    /// tracks, more registered views; 0 = off)
    #[argh(option, default = "0")]
    wide_baseline: usize,

    /// stop after the sparse reconstruction (skip dense depth and meshing)
    #[argh(switch)]
    sparse_only: bool,

    /// remove the dominant plane (table, floor) from the dense cloud before fusion
    #[argh(switch)]
    remove_plane: bool,

    /// dense depth-map resolution relative to the source (0.5 = half)
    #[argh(option, default = "0.5")]
    dense_scale: f64,

    /// inverse-depth planes tested per pixel in the plane sweep
    #[argh(option, default = "96")]
    dense_planes: usize,

    /// odd NCC window side for the plane sweep
    #[argh(option, default = "5")]
    dense_window: usize,

    /// neighbouring views each view compares against
    #[argh(option, default = "3")]
    dense_neighbors: usize,

    /// minimum NCC for a depth pixel to be kept
    #[argh(option, default = "0.4")]
    ncc_min: f32,

    /// neighbours that must agree on a depth (geometric consistency; 1 = loose)
    #[argh(option, default = "1")]
    lr_required: usize,

    /// volumetric diffusion iterations that close holes in the TSDF (0 = off;
    /// off by default because diffusion invents surface where there is no data)
    #[argh(option, default = "0")]
    hole_fill_iterations: usize,

    /// box-filter iterations on the TSDF before meshing (0 = off)
    #[argh(option, default = "2")]
    volume_smoothing: usize,

    /// taubin smoothing iterations after meshing (0 = off)
    #[argh(option, default = "8")]
    smooth_iterations: usize,

    /// TSDF grid resolution along the longest scene axis
    #[argh(option, default = "256")]
    voxel_resolution: usize,

    /// TSDF truncation distance in voxels
    #[argh(option, default = "3.0")]
    trunc_voxels: f64,

    /// simplify the mesh to this many faces (0 = keep the full mesh)
    #[argh(option, default = "20000")]
    target_faces: usize,

    /// drop connected surface patches smaller than this many faces (0 = keep all)
    #[argh(option, default = "200")]
    min_component_faces: usize,

    /// photometric cost: "ncc", "census" or "sgm" (default ncc)
    #[argh(option)]
    cost: Option<dense::CostMethod>,

    /// cardinal SGM paths (1..=4)
    #[argh(option, default = "4")]
    sgm_paths: usize,

    /// derive dense/mesh parameters from the measured frame texture
    #[argh(switch)]
    auto_config: bool,

    /// keep only the largest connected component of the dense cloud
    #[argh(switch)]
    auto_crop: bool,

    /// translate the outputs so their centre is the world origin
    #[argh(switch)]
    recenter: bool,
}

/// Validate CLI values that must be positive.
///
/// `--frame-step 0` otherwise reaches `video::read_frames`'s `assert!` and
/// panics (a backtrace hint, not a CLI error); `--match-window 0` produces zero
/// frame pairs and only fails later with an opaque bootstrap error. Reject both
/// up front.
fn validate_positive_args(frame_step: usize, match_window: usize) -> Result<(), String> {
    if frame_step == 0 {
        return Err("--frame-step must be >= 1".into());
    }
    if match_window == 0 {
        return Err("--match-window must be >= 1".into());
    }
    Ok(())
}

/// Build and validate the SPRT config from CLI flags.
///
/// Returns `None` when SPRT is disabled. An invalid config (epsilon/delta
/// outside `(0, 1)`) makes PnP silently skip SPRT, so reject it as a CLI error
/// instead of letting `--sprt` appear enabled while it does nothing.
fn build_sprt_config(
    enabled: bool,
    epsilon: f64,
    delta: f64,
) -> Result<Option<kornia_3d::ransac::SPRTConfig>, String> {
    if !enabled {
        return Ok(None);
    }
    let cfg = kornia_3d::ransac::SPRTConfig {
        epsilon,
        delta,
        ..Default::default()
    };
    if !cfg.is_valid() {
        return Err(format!(
            "invalid SPRT config (epsilon={epsilon}, delta={delta}): both must be in (0, 1)"
        ));
    }
    Ok(Some(cfg))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    // Configure the rayon thread pool (0 = auto-detect).
    features::configure_thread_pool(args.threads)?;

    // Reject zero values up front rather than panicking deep in the pipeline.
    validate_positive_args(args.frame_step, args.match_window)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    // Validate SPRT before the pipeline runs; an invalid config would otherwise
    // silently disable it.
    let sprt = build_sprt_config(args.sprt, args.sprt_epsilon, args.sprt_delta)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    // 1. Decode frames.
    eprintln!("[1/9] reading video: {}", args.video.display());
    let t = Instant::now();
    let (rgb_frames, gray_frames) = video::read_frames(&args.video, args.frame_step)?;
    let n_frames = gray_frames.len();
    let (frame_w, frame_h) = rgb_frames
        .first()
        .map(|f| (f.width(), f.height()))
        .unwrap_or((0, 0));
    eprintln!(
        "[1/9] decoded {} frames ({}x{}) in {:.1}s",
        n_frames,
        frame_w,
        frame_h,
        t.elapsed().as_secs_f64()
    );
    if n_frames < 2 {
        return Err("need at least two frames to reconstruct".into());
    }
    if n_frames > 200 {
        eprintln!(
            "  warning: {n_frames} frames may make reconstruction slow; consider a larger --frame-step"
        );
    }

    // 2. Extract features per frame (parallel via rayon; sequential for CUDA).
    // --cuda only affects SIFT; ignore it for the CPU-only ORB detector.
    let use_cuda = args.cuda && args.detector == features::DetectorKind::Sift;
    if args.cuda && !use_cuda {
        eprintln!(
            "  warning: --cuda only applies to SIFT; ignoring it for {:?}",
            args.detector
        );
    }
    eprintln!("[2/9] extracting features with {:?}", args.detector);
    if use_cuda {
        eprintln!("[2/9] CUDA SIFT extraction (device 0)");
    }
    let t = Instant::now();
    let extractor = features::make_extractor(args.detector, args.n_features, use_cuda)?;
    // CUDA shares one device stream, so extraction must be sequential.
    let all_features = if use_cuda {
        gray_frames
            .iter()
            .map(|frame| extractor.extract(frame))
            .collect::<Result<_, _>>()?
    } else {
        features::extract_features_parallel(&gray_frames, extractor.as_ref())
            .map_err(|e| -> Box<dyn Error> { e.into() })?
    };
    let total_keypoints: usize = all_features.iter().map(|f| f.n_keypoints()).sum();
    eprintln!(
        "[2/9] extracted {total_keypoints} keypoints across {} frames in {:.1}s",
        all_features.len(),
        t.elapsed().as_secs_f64()
    );
    // Gray frames stay resident for the dense stage; RGB frames for sampling
    // colours and dense texturing.

    // Device SIFT matcher (only when descriptors are CUDA-resident), sized to
    // the largest descriptor set so no pair can exceed its capacity.
    let cuda_matcher = if use_cuda {
        Some(features::CudaSiftMatcher::new(
            features::max_sift_descriptors(&all_features),
        )?)
    } else {
        None
    };

    // 3. Match frames in a sliding window (parallel via rayon).
    eprintln!("[3/9] matching frames (window={})", args.match_window);
    let t = Instant::now();
    let mut edges = matching::match_pairs_parallel(
        &all_features,
        args.match_window,
        args.ratio,
        cuda_matcher.as_ref(),
    )?;
    eprintln!(
        "[3/9] found {} matched correspondences in {:.1}s",
        edges.len(),
        t.elapsed().as_secs_f64()
    );

    // 3.25. Optional: wide-baseline matching (long jumps beyond the window).
    if args.wide_baseline > 0 {
        eprintln!(
            "[3/9] wide-baseline matching (stride={})",
            args.wide_baseline
        );
        let t = Instant::now();
        let added = matching::append_wide_baseline_edges(
            &all_features,
            args.match_window,
            args.wide_baseline,
            args.ratio,
            cuda_matcher.as_ref(),
            &mut edges,
        )?;
        eprintln!(
            "[3/9] added {added} wide-baseline matches in {:.1}s",
            t.elapsed().as_secs_f64()
        );
    }

    // 3.5. Optional: geometric verification (epipolar RANSAC) to reject false matches.
    let edges = if args.geo_verify {
        eprintln!(
            "[3.5/9] geometric verification (threshold={} px, min_inliers={})",
            args.geo_threshold, args.geo_min_inliers
        );
        if let Some(w) = matching::min_inliers_warning(args.geo_min_inliers) {
            eprintln!("  {w}");
        }
        let t = Instant::now();
        let before = edges.len();
        let edges = matching::verify_matches_geometrically(
            &edges,
            args.geo_threshold,
            args.geo_min_inliers,
        );
        eprintln!(
            "[3.5/9] filtered {before} -> {} matches in {:.1}s",
            edges.len(),
            t.elapsed().as_secs_f64()
        );
        edges
    } else {
        edges
    };

    // 4. Chain matches into multi-view tracks.
    eprintln!("[4/9] building tracks");
    let t = Instant::now();
    let tracks = build_tracks(&edges);
    eprintln!(
        "[4/9] built {} tracks in {:.1}s",
        tracks.len(),
        t.elapsed().as_secs_f64()
    );

    // Sample every raw observation's colour now; RGB frames stay resident for
    // the dense stage's texturing.
    let observation_colors = ply_writer::sample_observation_colors(&tracks, &rgb_frames);

    // 5. Reconstruct the scene.
    eprintln!("[5/9] reconstructing scene");
    let t = Instant::now();
    let progress_start = Instant::now();
    let progress: Arc<dyn Fn(usize, usize) + Send + Sync> = Arc::new(move |registered, n_cams| {
        eprintln!(
            "  reconstruct: {registered}/{n_cams} views registered ({:.1}s)",
            progress_start.elapsed().as_secs_f64()
        );
    });
    let overrides = reconstruction::ReconstructionOverrides {
        max_iterations: Some(args.max_ba_iterations),
        min_registration_inliers: Some(args.min_registration_inliers),
        motion_prior_sigma: Some(args.motion_prior_sigma),
        up_prior_sigma: Some(args.up_prior_sigma),
        max_reprojection_error: Some(args.max_reprojection_error),
        sprt,
        refine_intrinsics: args.refine_intrinsics.then_some(true),
    };
    let reconstruction = reconstruction::run_sfm(
        &tracks,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        n_frames,
        Some(progress),
        overrides,
    )?;
    let registered = reconstruction.views.iter().filter(|v| v.is_some()).count();
    eprintln!(
        "[5/9] reconstructed {} points across {} registered views (scale: {:?}, rmse {:.3} px) in {:.1}s",
        reconstruction.points.len(),
        registered,
        reconstruction.scale,
        reconstruction.reproj_rmse_px,
        t.elapsed().as_secs_f64(),
    );
    let registered_frames: Vec<usize> = reconstruction
        .views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|_| i * args.frame_step))
        .collect();
    eprintln!("  registered source frames: {registered_frames:?}");
    if let Some((gamma, k1, k2, p1, p2)) = reconstruction.camera_correction {
        eprintln!(
            "[5/9] refined intrinsics: gamma {gamma:.4} (fx ~ {:.1} px), k1 {k1:.4}, k2 {k2:.4}, \
             p1 {p1:.4}, p2 {p2:.4}",
            args.fx / gamma
        );
    }

    // 6. Export the sparse cloud, then continue to the dense stages.
    let vertices = ply_writer::build_vertices(&reconstruction, &observation_colors);
    let sparse_path = if args.sparse_only {
        args.output.clone()
    } else {
        sibling_path(&args.output, "_sparse")
    };
    let t = Instant::now();
    ply_writer::write_ply(&sparse_path, &vertices)?;
    eprintln!(
        "[6/9] wrote {} sparse vertices to {} in {:.1}s",
        vertices.len(),
        sparse_path.display(),
        t.elapsed().as_secs_f64(),
    );
    if args.sparse_only {
        return Ok(());
    }

    // 6. Dense depth maps by plane sweeping (one per registered view).
    eprintln!(
        "[6/9] estimating dense depth (scale {:.2}, {} planes, {} neighbours)",
        args.dense_scale, args.dense_planes, args.dense_neighbors
    );
    let t = Instant::now();
    let cameras = dense_cameras(
        &reconstruction,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        n_frames,
    );
    let mut dense_cfg = dense::DenseConfig {
        scale: args.dense_scale,
        planes: args.dense_planes,
        window: args.dense_window,
        ncc_min: args.ncc_min,
        neighbors: args.dense_neighbors,
        lr_tolerance: 0.05,
        lr_required: args.lr_required,
        method: args.cost.unwrap_or(dense::CostMethod::Ncc),
        sgm_paths: args.sgm_paths,
        census_max_cost: 0.35,
    };
    let mut target_faces = args.target_faces;
    let mut min_component_faces = args.min_component_faces;
    if args.auto_config {
        let score = texture_score(&gray_frames);
        let (low, medium) = (score < 0.012, score < 0.035);
        // Census WTA is ambiguous on perfectly flat regions (identical codes
        // give cost zero), which reconstructs the tabletop as if it were the
        // object. SGM resolves that by propagation, so it is the default for
        // anything below high texture. Measured on the keyboard clip: censu
        // WTA -> 30 patches, SGM -> 2 patches.
        let auto_method = if low || medium {
            dense::CostMethod::Sgm
        } else {
            dense::CostMethod::Ncc
        };
        let method = args.cost.unwrap_or(auto_method);
        dense_cfg.method = method;
        dense_cfg.window = if low || medium { 7 } else { 5 };
        dense_cfg.ncc_min = if low {
            0.30
        } else if medium {
            0.35
        } else {
            0.40
        };
        dense_cfg.planes = if low { 128 } else { 96 };
        eprintln!(
            "[auto] texture score {score:.4} ({}) -> cost {method:?}, window {}, ncc-min {:.2}, planes {}",
            if low {
                "low"
            } else if medium {
                "medium"
            } else {
                "high"
            },
            dense_cfg.window,
            dense_cfg.ncc_min,
            dense_cfg.planes
        );
    }
    let maps = dense::estimate_depth_maps(&reconstruction, &cameras, &gray_frames, &dense_cfg)?;
    let n_maps = maps.iter().flatten().count();
    let n_valid: usize = maps
        .iter()
        .flatten()
        .map(|m| m.inv_depth.iter().filter(|v| **v > 0.0).count())
        .sum();
    eprintln!(
        "[6/9] {n_maps} depth maps with {n_valid} valid pixels in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    if n_maps == 0 {
        return Err("no depth maps were produced from the registered views".into());
    }
    let (checked, median_err, p90_err) =
        dense::sparse_depth_check(&reconstruction, &maps, &cameras);
    if checked > 0 {
        eprintln!(
            "  depth vs sparse: {checked} samples, median {:.1}% error, p90 {:.1}%",
            median_err * 100.0,
            p90_err * 100.0
        );
    }

    let mut cloud = dense::dense_cloud(&maps, &cameras, &reconstruction.views, &rgb_frames);
    if args.remove_plane && cloud.points.len() >= 16 {
        let mut lo = cloud.points[0];
        let mut hi = cloud.points[0];
        for p in &cloud.points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let diagonal =
            ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let t = Instant::now();
        let removed = dense::remove_dominant_plane(&mut cloud, 0.01 * diagonal.max(1e-9));
        eprintln!(
            "  removed dominant plane: {removed} points in {:.1}s",
            t.elapsed().as_secs_f64()
        );
    }
    if args.auto_crop && cloud.points.len() >= 16 {
        let (mut lo, mut hi) = (cloud.points[0], cloud.points[0]);
        for p in &cloud.points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let diagonal =
            ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let t = Instant::now();
        let removed = dense::auto_crop_largest(&mut cloud, (diagonal / 48.0).max(1e-9));
        eprintln!(
            "  auto-crop: removed {removed} background points, {} remain in {:.1}s",
            cloud.points.len(),
            t.elapsed().as_secs_f64()
        );
    }
    if args.auto_config && !cloud.points.is_empty() {
        target_faces = (cloud.points.len() / 20).clamp(10_000, 150_000);
        min_component_faces = (target_faces / 500).max(200);
        eprintln!(
            "[auto] {} dense points -> target-faces {target_faces}, min-component-faces {min_component_faces}",
            cloud.points.len()
        );
    }
    eprintln!("  dense cloud: {} points", cloud.points.len());

    // 7. Fuse the depth maps into a signed distance field.
    eprintln!(
        "[7/9] fusing TSDF (resolution {}, truncation {:.1} voxels)",
        args.voxel_resolution, args.trunc_voxels
    );
    let t = Instant::now();
    let volume = tsdf::fuse(
        &maps,
        &cameras,
        &reconstruction.views,
        &cloud.points,
        &tsdf::TsdfConfig {
            resolution: args.voxel_resolution,
            trunc_voxels: args.trunc_voxels,
            depth_noise_relative: (checked > 0).then_some(median_err),
            ncc_min: args.ncc_min,
        },
    )?;
    eprintln!(
        "[7/9] volume {}x{}x{} at {:.4} units/voxel in {:.1}s",
        volume.dims[0],
        volume.dims[1],
        volume.dims[2],
        volume.voxel_size,
        t.elapsed().as_secs_f64()
    );
    let observed = volume.weight.iter().filter(|w| **w > 0.0).count();
    let inside = volume
        .weight
        .iter()
        .zip(&volume.sdf)
        .filter(|(w, s)| **w > 0.0 && **s < 0.0)
        .count();
    eprintln!(
        "  observed voxels: {observed}, solid (sdf<0): {inside}, truncation {:.2} units",
        volume.trunc
    );
    let mut volume = volume;
    if args.hole_fill_iterations > 0 {
        let t = Instant::now();
        let filled = tsdf::fill_holes(&mut volume, args.hole_fill_iterations);
        let observed_after = volume.weight.iter().filter(|w| **w > 0.0).count();
        eprintln!(
            "  diffusion filled {filled} voxels ({} iterations) -> {observed_after} observed in {:.1}s",
            args.hole_fill_iterations,
            t.elapsed().as_secs_f64()
        );
    }
    if args.volume_smoothing > 0 {
        let t = Instant::now();
        tsdf::smooth_volume(&mut volume, args.volume_smoothing);
        eprintln!(
            "  field smoothing: {} iterations in {:.1}s",
            args.volume_smoothing,
            t.elapsed().as_secs_f64()
        );
    }

    // 8. Extract the surface, simplify it, and compute vertex attributes.
    eprintln!("[8/9] extracting surface and simplifying");
    let t = Instant::now();
    let mut mesh = mesh::marching_tetrahedra(&volume);
    eprintln!(
        "  raw mesh: {} vertices, {} faces, {} patches",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.connected_components()
    );
    if mesh.face_count() == 0 {
        return Err(
            "the TSDF produced an empty mesh; try a lower --ncc-min or a higher --voxel-resolution"
                .into(),
        );
    }
    if min_component_faces > 0 {
        mesh = mesh.remove_small_components(min_component_faces);
        eprintln!(
            "  removed small patches: {} faces in {} patches remain",
            mesh.face_count(),
            mesh.connected_components()
        );
    }
    if args.smooth_iterations > 0 {
        let t = Instant::now();
        mesh = mesh::taubin_smooth(&mesh, args.smooth_iterations, 0.5, -0.53);
        eprintln!(
            "  Taubin smoothing: {} iterations in {:.1}s (closed={}, euler={})",
            args.smooth_iterations,
            t.elapsed().as_secs_f64(),
            mesh.is_closed(),
            mesh.euler_characteristic()
        );
    }
    if target_faces > 0 && mesh.face_count() > target_faces {
        mesh = simplify::simplify(&mesh, target_faces);
        eprintln!("  simplified to {} faces", mesh.face_count());
    }
    let normals = mesh::vertex_normals(&mesh, &volume);
    let colors = mesh::vertex_colors(
        &mesh,
        &normals,
        &cameras,
        &reconstruction.views,
        &rgb_frames,
    );
    let recenter = args.recenter || args.auto_crop;
    let shift = if recenter && !mesh.positions.is_empty() {
        let n = mesh.positions.len() as f64;
        let centre = mesh.positions.iter().fold(Vec3F64::ZERO, |acc, p| acc + *p) / n;
        for p in &mut mesh.positions {
            *p -= centre;
        }
        eprintln!(
            "  recentred to origin (shift {:.2}, {:.2}, {:.2})",
            centre.x, centre.y, centre.z
        );
        centre
    } else {
        Vec3F64::ZERO
    };
    mesh_ply::write_mesh_ply(&args.output, &mesh, &normals, &colors)?;
    eprintln!(
        "[9/9] wrote mesh ({} vertices, {} faces, patches={}, euler={}, closed={}) to {} in {:.1}s",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.connected_components(),
        mesh.euler_characteristic(),
        mesh.is_closed(),
        args.output.display(),
        t.elapsed().as_secs_f64()
    );

    // 9. A voxel-downsampled dense cloud for inspection.
    let down = dense::voxel_downsample_cloud(&cloud, volume.voxel_size * 2.0)?;
    let cloud_vertices: Vec<ply_writer::Vertex> = down
        .points
        .iter()
        .zip(&down.colors)
        .map(|(p, c)| ply_writer::Vertex {
            position: [p[0] - shift.x, p[1] - shift.y, p[2] - shift.z],
            color: *c,
            normal: volume.gradient(Vec3F64::new(p[0], p[1], p[2])).to_array(),
        })
        .collect();
    let cloud_path = sibling_path(&args.output, "_cloud");
    ply_writer::write_ply(&cloud_path, &cloud_vertices)?;
    eprintln!(
        "[9/9] wrote {} dense points to {}",
        cloud_vertices.len(),
        cloud_path.display()
    );

    // Oriented cloud for external reconstruction backends (e.g. Open3D's
    // screened Poisson): finer sampling, normals from the TSDF gradient.
    let oriented = dense::voxel_downsample_cloud(&cloud, volume.voxel_size)?;
    let oriented_vertices: Vec<ply_writer::Vertex> = oriented
        .points
        .iter()
        .zip(&oriented.colors)
        .map(|(p, c)| ply_writer::Vertex {
            position: [p[0] - shift.x, p[1] - shift.y, p[2] - shift.z],
            color: *c,
            normal: volume.gradient(Vec3F64::new(p[0], p[1], p[2])).to_array(),
        })
        .collect();
    let oriented_path = sibling_path(&args.output, "_oriented");
    ply_writer::write_ply(&oriented_path, &oriented_vertices)?;
    eprintln!(
        "[9/9] wrote {} oriented points to {} (Poisson input)",
        oriented_vertices.len(),
        oriented_path.display()
    );

    Ok(())
}

/// Sibling output path: `<stem><suffix><ext>` next to `output`.
fn sibling_path(output: &Path, suffix: &str) -> PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let ext = output
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    output.with_file_name(format!("{stem}{suffix}{ext}"))
}

/// Mean image gradient magnitude, normalized to `[0, 1]` — a texture proxy.
///
/// Textured scenes land above ~0.04; smooth objects (white plastic, painted
/// walls) and flat backgrounds stay below ~0.015, which is where NCC stops
/// working and census/SGM pay off.
fn texture_score(frames: &[Image<u8, 1>]) -> f64 {
    let stride = (frames.len() / 8).max(1);
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for frame in frames.iter().step_by(stride).take(8) {
        let (w, h) = (frame.width(), frame.height());
        if w < 3 || h < 3 {
            continue;
        }
        let s = frame.as_slice();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = (s[y * w + x + 1] as f64 - s[y * w + x - 1] as f64).abs();
                let gy = (s[(y + 1) * w + x] as f64 - s[(y - 1) * w + x] as f64).abs();
                sum += gx + gy;
                count += 1.0;
            }
        }
    }
    if count == 0.0 {
        0.0
    } else {
        (sum / count) / 255.0
    }
}

/// Cameras for the dense stage, corrected by the intrinsics bundle adjustment
/// fitted in the sparse solve.
///
/// `gamma` scales normalized coordinates, so it inverts into focal length:
/// `fx_true = fx_assumed / gamma`. Lens distortion is ignored by the dense
/// projector (the sweep is a pure pinhole homography), so a large fit is
/// reported rather than silently misused.
fn dense_cameras(
    reconstruction: &kornia_calib::Reconstruction,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    n_frames: usize,
) -> Vec<kornia_3d::camera::PinholeCamera> {
    let mut cameras = vec![reconstruction::make_camera(fx, fy, cx, cy); n_frames];
    if let Some((gamma, k1, k2, p1, p2)) = reconstruction.camera_correction {
        if gamma > 0.0 && gamma.is_finite() {
            for cam in &mut cameras {
                cam.fx = fx / gamma;
                cam.fy = fy / gamma;
            }
        }
        if k1.abs() > 1e-3 || k2.abs() > 1e-3 || p1.abs() > 1e-3 || p2.abs() > 1e-3 {
            eprintln!(
                "  warning: dense stage ignores fitted distortion (k1={k1:.4}, k2={k2:.4}, \
                 p1={p1:.4}, p2={p2:.4}); expect some depth bias toward the frame edges"
            );
        }
    }
    cameras
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_positive_args_rejects_zero() {
        assert!(validate_positive_args(1, 5).is_ok());
        assert!(validate_positive_args(10, 1).is_ok());
        assert!(validate_positive_args(0, 5).is_err(), "frame-step 0");
        assert!(validate_positive_args(1, 0).is_err(), "match-window 0");
        assert!(validate_positive_args(0, 0).is_err());
    }

    #[test]
    fn texture_score_separates_flat_from_textured() {
        use kornia_image::ImageSize;
        let flat = vec![Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 32,
                height: 32,
            },
            100,
        )
        .unwrap()];
        let mut checker = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 32,
                height: 32,
            },
            0,
        )
        .unwrap();
        {
            let s = checker.as_slice_mut();
            for y in 0..32 {
                for x in 0..32 {
                    s[y * 32 + x] = if (x / 4 + y / 4) % 2 == 0 { 0 } else { 255 };
                }
            }
        }
        assert!(texture_score(&flat) < 0.001, "flat image has no texture");
        assert!(texture_score(&[checker]) > 0.2, "checkerboard is textured");
    }

    #[test]
    fn build_sprt_config_validates_bounds() {
        // Disabled: always None, whatever the values.
        assert!(build_sprt_config(false, 1.0, 0.0).unwrap().is_none());
        // Valid bounds.
        assert!(build_sprt_config(true, 0.5, 0.05).unwrap().is_some());
        // epsilon/delta must be strictly inside (0, 1).
        assert!(build_sprt_config(true, 1.0, 0.05).is_err(), "epsilon = 1.0");
        assert!(build_sprt_config(true, 0.5, 0.0).is_err(), "delta = 0.0");
        assert!(build_sprt_config(true, 0.0, 0.05).is_err(), "epsilon = 0.0");
    }
}
