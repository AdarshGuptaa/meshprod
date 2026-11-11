//! Dense depth estimation by multi-view plane sweeping.
//!
//! For each registered view, sweep planes of constant inverse depth along the
//! viewing rays. A plane induces a homography between the reference view and
//! every neighbour; the depth whose warped neighbour agrees best (normalized
//! cross-correlation) wins per pixel. Winners are cross-checked against the
//! neighbours' own depth maps (left-right consistency) before use.

use std::error::Error;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_calib::Reconstruction;
use kornia_image::Image;
use rayon::prelude::*;

/// Tunables for the plane sweep.
pub struct DenseConfig {
    /// Depth-map resolution relative to the source frame (0.5 = half).
    pub scale: f64,
    /// Number of inverse-depth planes to test per pixel.
    pub planes: usize,
    /// Odd NCC window side, in source pixels (`5` = 5x5).
    pub window: usize,
    /// Minimum NCC of the winning plane for a pixel to be kept.
    pub ncc_min: f32,
    /// How many neighbouring views each reference view compares against.
    pub neighbors: usize,
    /// Relative tolerance of the left-right depth check.
    pub lr_tolerance: f64,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            scale: 0.5,
            planes: 96,
            window: 5,
            ncc_min: 0.4,
            neighbors: 3,
            lr_tolerance: 0.05,
        }
    }
}

/// A depth map in inverse-depth form.
///
/// `inv_depth[i] == 0.0` marks an invalid pixel. Depth is the camera-frame
/// `z` coordinate (distance along the optical axis), not ray length.
pub struct DepthMap {
    /// Index into the frame/camera arrays.
    pub view: usize,
    /// Dense grid width.
    pub width: usize,
    /// Dense grid height.
    pub height: usize,
    /// Dense grid pixels per source pixel.
    pub scale: f64,
    /// Inverse depth per pixel (`0.0` = invalid).
    pub inv_depth: Vec<f32>,
    /// Frame indices of the neighbours used to produce this map.
    pub neighbors: Vec<usize>,
}

impl DepthMap {
    /// Depth at dense pixel `(x, y)`, or `None` when invalid.
    pub fn depth_at(&self, x: usize, y: usize) -> Option<f64> {
        let rho = *self.inv_depth.get(y * self.width + x)? as f64;
        (rho > 0.0).then_some(1.0 / rho)
    }
}

/// A coloured dense point cloud (world frame).
pub struct DenseCloud {
    /// World-frame positions.
    pub points: Vec<[f64; 3]>,
    /// RGB per point.
    pub colors: Vec<[u8; 3]>,
}

/// Largest supported NCC window side (stack storage, no per-pixel allocation).
const MAX_WINDOW: usize = 7;

/// Camera intrinsics matrix.
fn k_matrix(cam: &PinholeCamera) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(cam.fx, 0.0, 0.0),
        Vec3F64::new(0.0, cam.fy, 0.0),
        Vec3F64::new(cam.cx, cam.cy, 1.0),
    )
}

/// Inverse of the intrinsics matrix.
fn k_inverse(cam: &PinholeCamera) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(1.0 / cam.fx, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0 / cam.fy, 0.0),
        Vec3F64::new(-cam.cx / cam.fx, -cam.cy / cam.fy, 1.0),
    )
}

/// Relative rotation and translation from the reference camera to a neighbour.
///
/// `views` are `T_world_cam`; both returned terms are in reference-camera
/// coordinates: `X_neighbour = r * X_ref + t`.
fn relative_rt(ref_view: &Pose3d, nbr_view: &Pose3d) -> (Mat3F64, Vec3F64) {
    let nbr_w2c = nbr_view.inverse();
    let r = nbr_w2c.rotation * ref_view.rotation;
    let t = nbr_w2c.rotation * (ref_view.translation - nbr_view.translation);
    (r, t)
}

/// Homography mapping reference pixels to a neighbour for a fronto-parallel
/// plane at inverse depth `rho` (`depth = 1 / rho`).
fn plane_homography(
    ref_cam: &PinholeCamera,
    nbr_cam: &PinholeCamera,
    r: &Mat3F64,
    t: &Vec3F64,
    rho: f64,
) -> Mat3F64 {
    // H(rho) = K_n (R + rho * t * e3^T) K_r^-1, with e3 = (0, 0, 1), for the
    // convention X_n = R * X_r + t. (The minus form belongs to X = R(X - C).)
    let third_col = r.z_axis() + *t * rho;
    let m = Mat3F64::from_cols(r.x_axis(), r.y_axis(), third_col);
    k_matrix(nbr_cam) * m * k_inverse(ref_cam)
}

/// Apply a homography to a pixel; `None` when the point maps behind infinity.
fn apply_homography(h: &Mat3F64, x: f64, y: f64) -> Option<(f64, f64)> {
    let p = *h * Vec3F64::new(x, y, 1.0);
    if p.z.abs() < 1e-12 {
        return None;
    }
    Some((p.x / p.z, p.y / p.z))
}

/// Bilinear sample of a single-channel image; `None` outside the image.
fn sample_bilinear(img: &[u8], width: usize, height: usize, x: f64, y: f64) -> Option<f32> {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let p00 = img[y0 * width + x0] as f32;
    let p10 = img[y0 * width + x1] as f32;
    let p01 = img[y1 * width + x0] as f32;
    let p11 = img[y1 * width + x1] as f32;
    let top = p00 + (p10 - p00) * fx;
    let bottom = p01 + (p11 - p01) * fx;
    Some(top + (bottom - top) * fy)
}

/// Normalized cross-correlation of two equal-length windows.
fn ncc(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (&va, &vb) in a.iter().zip(b) {
        let da = va - mean_a;
        let db = vb - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-6 {
        // A flat patch carries no depth information.
        0.0
    } else {
        cov / denom
    }
}

/// Pick the `k` nearest registered views by camera-centre distance.
fn select_neighbors(views: &[Option<Pose3d>], view: usize, k: usize) -> Vec<usize> {
    let Some(centre) = views[view].map(|p| p.translation) else {
        return Vec::new();
    };
    let mut candidates: Vec<(f64, usize)> = views
        .iter()
        .enumerate()
        .filter_map(|(j, pose)| {
            if j == view {
                return None;
            }
            let pose = pose.as_ref()?;
            let d = (pose.translation - centre).length();
            (d > 1e-9).then_some((d, j))
        })
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates.into_iter().take(k).map(|(_, j)| j).collect()
}

/// Depth search range for a view, from the sparse reconstruction points.
///
/// Takes the 2nd..98th percentile camera-frame depth and pads it, so a few
/// stray points cannot stretch the sweep over the whole scene.
fn depth_range(points: &[kornia_calib::Point], w2c: &Pose3d, centre: Vec3F64) -> (f64, f64) {
    let mut depths: Vec<f64> = points
        .iter()
        .map(|p| w2c.transform_point(&p.position).z)
        .filter(|z| *z > 1e-9)
        .collect();
    if depths.len() >= 8 {
        depths.sort_by(f64::total_cmp);
        let lo = depths[(depths.len() as f64 * 0.02) as usize];
        let hi = depths[((depths.len() as f64 * 0.98) as usize).min(depths.len() - 1)];
        return ((lo * 0.5).max(1e-6), (hi * 1.5).max(lo * 0.5 + 1e-6));
    }
    // Fallback: camera distance to the scene centroid.
    let n = points.len().max(1) as f64;
    let centroid = points.iter().fold(Vec3F64::ZERO, |acc, p| acc + p.position) / n;
    let d = (centroid - centre).length().max(1e-6);
    (0.5 * d, 1.5 * d)
}

/// Estimate one view's depth map.
fn sweep_view(
    view: usize,
    gray_frames: &[Image<u8, 1>],
    cameras: &[PinholeCamera],
    views: &[Option<Pose3d>],
    reconstruction: &Reconstruction,
    cfg: &DenseConfig,
) -> Option<DepthMap> {
    let ref_pose = views[view]?;
    let ref_cam = &cameras[view];
    let gray = &gray_frames[view];
    let neighbours = select_neighbors(views, view, cfg.neighbors);
    if neighbours.is_empty() {
        return None;
    }

    let (src_w, src_h) = (gray.width(), gray.height());
    let scale = cfg.scale.clamp(0.1, 1.0);
    let width = ((src_w as f64 * scale).round() as usize).max(2);
    let height = ((src_h as f64 * scale).round() as usize).max(2);
    let (d_min, d_max) = depth_range(
        &reconstruction.points,
        &ref_pose.inverse(),
        ref_pose.translation,
    );
    let rho_lo = 1.0 / d_max;
    let rho_hi = 1.0 / d_min;

    // Per-neighbour precomputed data: homography per plane is rebuilt per row
    // (cheap) to keep the hot loop free of shared mutable state.
    let rel: Vec<(usize, Mat3F64, Vec3F64)> = neighbours
        .iter()
        .map(|&j| {
            let (r, t) = relative_rt(&ref_pose, &views[j].expect("registered neighbour"));
            (j, r, t)
        })
        .collect();

    let window = cfg.window.max(3) | 1; // odd
    let window = window.min(MAX_WINDOW);
    let half = window / 2;
    let mut inv_depth = vec![0.0f32; width * height];

    inv_depth
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let sy = row as f64 / scale;
            for (col, out) in out_row.iter_mut().enumerate() {
                let sx = col as f64 / scale;

                // Reference window statistics are plane-independent.
                let mut ref_win = [0.0f32; MAX_WINDOW * MAX_WINDOW];
                let mut ok = true;
                for (i, dy) in (-(half as isize)..=half as isize).enumerate() {
                    for (j, dx) in (-(half as isize)..=half as isize).enumerate() {
                        let px = sx + dx as f64 / scale;
                        let py = sy + dy as f64 / scale;
                        match sample_bilinear(gray.as_slice(), src_w, src_h, px, py) {
                            Some(v) => ref_win[i * window + j] = v,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let ref_win = &ref_win[..window * window];

                let mut best_ncc = f32::NEG_INFINITY;
                let mut best_rho = 0.0f64;

                for plane in 0..cfg.planes {
                    let frac = if cfg.planes > 1 {
                        plane as f64 / (cfg.planes - 1) as f64
                    } else {
                        0.0
                    };
                    let rho = rho_lo + (rho_hi - rho_lo) * frac;

                    let mut ncc_sum = 0.0f32;
                    let mut ncc_count = 0.0f32;
                    for &(j, ref r, ref t) in &rel {
                        let nbr_cam = &cameras[j];
                        let nbr_gray = &gray_frames[j];
                        let h = plane_homography(ref_cam, nbr_cam, r, t, rho);
                        let Some((nx, ny)) = apply_homography(&h, sx, sy) else {
                            continue;
                        };
                        let mut nbr_win = [0.0f32; MAX_WINDOW * MAX_WINDOW];
                        let mut nbr_ok = true;
                        for (i, dy) in (-(half as isize)..=half as isize).enumerate() {
                            for (j2, dx) in (-(half as isize)..=half as isize).enumerate() {
                                let px = nx + dx as f64 / scale;
                                let py = ny + dy as f64 / scale;
                                match sample_bilinear(nbr_gray.as_slice(), src_w, src_h, px, py) {
                                    Some(v) => nbr_win[i * window + j2] = v,
                                    None => {
                                        nbr_ok = false;
                                        break;
                                    }
                                }
                            }
                            if !nbr_ok {
                                break;
                            }
                        }
                        if !nbr_ok {
                            continue;
                        }
                        ncc_sum += ncc(ref_win, &nbr_win[..window * window]);
                        ncc_count += 1.0;
                    }
                    if ncc_count == 0.0 {
                        continue;
                    }
                    let score = ncc_sum / ncc_count;
                    if score > best_ncc {
                        best_ncc = score;
                        best_rho = rho;
                    }
                }

                if best_ncc >= cfg.ncc_min && best_rho > 0.0 {
                    *out = best_rho as f32;
                }
            }
        });

    Some(DepthMap {
        view,
        width,
        height,
        scale,
        inv_depth,
        neighbors: neighbours,
    })
}

/// Estimate depth maps for every registered view.
///
/// The returned vector is indexed by frame; unregistered views hold `None`.
pub fn estimate_depth_maps(
    reconstruction: &Reconstruction,
    cameras: &[PinholeCamera],
    gray_frames: &[Image<u8, 1>],
    cfg: &DenseConfig,
) -> Result<Vec<Option<DepthMap>>, Box<dyn Error>> {
    let registered: Vec<usize> = reconstruction
        .views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|_| i))
        .collect();
    if registered.is_empty() {
        return Err("no registered views to densify".into());
    }

    let mut maps: Vec<Option<DepthMap>> = (0..reconstruction.views.len()).map(|_| None).collect();

    let computed: Vec<(usize, DepthMap)> = registered
        .par_iter()
        .filter_map(|&i| {
            sweep_view(
                i,
                gray_frames,
                cameras,
                &reconstruction.views,
                reconstruction,
                cfg,
            )
            .map(|m| (i, m))
        })
        .collect();
    for (i, m) in computed {
        maps[i] = Some(m);
    }

    // Left-right consistency: a depth must reproject into a neighbour and
    // agree with that neighbour's own estimate.
    for i in registered {
        let Some(map) = maps[i].as_ref() else {
            continue;
        };
        let (width, height, scale, view_idx, neighbors) = (
            map.width,
            map.height,
            map.scale,
            map.view,
            map.neighbors.clone(),
        );
        let pose_i = reconstruction.views[i].expect("registered");
        let cam_i = &cameras[i];
        let mut filtered = map.inv_depth.clone();
        for y in 0..height {
            for x in 0..width {
                let rho = filtered[y * width + x];
                if rho <= 0.0 {
                    continue;
                }
                let d = 1.0 / rho as f64;
                let pix = Vec2F64::new(x as f64 / scale, y as f64 / scale);
                let ray = k_inverse(cam_i) * Vec3F64::new(pix.x, pix.y, 1.0);
                let p_cam = ray * d;
                let p_world = pose_i.transform_point(&p_cam);

                let mut consistent = false;
                for &j in &neighbors {
                    let Some(nbr_map) = maps[j].as_ref() else {
                        continue;
                    };
                    let w2c = reconstruction.views[j].expect("registered").inverse();
                    let p_nbr = w2c.transform_point(&p_world);
                    if p_nbr.z <= 1e-9 {
                        continue;
                    }
                    let Some(nbr_pix) = cameras[j].project_to_pixel(&p_nbr, 1e-9) else {
                        continue;
                    };
                    let nx = (nbr_pix.x * nbr_map.scale).round() as isize;
                    let ny = (nbr_pix.y * nbr_map.scale).round() as isize;
                    if nx < 0
                        || ny < 0
                        || nx >= nbr_map.width as isize
                        || ny >= nbr_map.height as isize
                    {
                        continue;
                    }
                    let Some(d_nbr) = nbr_map.depth_at(nx as usize, ny as usize) else {
                        continue;
                    };
                    if (p_nbr.z - d_nbr).abs() <= cfg.lr_tolerance * d_nbr {
                        consistent = true;
                        break;
                    }
                }
                if !consistent {
                    filtered[y * width + x] = 0.0;
                }
            }
        }
        maps[i] = Some(DepthMap {
            view: view_idx,
            width,
            height,
            scale,
            inv_depth: filtered,
            neighbors,
        });
    }

    Ok(maps)
}

/// Agreement between the dense depth maps and the sparse reconstruction.
///
/// Projects every sparse point into the views that have a depth map and
/// compares the point's camera depth with the map's estimate at that pixel.
/// Returns `(samples, median_relative_error, p90_relative_error)`.
pub fn sparse_depth_check(
    reconstruction: &Reconstruction,
    maps: &[Option<DepthMap>],
    cameras: &[PinholeCamera],
) -> (usize, f64, f64) {
    let mut errors: Vec<f64> = Vec::new();
    for (view, map) in maps.iter().enumerate() {
        let (Some(map), Some(pose)) = (map, reconstruction.views[view]) else {
            continue;
        };
        let w2c = pose.inverse();
        for point in &reconstruction.points {
            let p_cam = w2c.transform_point(&point.position);
            if p_cam.z <= 1e-9 {
                continue;
            }
            let Some(pix) = cameras[view].project_to_pixel(&p_cam, 1e-9) else {
                continue;
            };
            let mx = (pix.x * map.scale).round() as isize;
            let my = (pix.y * map.scale).round() as isize;
            if mx < 0 || my < 0 || mx >= map.width as isize || my >= map.height as isize {
                continue;
            }
            if let Some(depth) = map.depth_at(mx as usize, my as usize) {
                errors.push((p_cam.z - depth).abs() / p_cam.z);
            }
        }
    }
    if errors.is_empty() {
        return (0, f64::NAN, f64::NAN);
    }
    errors.sort_by(f64::total_cmp);
    let at = |q: f64| errors[((errors.len() as f64 * q) as usize).min(errors.len() - 1)];
    (errors.len(), at(0.5), at(0.9))
}

/// Backproject every valid depth pixel into a coloured world-frame cloud.
pub fn dense_cloud(
    maps: &[Option<DepthMap>],
    cameras: &[PinholeCamera],
    views: &[Option<Pose3d>],
    rgb_frames: &[Image<u8, 3>],
) -> DenseCloud {
    let mut points = Vec::new();
    let mut colors = Vec::new();
    for map in maps.iter().flatten() {
        let Some(pose) = views[map.view] else {
            continue;
        };
        let cam = &cameras[map.view];
        let rgb = &rgb_frames[map.view];
        let (rw, rh) = (rgb.width(), rgb.height());
        let k_inv = k_inverse(cam);
        for y in 0..map.height {
            for x in 0..map.width {
                let Some(d) = map.depth_at(x, y) else {
                    continue;
                };
                let sx = x as f64 / map.scale;
                let sy = y as f64 / map.scale;
                let ray = k_inv * Vec3F64::new(sx, sy, 1.0);
                let p_world = pose.transform_point(&(ray * d));
                let color =
                    sample_rgb_bilinear(rgb.as_slice(), rw, rh, sx, sy).unwrap_or([0, 0, 0]);
                points.push(p_world.to_array());
                colors.push(color);
            }
        }
    }
    DenseCloud { points, colors }
}

/// Bilinear RGB sample; `None` outside the image.
pub(crate) fn sample_rgb_bilinear(
    img: &[u8],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> Option<[u8; 3]> {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let at = |xx: usize, yy: usize| {
        let i = (yy * width + xx) * 3;
        [img[i], img[i + 1], img[i + 2]]
    };
    let p00 = at(x0, y0);
    let p10 = at(x1, y0);
    let p01 = at(x0, y1);
    let p11 = at(x1, y1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 + (p10[c] as f32 - p00[c] as f32) * fx;
        let bottom = p01[c] as f32 + (p11[c] as f32 - p01[c] as f32) * fx;
        out[c] = (top + (bottom - top) * fy).clamp(0.0, 255.0) as u8;
    }
    Some(out)
}

/// Downsample a dense cloud with the kornia-3d sparse voxel grid.
///
/// `voxel_size` is in world units; one point (and its colour) survives per
/// occupied voxel.
pub fn voxel_downsample_cloud(
    cloud: &DenseCloud,
    voxel_size: f64,
) -> Result<DenseCloud, Box<dyn Error>> {
    use kornia_3d::pointcloud::PointCloud;
    use kornia_3d::voxel::{voxel_downsample, DownsampleMode, VoxelLayout};

    let pc = PointCloud::new(cloud.points.clone(), Some(cloud.colors.clone()), None);
    let layout = VoxelLayout::new(voxel_size, [0.0; 3])?;
    let down = voxel_downsample(&pc, &layout, DownsampleMode::Centroid)?;
    Ok(DenseCloud {
        points: down.points().clone(),
        colors: down.colors().cloned().unwrap_or_default(),
    })
}

/// Unused helper kept for symmetry with the neighbour bookkeeping.
#[allow(dead_code)]
fn _centre_of(views: &[Option<Pose3d>], i: usize) -> Option<Vec3F64> {
    views.get(i).and_then(|v| v.as_ref()).map(|p| p.translation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_calib::Point;

    fn cam(fx: f64) -> PinholeCamera {
        PinholeCamera {
            fx,
            fy: fx,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn pose(translation: Vec3F64) -> Pose3d {
        // T_world_cam with identity rotation.
        Pose3d::new(Mat3F64::IDENTITY, translation)
    }

    #[test]
    fn homography_maps_plane_points_to_the_neighbour_pixel() {
        let c = cam(400.0);
        let ref_pose = pose(Vec3F64::ZERO);
        let nbr_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-0.3, 0.0, 0.0));
        let (r, t) = relative_rt(&ref_pose, &nbr_pose);
        let depth = 2.0;
        let h = plane_homography(&c, &c, &r, &t, 1.0 / depth);

        // A world point on the reference plane z = depth.
        for (u, v) in [(100.0, 100.0), (320.0, 240.0), (600.0, 400.0)] {
            let x = (u - c.cx) / c.fx * depth;
            let y = (v - c.cy) / c.fy * depth;
            let p_world = Vec3F64::new(x, y, depth);
            let p_nbr = nbr_pose.inverse().transform_point(&p_world);
            let (want_u, want_v) = (
                c.fx * p_nbr.x / p_nbr.z + c.cx,
                c.fy * p_nbr.y / p_nbr.z + c.cy,
            );
            let (got_u, got_v) = apply_homography(&h, u, v).expect("maps");
            assert!((got_u - want_u).abs() < 1e-6, "{got_u} vs {want_u}");
            assert!((got_v - want_v).abs() < 1e-6, "{got_v} vs {want_v}");
        }
    }

    #[test]
    fn ncc_of_identical_windows_is_one_and_inverted_is_minus_one() {
        let a = [10.0, 20.0, 30.0, 40.0];
        assert!((ncc(&a, &a) - 1.0).abs() < 1e-6);
        let b = [40.0, 30.0, 20.0, 10.0];
        assert!((ncc(&a, &b) + 1.0).abs() < 1e-6);
        let flat = [5.0; 4];
        assert_eq!(ncc(&a, &flat), 0.0);
    }

    #[test]
    fn bilinear_sample_hits_pixel_centres_and_edges() {
        let img = [0u8, 100, 200, 255];
        assert_eq!(sample_bilinear(&img, 2, 2, 0.0, 0.0), Some(0.0));
        assert_eq!(sample_bilinear(&img, 2, 2, 1.0, 0.0), Some(100.0));
        assert_eq!(sample_bilinear(&img, 2, 2, 0.5, 0.5), Some(138.75));
        assert_eq!(sample_bilinear(&img, 2, 2, -0.1, 0.0), None);
    }

    #[test]
    fn select_neighbors_picks_nearest_registered_views() {
        let views = vec![
            Some(pose(Vec3F64::ZERO)),
            Some(pose(Vec3F64::new(0.1, 0.0, 0.0))),
            None,
            Some(pose(Vec3F64::new(1.0, 0.0, 0.0))),
        ];
        let n = select_neighbors(&views, 0, 2);
        assert_eq!(n, vec![1, 3]);
    }

    #[test]
    fn depth_range_tracks_the_sparse_points() {
        let points: Vec<Point> = (0..20)
            .map(|i| Point {
                position: Vec3F64::new(0.0, 0.0, 1.0 + 0.05 * i as f64),
                track_id: i,
            })
            .collect();
        let (lo, hi) = depth_range(&points, &Pose3d::IDENTITY, Vec3F64::ZERO);
        assert!((0.4..1.0).contains(&lo), "lo = {lo}");
        assert!(hi > 1.9 && hi < 3.0, "hi = {hi}");
    }
}
