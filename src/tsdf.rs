//! Truncated signed-distance-field fusion of the dense depth maps.
//!
//! Every depth map votes on the space it sees: in front of the measured
//! surface is empty (positive distance), behind it is solid (negative). The
//! weighted average of all votes is a signed distance field whose zero level
//! set is the surface. Voxels farther than the truncation distance from any
//! measurement are left unobserved (weight zero).

use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use rayon::prelude::*;

use crate::dense::DepthMap;

/// Tunables for fusion.
pub struct TsdfConfig {
    /// Maximum number of grid points along the longest scene axis.
    pub resolution: usize,
    /// Truncation distance, in voxels.
    pub trunc_voxels: f64,
    /// Measured relative depth error (e.g. from the sparse-depth check). When
    /// present, the truncation band is widened to `1.5 * error * median depth`,
    /// so noisy depth maps vote consistently instead of cancelling out.
    pub depth_noise_relative: Option<f64>,
    /// NCC below which a depth pixel carries no weight (matches the sweep's
    /// acceptance threshold).
    pub ncc_min: f32,
}

/// Minimum truncation distance as a fraction of the median camera-to-scene
/// distance. Depth error grows with distance (roughly proportionally), so a
/// band tied only to the voxel size can be narrower than the noise it must
/// absorb — the fused votes then disagree and the surface washes out.
const MIN_TRUNC_RELATIVE_DEPTH: f64 = 0.03;

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            resolution: 192,
            trunc_voxels: 4.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        }
    }
}

/// A sampled signed distance field.
///
/// `sdf[idx]` holds the truncated signed distance divided by the truncation
/// distance, so values lie in `[-1, 1]`: positive outside, negative inside.
/// `weight == 0` marks an unobserved grid point.
pub struct Volume {
    /// World position of grid point `(0, 0, 0)`.
    pub origin: Vec3F64,
    /// Grid spacing.
    pub voxel_size: f64,
    /// Effective truncation distance in world units.
    pub trunc: f64,
    /// Grid dimensions (grid points, not cells).
    pub dims: [usize; 3],
    /// Signed distance per grid point, normalized by the truncation distance.
    pub sdf: Vec<f32>,
    /// Observation weight per grid point.
    pub weight: Vec<f32>,
}

impl Volume {
    /// Flat index of grid point `(i, j, k)`.
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims[1] + j) * self.dims[2] + k
    }

    /// World position of grid point `(i, j, k)`.
    #[inline]
    pub fn position(&self, i: usize, j: usize, k: usize) -> Vec3F64 {
        self.origin + Vec3F64::new(i as f64, j as f64, k as f64) * self.voxel_size
    }

    /// Trilinear signed-distance sample; unobserved corners count as `+1`
    /// (empty), so the surface only appears where observations exist.
    pub fn sample(&self, p: Vec3F64) -> f32 {
        let rel = (p - self.origin) / self.voxel_size;
        let (x, y, z) = (rel.x, rel.y, rel.z);
        if x < 0.0
            || y < 0.0
            || z < 0.0
            || x > (self.dims[0] - 1) as f64
            || y > (self.dims[1] - 1) as f64
            || z > (self.dims[2] - 1) as f64
        {
            return 1.0;
        }
        let (x0, y0, z0) = (x.floor() as usize, y.floor() as usize, z.floor() as usize);
        let (x1, y1, z1) = (
            (x0 + 1).min(self.dims[0] - 1),
            (y0 + 1).min(self.dims[1] - 1),
            (z0 + 1).min(self.dims[2] - 1),
        );
        let (fx, fy, fz) = (
            (x - x0 as f64) as f32,
            (y - y0 as f64) as f32,
            (z - z0 as f64) as f32,
        );
        let at = |i: usize, j: usize, k: usize| {
            let idx = self.index(i, j, k);
            if self.weight[idx] > 0.0 {
                self.sdf[idx]
            } else {
                1.0
            }
        };
        let c00 = at(x0, y0, z0) * (1.0 - fx) + at(x1, y0, z0) * fx;
        let c10 = at(x0, y1, z0) * (1.0 - fx) + at(x1, y1, z0) * fx;
        let c01 = at(x0, y0, z1) * (1.0 - fx) + at(x1, y0, z1) * fx;
        let c11 = at(x0, y1, z1) * (1.0 - fx) + at(x1, y1, z1) * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// Central-difference gradient of the field (points outward: negative
    /// inside → positive outside).
    pub fn gradient(&self, p: Vec3F64) -> Vec3F64 {
        let h = self.voxel_size;
        let dx = (self.sample(p + Vec3F64::new(h, 0.0, 0.0))
            - self.sample(p - Vec3F64::new(h, 0.0, 0.0))) as f64;
        let dy = (self.sample(p + Vec3F64::new(0.0, h, 0.0))
            - self.sample(p - Vec3F64::new(0.0, h, 0.0))) as f64;
        let dz = (self.sample(p + Vec3F64::new(0.0, 0.0, h))
            - self.sample(p - Vec3F64::new(0.0, 0.0, h))) as f64;
        let g = Vec3F64::new(dx, dy, dz);
        let len = g.length();
        if len > 1e-12 {
            g / len
        } else {
            Vec3F64::new(0.0, 0.0, 1.0)
        }
    }
}

/// Axis-aligned bounds of a point set.
fn bounds(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    let first = *points.first()?;
    let mut lo = first;
    let mut hi = first;
    for p in points {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    Some((lo, hi))
}

/// Outlier-robust bounds: per-axis 1st and 99th percentiles.
///
/// A few erroneous depth points must not stretch the volume: every extra
/// unit of bounding box is voxels not spent on the object.
fn robust_bounds(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    if points.len() < 16 {
        return bounds(points);
    }
    let n = points.len();
    let mut axes: [Vec<f64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for p in points {
        for (axis, values) in axes.iter_mut().enumerate() {
            values.push(p[axis]);
        }
    }
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for (axis, values) in axes.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        lo[axis] = values[n / 100];
        hi[axis] = values[(n * 99 / 100).min(n - 1)];
        if hi[axis] <= lo[axis] {
            return bounds(points);
        }
    }
    Some((lo, hi))
}

/// Fuse depth maps into a volume.
///
/// The grid covers the bounding box of `points` (typically the dense cloud)
/// with a small padding, at a resolution derived from `cfg.resolution`.
///
/// # Errors
///
/// Returns an error when there are no points to bound or no depth maps.
pub fn fuse(
    maps: &[Option<DepthMap>],
    cameras: &[PinholeCamera],
    views: &[Option<Pose3d>],
    points: &[[f64; 3]],
    cfg: &TsdfConfig,
) -> Result<Volume, Box<dyn Error>> {
    let (lo, hi) = robust_bounds(points).ok_or("no points to bound the TSDF volume")?;
    let active: Vec<&DepthMap> = maps.iter().flatten().collect();
    if active.is_empty() {
        return Err("no depth maps to fuse".into());
    }

    let extent = [
        (hi[0] - lo[0]).max(1e-6),
        (hi[1] - lo[1]).max(1e-6),
        (hi[2] - lo[2]).max(1e-6),
    ];
    let longest = extent[0].max(extent[1]).max(extent[2]).max(1e-6);
    let voxel_size = longest / cfg.resolution.max(8) as f64;

    // Physical floor for the truncation band: depth uncertainty scales with
    // the viewing distance, not with the object size.
    let centroid = Vec3F64::new(
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    );
    let mut distances: Vec<f64> = active
        .iter()
        .filter_map(|m| views[m.view].map(|p| (p.translation - centroid).length()))
        .collect();
    distances.sort_by(f64::total_cmp);
    let median_distance = distances
        .get(distances.len() / 2)
        .copied()
        .unwrap_or(longest);

    let noise_floor = 1.5
        * cfg
            .depth_noise_relative
            .unwrap_or(MIN_TRUNC_RELATIVE_DEPTH)
            .clamp(0.005, 0.15)
        * median_distance;
    let trunc = (cfg.trunc_voxels.max(1.0) * voxel_size)
        .max(noise_floor)
        .max(1.5 * voxel_size);
    let pad = trunc + 2.0 * voxel_size;
    let origin = Vec3F64::new(lo[0] - pad, lo[1] - pad, lo[2] - pad);
    let dim = |e: f64| (((e + 2.0 * pad) / voxel_size).ceil() as usize + 1).max(2);
    let dims = [dim(extent[0]), dim(extent[1]), dim(extent[2])];

    let n = dims[0] * dims[1] * dims[2];
    let mut sdf = vec![0.0f32; n];
    let mut weight = vec![0.0f32; n];

    // Parallel over x-slices; each slice owns disjoint output ranges.
    let slice = dims[1] * dims[2];
    let done_slices = AtomicUsize::new(0);
    sdf.par_chunks_mut(slice)
        .zip(weight.par_chunks_mut(slice))
        .enumerate()
        .for_each(|(i, (sdf_slice, weight_slice))| {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let p = origin + Vec3F64::new(i as f64, j as f64, k as f64) * voxel_size;
                    let mut w_sum = 0.0f32;
                    let mut d_sum = 0.0f32;
                    for map in &active {
                        let Some(pose) = views[map.view] else {
                            continue;
                        };
                        let w2c = pose.inverse();
                        let p_cam = w2c.transform_point(&p);
                        if p_cam.z <= 1e-9 {
                            continue;
                        }
                        let Some(pix) = cameras[map.view].project_to_pixel(&p_cam, 1e-9) else {
                            continue;
                        };
                        let mx = (pix.x * map.scale).round() as isize;
                        let my = (pix.y * map.scale).round() as isize;
                        if mx < 0 || my < 0 || mx >= map.width as isize || my >= map.height as isize
                        {
                            continue;
                        }
                        let Some(depth) = map.depth_at(mx as usize, my as usize) else {
                            continue;
                        };
                        // Positive in front of the surface (empty), negative
                        // behind it (solid).
                        let dist = depth - p_cam.z;
                        if dist.abs() > trunc {
                            continue;
                        }
                        // Curless-Levoy weighting: trust a vote by its
                        // photometric confidence and by how squarely the
                        // surface faces this camera.
                        let pix = my as usize * map.width + mx as usize;
                        let conf = map.confidence.get(pix).copied().unwrap_or(1.0);
                        let conf_w = ((conf - cfg.ncc_min).max(0.01) as f64).powi(2);
                        let n = map.normals.get(pix).copied().unwrap_or([0.0; 3]);
                        let angle_w = if n == [0.0; 3] {
                            1.0
                        } else {
                            let normal = Vec3F64::new(n[0] as f64, n[1] as f64, n[2] as f64);
                            normal.dot((-p_cam).normalize()).clamp(0.05, 1.0)
                        };
                        let w = (conf_w * angle_w) as f32;
                        w_sum += w;
                        d_sum += dist as f32 * w;
                    }
                    let idx = j * dims[2] + k;
                    if w_sum > 0.0 {
                        let mean = d_sum / w_sum;
                        sdf_slice[idx] = (mean / trunc as f32).clamp(-1.0, 1.0);
                        weight_slice[idx] = w_sum;
                    }
                }
            }
            let n = done_slices.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(8) || n == dims[0] {
                crate::error::progress(8, n, dims[0]);
            }
        });

    Ok(Volume {
        origin,
        voxel_size,
        trunc,
        dims,
        sdf,
        weight,
    })
}

/// Weight given to voxels synthesized by [`fill_holes`]: enough for the mesher
/// to treat them as evidence, small enough to distinguish them from camera
/// observations (whose weights are at least 1).
const FILLED_WEIGHT: f32 = 0.5;

/// Six-neighbour offsets.
const NEIGHBOURS6: [(isize, isize, isize); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Fill unobserved voxels by **volumetric diffusion** (Davis et al. 2002).
///
/// Unobserved voxels adjacent to observed ones take the weighted mean of their
/// observed neighbours; the front then advances one voxel per iteration, so
/// small and medium holes in the signed distance field are closed before
/// isosurface extraction. Values are computed from a snapshot and applied
/// together, keeping the result independent of traversal order.
///
/// Returns the number of voxels filled.
pub fn fill_holes(volume: &mut Volume, iterations: usize) -> usize {
    if iterations == 0 || volume.weight.is_empty() {
        return 0;
    }
    let dims = volume.dims;
    let mut filled = 0usize;

    let mut frontier: Vec<usize> = Vec::new();
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let idx = volume.index(i, j, k);
                if volume.weight[idx] == 0.0 && has_known_neighbour(volume, i, j, k) {
                    frontier.push(idx);
                }
            }
        }
    }

    for _ in 0..iterations {
        if frontier.is_empty() {
            break;
        }
        // Snapshot: compute all means before writing any value.
        let mut updates: Vec<(usize, f32)> = Vec::with_capacity(frontier.len());
        for &idx in &frontier {
            if volume.weight[idx] > 0.0 {
                continue;
            }
            let (i, j, k) = unindex(dims, idx);
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for (di, dj, dk) in NEIGHBOURS6 {
                if let Some(nbr) = neighbour_index(dims, i, j, k, di, dj, dk) {
                    if volume.weight[nbr] > 0.0 {
                        sum += volume.sdf[nbr];
                        count += 1;
                    }
                }
            }
            if count > 0 {
                updates.push((idx, sum / count as f32));
            }
        }
        if updates.is_empty() {
            break;
        }
        let mut next: Vec<usize> = Vec::new();
        for &(idx, value) in &updates {
            volume.sdf[idx] = value;
            volume.weight[idx] = FILLED_WEIGHT;
            filled += 1;
            let (i, j, k) = unindex(dims, idx);
            for (di, dj, dk) in NEIGHBOURS6 {
                if let Some(nbr) = neighbour_index(dims, i, j, k, di, dj, dk) {
                    if volume.weight[nbr] == 0.0 {
                        next.push(nbr);
                    }
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        frontier = next;
    }

    filled
}

/// Box-filter the signed distance field over observed voxels.
///
/// Voxel-scale noise in the depth maps becomes pyramids and stair-steps in the
/// extracted surface. Smoothing the *field* before isosurface extraction
/// removes them at the source, which mesh smoothing cannot do as cheaply.
/// Unobserved voxels stay untouched.
pub fn smooth_volume(volume: &mut Volume, iterations: usize) {
    for _ in 0..iterations {
        let sdf = volume.sdf.clone();
        let weight = volume.weight.clone();
        let dims = volume.dims;
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let idx = volume.index(i, j, k);
                    if weight[idx] == 0.0 {
                        continue;
                    }
                    let mut sum = sdf[idx];
                    let mut count = 1.0f32;
                    for (di, dj, dk) in NEIGHBOURS6 {
                        if let Some(nbr) = neighbour_index(dims, i, j, k, di, dj, dk) {
                            if weight[nbr] > 0.0 {
                                sum += sdf[nbr];
                                count += 1.0;
                            }
                        }
                    }
                    volume.sdf[idx] = sum / count;
                }
            }
        }
    }
}

/// Whether a grid point has at least one observed six-neighbour.
fn has_known_neighbour(volume: &Volume, i: usize, j: usize, k: usize) -> bool {
    NEIGHBOURS6.iter().any(|&(di, dj, dk)| {
        neighbour_index(volume.dims, i, j, k, di, dj, dk)
            .is_some_and(|nbr| volume.weight[nbr] > 0.0)
    })
}

/// Flat index of the six-neighbour, or `None` outside the grid.
fn neighbour_index(
    dims: [usize; 3],
    i: usize,
    j: usize,
    k: usize,
    di: isize,
    dj: isize,
    dk: isize,
) -> Option<usize> {
    let ni = i as isize + di;
    let nj = j as isize + dj;
    let nk = k as isize + dk;
    if ni < 0
        || nj < 0
        || nk < 0
        || ni >= dims[0] as isize
        || nj >= dims[1] as isize
        || nk >= dims[2] as isize
    {
        return None;
    }
    Some((ni as usize * dims[1] + nj as usize) * dims[2] + nk as usize)
}

/// Grid coordinates of a flat index.
fn unindex(dims: [usize; 3], idx: usize) -> (usize, usize, usize) {
    let k = idx % dims[2];
    let j = (idx / dims[2]) % dims[1];
    let i = idx / (dims[1] * dims[2]);
    (i, j, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DepthMap;

    fn cam() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 160.0,
            cy: 120.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// A depth map of a fronto-parallel plane at `distance` from the camera.
    fn plane_depth(view: usize, distance: f32) -> DepthMap {
        let (width, height) = (320, 240);
        DepthMap {
            view,
            width,
            height,
            scale: 1.0,
            inv_depth: vec![1.0 / distance; width * height],
            confidence: vec![1.0; width * height],
            normals: vec![[0.0; 3]; width * height],
            neighbors: Vec::new(),
        }
    }

    #[test]
    fn plane_fusion_recovers_the_true_signed_distance() {
        let cameras = vec![cam()];
        let views = vec![Some(Pose3d::IDENTITY)];
        let map = Some(plane_depth(0, 2.0));
        let maps = vec![map];
        // Points that bound the region of interest around the plane.
        let points = vec![[-0.2, -0.2, 1.8], [0.2, 0.2, 2.2], [-0.2, 0.2, 1.8]];
        let cfg = TsdfConfig {
            resolution: 32,
            trunc_voxels: 4.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        };
        let vol = fuse(&maps, &cameras, &views, &points, &cfg).unwrap();

        // Sample the field just in front of and behind the plane, at the
        // optical axis. The normalized value must match dist / trunc.
        let trunc = vol.trunc;
        let front = Vec3F64::new(0.0, 0.0, 2.0 - 0.5 * trunc);
        let behind = Vec3F64::new(0.0, 0.0, 2.0 + 0.5 * trunc);
        let s_front = vol.sample(front);
        let s_behind = vol.sample(behind);
        assert!((s_front - 0.5).abs() < 0.15, "front {s_front}");
        assert!((s_behind + 0.5).abs() < 0.15, "behind {s_behind}");
        // Exactly on the plane the field crosses zero.
        assert!(vol.sample(Vec3F64::new(0.0, 0.0, 2.0)).abs() < 0.1);
    }

    #[test]
    fn voxels_beyond_the_truncation_band_are_unobserved() {
        let cameras = vec![cam()];
        let views = vec![Some(Pose3d::IDENTITY)];
        let maps = vec![Some(plane_depth(0, 2.0))];
        let points = vec![[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]];
        let cfg = TsdfConfig {
            resolution: 32,
            trunc_voxels: 3.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        };
        let vol = fuse(&maps, &cameras, &views, &points, &cfg).unwrap();
        let far_in_front = Vec3F64::new(0.0, 0.0, 2.0 - 10.0 * vol.voxel_size);
        assert_eq!(
            vol.sample(far_in_front),
            1.0,
            "unobserved space reads as empty"
        );
    }

    #[test]
    fn hole_filling_reduces_unobserved_voxels() {
        let cameras = vec![cam()];
        let views = vec![Some(Pose3d::IDENTITY)];
        let maps = vec![Some(plane_depth(0, 2.0))];
        let points = vec![[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]];
        let cfg = TsdfConfig {
            resolution: 24,
            trunc_voxels: 4.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        };
        let mut vol = fuse(&maps, &cameras, &views, &points, &cfg).unwrap();

        // Punch a hole: forget a slab of grid points near the plane centre.
        let mut removed = 0;
        for i in 0..vol.dims[0] {
            for j in 0..vol.dims[1] {
                for k in 0..vol.dims[2] {
                    let idx = vol.index(i, j, k);
                    let p = vol.position(i, j, k);
                    if vol.weight[idx] > 0.0 && (p.x.abs() < 0.08 && p.y.abs() < 0.08) {
                        vol.weight[idx] = 0.0;
                        removed += 1;
                    }
                }
            }
        }
        assert!(removed > 0, "the test must actually create a hole");
        let filled = fill_holes(&mut vol, 8);
        assert!(filled > 0, "diffusion must fill something");
        // The hole is bridged: the field is observed again where it was cut.
        let mut still_missing = 0;
        for i in 0..vol.dims[0] {
            for j in 0..vol.dims[1] {
                for k in 0..vol.dims[2] {
                    let idx = vol.index(i, j, k);
                    let p = vol.position(i, j, k);
                    if p.x.abs() < 0.04
                        && p.y.abs() < 0.04
                        && (p.z - 2.0).abs() < 0.08
                        && vol.weight[idx] == 0.0
                    {
                        still_missing += 1;
                    }
                }
            }
        }
        assert_eq!(still_missing, 0, "the centre of the hole must be filled");
    }

    #[test]
    fn hole_filling_is_zero_with_no_iterations() {
        let cameras = vec![cam()];
        let views = vec![Some(Pose3d::IDENTITY)];
        let maps = vec![Some(plane_depth(0, 2.0))];
        let points = vec![[0.0, 0.0, 1.8], [0.0, 0.0, 2.2]];
        let cfg = TsdfConfig {
            resolution: 16,
            trunc_voxels: 4.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        };
        let mut vol = fuse(&maps, &cameras, &views, &points, &cfg).unwrap();
        assert_eq!(fill_holes(&mut vol, 0), 0);
    }

    #[test]
    fn gradient_points_along_the_surface_normal() {
        let cameras = vec![cam()];
        let views = vec![Some(Pose3d::IDENTITY)];
        let maps = vec![Some(plane_depth(0, 2.0))];
        let points = vec![[0.0, 0.0, 1.8], [0.0, 0.0, 2.2]];
        let cfg = TsdfConfig {
            resolution: 24,
            trunc_voxels: 4.0,
            depth_noise_relative: None,
            ncc_min: 0.4,
        };
        let vol = fuse(&maps, &cameras, &views, &points, &cfg).unwrap();
        let g = vol.gradient(Vec3F64::new(0.0, 0.0, 2.0));
        // The camera looks along +z, so the visible surface's outward normal
        // (toward the empty space in front of it) points back at the camera:
        // the field increases as z decreases.
        assert!(g.z < -0.9, "gradient {g:?}");
    }
}
