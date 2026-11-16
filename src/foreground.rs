//! Foreground isolation: keep the orbit-centered object, drop the environment.
//!
//! The object being orbited is the only structure the camera sees from a **wide
//! angular range**; the table/background is seen from a narrow cone (or is a
//! plane). We score every sparse point by how many registered views observe it
//! and how widely those rays fan out, keep the well-supported points, and take
//! the largest spatially connected component. The result masks the sparse cloud
//! and hands the dense stage a tight bounding box to fuse inside — which is what
//! stops the table/background from inflating the TSDF volume and shattering the
//! object mesh.

use kornia_calib::Reconstruction;

use crate::dense;

/// Tuning for [`isolate_object`].
#[derive(Debug, Clone)]
pub struct ForegroundConfig {
    /// Minimum registered views that must observe a point.
    pub support_min: usize,
    /// A sighting counts as "central" when it projects within this fraction of
    /// the image half-diagonal from the centre. The capture centres the object,
    /// so object points are central in most of their views; background is not.
    pub central_radius_frac: f64,
    /// Minimum fraction of a point's sightings that must be central.
    pub centrality_min: f64,
    /// Voxel size for the largest-component step, as a fraction of the kept
    /// points' bounding-box diagonal.
    pub component_voxel_frac: f64,
}

impl Default for ForegroundConfig {
    fn default() -> Self {
        Self {
            support_min: 3,
            central_radius_frac: 0.5,
            centrality_min: 0.6,
            component_voxel_frac: 1.0 / 48.0,
        }
    }
}

/// Which sparse points belong to the object, plus a robust box around them.
#[derive(Debug, Clone)]
pub struct ObjectRegion {
    /// `keep[i]` for `reconstruction.points[i]`.
    pub keep: Vec<bool>,
    /// Robust lower corner of the object box (world frame).
    pub lo: [f64; 3],
    /// Robust upper corner of the object box (world frame).
    pub hi: [f64; 3],
    /// Number of points kept.
    pub kept: usize,
}

/// Isolate the object from `reconstruction`.
///
/// # Arguments
///
/// * `reconstruction` - The sparse solve, with views and surviving observations.
/// * `cfg` - Support/coverage thresholds.
///
/// # Returns
///
/// `Some(region)` when a confident object cluster is found, otherwise `None`
/// (too few points, or nothing passes the thresholds). `None` is a normal
/// outcome on a poor capture — the caller keeps the whole cloud.
pub fn isolate_object(
    reconstruction: &Reconstruction,
    cameras: &[kornia_3d::camera::PinholeCamera],
    image_size: (usize, usize),
    cfg: &ForegroundConfig,
) -> Option<ObjectRegion> {
    let points = &reconstruction.points;
    let n = points.len();
    if n == 0 {
        return None;
    }

    // Observing registered views per point.
    let mut observers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for obs in &reconstruction.observations {
        if let Some(list) = observers.get_mut(obs.point) {
            list.push(obs.view);
        }
    }

    // Per point: how many views see it, and how often it projects near the
    // image centre. The capture frames the object, so object points are central
    // in most views; the table/walls drift to the periphery.
    let (width, height) = (image_size.0 as f64, image_size.1 as f64);
    let half_diag = 0.5 * (width * width + height * height).sqrt();
    let mut keep = vec![false; n];
    let mut centrality = vec![0.0f64; n];
    for (pi, views) in observers.iter().enumerate() {
        let centre = points[pi].position;
        let mut central = 0usize;
        let mut seen = 0usize;
        for &v in views {
            let (Some(pose), Some(cam)) = (
                reconstruction.views.get(v).and_then(|p| p.as_ref()),
                cameras.get(v),
            ) else {
                continue;
            };
            let p_cam = pose.inverse().transform_point(&centre);
            if p_cam.z <= 1e-6 {
                continue;
            }
            seen += 1;
            if let Some(uv) = cam.project_to_pixel(&p_cam, 1e-6) {
                let d = ((uv.x - cam.cx).powi(2) + (uv.y - cam.cy).powi(2)).sqrt() / half_diag;
                if d <= cfg.central_radius_frac {
                    central += 1;
                }
            }
        }
        if seen == 0 {
            continue;
        }
        centrality[pi] = central as f64 / seen as f64;
        keep[pi] = seen >= cfg.support_min && centrality[pi] >= cfg.centrality_min;
    }

    let kept_pts: Vec<[f64; 3]> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| p.position.to_array())
        .collect();
    if kept_pts.len() < 16 {
        return None;
    }

    // Largest spatially connected component of the confident points.
    let diagonal = bbox_diagonal(&kept_pts).max(1e-9);
    let voxel = (diagonal * cfg.component_voxel_frac).max(1e-9);
    let mask = dense::largest_component_mask(&kept_pts, voxel);

    let mut final_keep = vec![false; n];
    let mut k = 0;
    for (pi, &was_kept) in keep.iter().enumerate() {
        if was_kept {
            if mask.get(k).copied().unwrap_or(false) {
                final_keep[pi] = true;
            }
            k += 1;
        }
    }
    let (lo, hi) = robust_bounds(points, &final_keep)?;
    // Reject an implausibly small cluster (a few noise points that happen to sit
    // near the image centre). It is almost never the object, and a tiny box
    // would over-resolve the depth and shatter the mesh.
    let all: Vec<[f64; 3]> = points.iter().map(|p| p.position.to_array()).collect();
    let full_diag = robust_extent(&all).max(1e-9);
    let box_diag =
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    if box_diag < 0.005 * full_diag {
        return None;
    }
    let kept = final_keep.iter().filter(|&&k| k).count();
    Some(ObjectRegion {
        keep: final_keep,
        lo,
        hi,
        kept,
    })
}

/// Largest per-axis diagonal of the point set.
fn bbox_diagonal(points: &[[f64; 3]]) -> f64 {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Robust diagonal (per-axis 5th-95th percentile span) of a point set.
fn robust_extent(points: &[[f64; 3]]) -> f64 {
    if points.len() < 16 {
        return 0.0;
    }
    let mut axes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for p in points {
        for a in 0..3 {
            axes[a].push(p[a]);
        }
    }
    let mut d = [0.0f64; 3];
    for a in 0..3 {
        axes[a].sort_by(f64::total_cmp);
        let lo = axes[a][(axes[a].len() as f64 * 0.05) as usize];
        let hi = axes[a][((axes[a].len() as f64 * 0.95) as usize).min(axes[a].len() - 1)];
        d[a] = hi - lo;
    }
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Robust (2nd-98th percentile) per-axis bounds of the kept points.
fn robust_bounds(points: &[kornia_calib::Point], keep: &[bool]) -> Option<([f64; 3], [f64; 3])> {
    let mut axes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, p) in points.iter().enumerate() {
        if keep[i] {
            axes[0].push(p.position.x);
            axes[1].push(p.position.y);
            axes[2].push(p.position.z);
        }
    }
    if axes[0].len() < 16 {
        return None;
    }
    let (mut lo, mut hi) = ([0.0; 3], [0.0; 3]);
    for (a, axis) in axes.iter_mut().enumerate() {
        axis.sort_by(f64::total_cmp);
        lo[a] = axis[(axis.len() as f64 * 0.02) as usize];
        hi[a] = axis[((axis.len() as f64 * 0.98) as usize).min(axis.len() - 1)];
    }
    Some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_component_keeps_one_cluster() {
        // Two well-separated cubic clusters; the larger one should survive.
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for i in 0..30 {
            let f = i as f64 * 0.05;
            pts.push([f, 0.0, 0.0]);
        }
        for i in 0..10 {
            let f = i as f64 * 0.05;
            pts.push([100.0 + f, 0.0, 0.0]);
        }
        let mask = dense::largest_component_mask(&pts, 0.2);
        assert_eq!(mask.iter().filter(|&&k| k).count(), 30);
    }

    #[test]
    fn crop_to_bounds_drops_outside_points() {
        use crate::dense::DenseCloud;
        let mut cloud = DenseCloud {
            points: (0..20).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect(),
            colors: vec![[0u8, 0, 0]; 20],
        };
        let removed = dense::crop_to_bounds(&mut cloud, [0.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0.0);
        assert_eq!(removed, 9); // points 11..19 are outside x<=1.0
        assert!(cloud.points.iter().all(|p| p[0] <= 1.0 + 1e-9));
    }
}
