//! Triangle mesh extraction (marching tetrahedra) and vertex attributes.
//!
//! Each cube of the TSDF grid is split into six tetrahedra around one body
//! diagonal. A tetrahedron has only three non-trivial zero-crossing patterns,
//! so the surface needs no large lookup tables and every triangle can be
//! checked against the field gradient for outward orientation.

use std::collections::HashMap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::Image;

use crate::dense::sample_rgb_bilinear;
use crate::tsdf::Volume;

/// A triangle mesh with per-vertex positions.
#[derive(Debug, Default)]
pub struct Mesh {
    /// Vertex positions in the world frame.
    pub positions: Vec<Vec3F64>,
    /// Triangle vertex indices.
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    /// Number of triangles.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Geometric normal of face `f` (not normalized).
    #[allow(dead_code)] // used by the surface tests and TAI metrics
    pub fn face_normal(&self, f: usize) -> Vec3F64 {
        let [a, b, c] = self.faces[f];
        let pa = self.positions[a as usize];
        (self.positions[b as usize] - pa).cross(self.positions[c as usize] - pa)
    }

    /// Whether every edge is shared by exactly two faces (a closed surface).
    pub fn is_closed(&self) -> bool {
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for [a, b, c] in &self.faces {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (*u, *v) } else { (*v, *u) };
                *edges.entry(key).or_default() += 1;
            }
        }
        !edges.is_empty() && edges.values().all(|&count| count == 2)
    }

    /// Euler characteristic `V - E + F` (2 for a closed genus-0 surface).
    pub fn euler_characteristic(&self) -> i64 {
        let mut edges: HashMap<(u32, u32), ()> = HashMap::new();
        for [a, b, c] in &self.faces {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (*u, *v) } else { (*v, *u) };
                edges.insert(key, ());
            }
        }
        self.positions.len() as i64 - edges.len() as i64 + self.faces.len() as i64
    }

    /// Number of connected surface patches (union-find over shared vertices).
    pub fn connected_components(&self) -> usize {
        self.component_sizes().len()
    }

    /// Number of faces in each connected patch.
    pub fn component_sizes(&self) -> Vec<usize> {
        let mut parent: Vec<u32> = (0..self.positions.len() as u32).collect();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        for f in &self.faces {
            for (u, v) in [(f[0], f[1]), (f[1], f[2])] {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv {
                    parent[ru as usize] = rv;
                }
            }
        }
        let mut sizes: HashMap<u32, usize> = HashMap::new();
        for f in &self.faces {
            *sizes.entry(find(&mut parent, f[0])).or_default() += 1;
        }
        sizes.into_values().collect()
    }

    /// Drop connected patches with fewer than `min_faces` triangles.
    ///
    /// Photogrammetry routinely produces small floating fragments around a
    /// scanned object (specular highlights, moving background, noisy depth).
    /// Keeping only substantial patches turns the triangle soup into a solid
    /// object surface without changing the surviving geometry.
    pub fn remove_small_components(&self, min_faces: usize) -> Mesh {
        if min_faces == 0 {
            return Mesh {
                positions: self.positions.clone(),
                faces: self.faces.clone(),
            };
        }
        let mut parent: Vec<u32> = (0..self.positions.len() as u32).collect();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        for f in &self.faces {
            for (u, v) in [(f[0], f[1]), (f[1], f[2])] {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv {
                    parent[ru as usize] = rv;
                }
            }
        }
        let mut sizes: HashMap<u32, usize> = HashMap::new();
        for f in &self.faces {
            *sizes.entry(find(&mut parent, f[0])).or_default() += 1;
        }
        let keep: std::collections::HashSet<u32> = sizes
            .into_iter()
            .filter_map(|(root, n)| (n >= min_faces).then_some(root))
            .collect();
        let mut used = vec![false; self.positions.len()];
        for f in &self.faces {
            if keep.contains(&find(&mut parent, f[0])) {
                for &v in f {
                    used[v as usize] = true;
                }
            }
        }
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut positions = Vec::new();
        for (v, &u) in used.iter().enumerate() {
            if u {
                remap[v] = positions.len() as u32;
                positions.push(self.positions[v]);
            }
        }
        let mut faces = Vec::new();
        for f in &self.faces {
            if keep.contains(&find(&mut parent, f[0])) {
                faces.push([
                    remap[f[0] as usize],
                    remap[f[1] as usize],
                    remap[f[2] as usize],
                ]);
            }
        }
        Mesh { positions, faces }
    }
}

/// The six tetrahedra of a cube, all sharing the body diagonal 0-7.
///
/// Corner order matches the gather loop's bit fields:
/// 0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(1,1,0) 4=(0,0,1) 5=(1,0,1) 6=(0,1,1) 7=(1,1,1).
/// Each face is split by the same diagonal in the cube next to it, which is
/// what keeps the extracted surface watertight across cells.
const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// Extract the zero level set of `volume` as a triangle mesh.
///
/// Unobserved grid points count as empty (`+1`), and a cell is only meshed when
/// it contains at least one observed corner and one inside corner — so no
/// surface is invented in regions the cameras never saw.
pub fn marching_tetrahedra(volume: &Volume) -> Mesh {
    let (nx, ny, nz) = (volume.dims[0], volume.dims[1], volume.dims[2]);
    let mut mesh = Mesh::default();
    // Weld vertices that land on the same edge from adjacent cells. The key
    // quantizes world positions to a thousandth of a voxel.
    let eps = (volume.voxel_size * 1e-3).max(1e-12);
    let mut vertex_ids: HashMap<(i64, i64, i64), u32> = HashMap::new();

    let mut corner = [(0.0f32, Vec3F64::ZERO); 8];

    for i in 0..nx.saturating_sub(1) {
        for j in 0..ny.saturating_sub(1) {
            for k in 0..nz.saturating_sub(1) {
                // Gather the eight corners of this cell. A cell is meshed only
                // when every corner is observed: an unobserved corner has no
                // reliable sign, and guessing it manufactures surfaces at the
                // edge of the truncation band (e.g. inside the object).
                let mut all_observed = true;
                let mut inside = false;
                for (c, entry) in corner.iter_mut().enumerate() {
                    let (di, dj, dk) = ((c & 1), ((c >> 1) & 1), ((c >> 2) & 1));
                    let idx = volume.index(i + di, j + dj, k + dk);
                    let w = volume.weight[idx];
                    all_observed &= w > 0.0;
                    let value = volume.sdf[idx];
                    inside |= value < 0.0;
                    *entry = (value, volume.position(i + di, j + dj, k + dk));
                }
                if !all_observed || !inside {
                    continue;
                }

                for tet in &TETS {
                    emit_tet(&corner, tet, volume, eps, &mut vertex_ids, &mut mesh);
                }
            }
        }
    }

    mesh
}

/// Emit the zero-crossing triangles of one tetrahedron.
fn emit_tet(
    corner: &[(f32, Vec3F64); 8],
    tet: &[usize; 4],
    volume: &Volume,
    eps: f64,
    vertex_ids: &mut HashMap<(i64, i64, i64), u32>,
    mesh: &mut Mesh,
) {
    let values = [
        corner[tet[0]].0,
        corner[tet[1]].0,
        corner[tet[2]].0,
        corner[tet[3]].0,
    ];
    let inside_count = values.iter().filter(|v| **v < 0.0).count();
    if inside_count == 0 || inside_count == 4 {
        return;
    }

    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut crossing: Vec<((usize, usize), Vec3F64)> = Vec::with_capacity(4);
    for (a, b) in EDGES {
        let (va, vb) = (values[a], values[b]);
        if (va < 0.0) == (vb < 0.0) {
            continue;
        }
        let denom = va - vb;
        if denom.abs() < 1e-12 {
            continue;
        }
        let t = (va / denom) as f64;
        let pa = corner[tet[a]].1;
        let pb = corner[tet[b]].1;
        crossing.push(((a, b), pa + (pb - pa) * t));
    }

    match crossing.len() {
        3 => push_triangle(
            &crossing[0].1,
            &crossing[1].1,
            &crossing[2].1,
            volume,
            eps,
            vertex_ids,
            mesh,
        ),
        4 => {
            // Two inside, two outside: the crossings form a quad whose cycle
            // follows the four inside-outside edges. Emitting them in edge-list
            // order instead would bowtie the quad and leave the surface open.
            let inside: Vec<usize> = (0..4).filter(|&i| values[i] < 0.0).collect();
            let outside: Vec<usize> = (0..4).filter(|&i| values[i] >= 0.0).collect();
            if inside.len() != 2 || outside.len() != 2 {
                return;
            }
            let at = |a: usize, b: usize| -> Option<Vec3F64> {
                crossing
                    .iter()
                    .find(|((u, v), _)| (*u == a && *v == b) || (*u == b && *v == a))
                    .map(|(_, p)| *p)
            };
            let (i0, i1) = (inside[0], inside[1]);
            let (o0, o1) = (outside[0], outside[1]);
            let (Some(q0), Some(q1), Some(q2), Some(q3)) =
                (at(i0, o0), at(i1, o0), at(i1, o1), at(i0, o1))
            else {
                return;
            };
            push_triangle(&q0, &q1, &q2, volume, eps, vertex_ids, mesh);
            push_triangle(&q0, &q2, &q3, volume, eps, vertex_ids, mesh);
        }
        _ => {}
    }
}

/// Add a triangle, welding vertices by quantized position and orienting the
/// face normal along the field gradient (outward).
fn push_triangle(
    a: &Vec3F64,
    b: &Vec3F64,
    c: &Vec3F64,
    volume: &Volume,
    eps: f64,
    vertex_ids: &mut HashMap<(i64, i64, i64), u32>,
    mesh: &mut Mesh,
) {
    let normal = (*b - *a).cross(*c - *a);
    if normal.length() < 1e-14 {
        return;
    }
    let centroid = (*a + *b + *c) / 3.0;
    let gradient = volume.gradient(centroid);
    let (b, c) = if normal.dot(gradient) < 0.0 {
        (c, b)
    } else {
        (b, c)
    };

    let ia = vertex_id(a, eps, vertex_ids, mesh);
    let ib = vertex_id(b, eps, vertex_ids, mesh);
    let ic = vertex_id(c, eps, vertex_ids, mesh);
    if ia != ib && ib != ic && ia != ic {
        mesh.faces.push([ia, ib, ic]);
    }
}

/// Look up or insert a welded vertex.
fn vertex_id(
    p: &Vec3F64,
    eps: f64,
    vertex_ids: &mut HashMap<(i64, i64, i64), u32>,
    mesh: &mut Mesh,
) -> u32 {
    let key = (
        (p.x / eps).round() as i64,
        (p.y / eps).round() as i64,
        (p.z / eps).round() as i64,
    );
    if let Some(&id) = vertex_ids.get(&key) {
        return id;
    }
    let id = mesh.positions.len() as u32;
    mesh.positions.push(*p);
    vertex_ids.insert(key, id);
    id
}

/// Vertex normals from the TSDF gradient (outward-facing).
pub fn vertex_normals(mesh: &Mesh, volume: &Volume) -> Vec<Vec3F64> {
    mesh.positions.iter().map(|p| volume.gradient(*p)).collect()
}

/// Per-vertex colours sampled from the source frames.
///
/// Each view that sees the vertex votes, weighted by how squarely it faces
/// that view (`cos²` of the angle between the normal and the viewing ray), so
/// grazing views cannot smear the colour.
pub fn vertex_colors(
    mesh: &Mesh,
    normals: &[Vec3F64],
    cameras: &[PinholeCamera],
    views: &[Option<Pose3d>],
    rgb_frames: &[Image<u8, 3>],
) -> Vec<[u8; 3]> {
    mesh.positions
        .iter()
        .zip(normals)
        .map(|(p, normal)| {
            let mut sum = [0.0f64; 3];
            let mut weight_sum = 0.0f64;
            for (v, pose) in views.iter().enumerate() {
                let Some(pose) = pose else { continue };
                let w2c = pose.inverse();
                let p_cam = w2c.transform_point(p);
                let Some(pix) = cameras[v].project_to_pixel(&p_cam, 1e-9) else {
                    continue;
                };
                let rgb = &rgb_frames[v];
                let Some(c) =
                    sample_rgb_bilinear(rgb.as_slice(), rgb.width(), rgb.height(), pix.x, pix.y)
                else {
                    continue;
                };
                let dir = (pose.translation - *p).normalize();
                let facing = normal.dot(dir).max(0.0);
                let w = facing * facing;
                if w <= 0.0 {
                    continue;
                }
                for channel in 0..3 {
                    sum[channel] += c[channel] as f64 * w;
                }
                weight_sum += w;
            }
            if weight_sum > 0.0 {
                [
                    (sum[0] / weight_sum).round().clamp(0.0, 255.0) as u8,
                    (sum[1] / weight_sum).round().clamp(0.0, 255.0) as u8,
                    (sum[2] / weight_sum).round().clamp(0.0, 255.0) as u8,
                ]
            } else {
                [180, 180, 180]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere sampled on a regular grid: the exact TSDF of a ball.
    fn sphere_volume(n: usize, voxel: f64, radius: f64, trunc: f64) -> Volume {
        let dims = [n, n, n];
        let origin = Vec3F64::new(
            -(n as f64 - 1.0) * voxel / 2.0,
            -(n as f64 - 1.0) * voxel / 2.0,
            -(n as f64 - 1.0) * voxel / 2.0,
        );
        let mut sdf = vec![0.0f32; n * n * n];
        let mut weight = vec![0.0f32; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = origin + Vec3F64::new(i as f64, j as f64, k as f64) * voxel;
                    let d = p.length() - radius;
                    let idx = (i * n + j) * n + k;
                    if d.abs() <= trunc {
                        sdf[idx] = (d / trunc).clamp(-1.0, 1.0) as f32;
                        weight[idx] = 1.0;
                    }
                }
            }
        }
        Volume {
            origin,
            voxel_size: voxel,
            dims,
            sdf,
            weight,
        }
    }

    #[test]
    fn sphere_surface_is_closed_and_lies_on_the_sphere() {
        let voxel = 0.1;
        let radius = 1.23;
        let vol = sphere_volume(40, voxel, radius, 4.0 * voxel);
        let mesh = marching_tetrahedra(&vol);

        assert!(mesh.face_count() > 100, "faces = {}", mesh.face_count());
        for p in &mesh.positions {
            let r = p.length();
            assert!(
                (r - radius).abs() < 2.0 * voxel,
                "vertex radius {r} vs {radius}"
            );
        }
        assert!(mesh.is_closed(), "sphere surface must be closed");
        assert_eq!(mesh.euler_characteristic(), 2, "genus-0 surface");

        // Every face must point away from the centre (the field's gradient).
        for f in 0..mesh.face_count() {
            let [a, b, c] = mesh.faces[f];
            let centroid = (mesh.positions[a as usize]
                + mesh.positions[b as usize]
                + mesh.positions[c as usize])
                / 3.0;
            assert!(
                mesh.face_normal(f).dot(centroid) > 0.0,
                "face {f} points inward"
            );
        }
    }

    #[test]
    fn empty_volume_yields_no_mesh() {
        let vol = Volume {
            origin: Vec3F64::ZERO,
            voxel_size: 1.0,
            dims: [4, 4, 4],
            sdf: vec![1.0; 64],
            weight: vec![0.0; 64],
        };
        let mesh = marching_tetrahedra(&vol);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn vertex_normals_point_outward_on_a_sphere() {
        let vol = sphere_volume(24, 0.12, 1.0, 0.48);
        let mesh = marching_tetrahedra(&vol);
        let normals = vertex_normals(&mesh, &vol);
        for (p, n) in mesh.positions.iter().zip(&normals) {
            assert!(p.dot(*n) > 0.0, "normal points inward at {p:?}");
        }
    }

    #[test]
    fn small_components_are_removed() {
        let mesh = Mesh {
            positions: vec![
                Vec3F64::new(0.0, 0.0, 0.0),
                Vec3F64::new(1.0, 0.0, 0.0),
                Vec3F64::new(1.0, 1.0, 0.0),
                Vec3F64::new(0.0, 1.0, 0.0),
                Vec3F64::new(5.0, 5.0, 5.0),
                Vec3F64::new(6.0, 5.0, 5.0),
                Vec3F64::new(5.0, 6.0, 5.0),
            ],
            faces: vec![[0, 1, 2], [0, 2, 3], [4, 5, 6]],
        };
        assert_eq!(mesh.connected_components(), 2);
        let cleaned = mesh.remove_small_components(2);
        assert_eq!(cleaned.face_count(), 2, "the lone triangle is dropped");
        assert_eq!(cleaned.vertex_count(), 4, "its vertices are compacted away");
        assert_eq!(cleaned.connected_components(), 1);
    }
}
