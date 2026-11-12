//! Mesh simplification by quadric error metrics (Garland & Heckbert 1997).
//!
//! Every vertex accumulates a quadric `Q` that measures the squared distance
//! to all planes of its incident faces. Collapsing an edge costs the quadric
//! error at the optimal merged position, so the heap always removes the
//! cheapest shape change first. The link condition and a normal-flip check
//! keep the collapsed mesh manifold and fold-free.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use kornia_algebra::Vec3F64;

use crate::mesh::Mesh;

/// A plane quadric in the compact symmetric form
/// `[a00, a01, a02, a11, a12, a22, b0, b1, b2, c]`.
type Quadric = [f64; 10];

/// A collapse candidate.
struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
    version_a: u32,
    version_b: u32,
    position: Vec3F64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so `BinaryHeap` (a max-heap) yields the cheapest collapse.
        other.cost.total_cmp(&self.cost)
    }
}

/// Add two quadrics.
fn add_quadrics(a: &Quadric, b: &Quadric) -> Quadric {
    let mut out = [0.0; 10];
    for i in 0..10 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Plane quadric from a face: `[n nᵀ, d n; d nᵀ, d²]` with `n` normalized.
fn plane_quadric(p0: Vec3F64, p1: Vec3F64, p2: Vec3F64) -> Option<Quadric> {
    let n = (p1 - p0).cross(p2 - p0);
    let len = n.length();
    if len < 1e-14 {
        return None;
    }
    let n = n / len;
    let d = -n.dot(p0);
    Some([
        n.x * n.x,
        n.x * n.y,
        n.x * n.z,
        n.y * n.y,
        n.y * n.z,
        n.z * n.z,
        d * n.x,
        d * n.y,
        d * n.z,
        d * d,
    ])
}

/// Quadric error at `p`.
fn error(q: &Quadric, p: Vec3F64) -> f64 {
    let (x, y, z) = (p.x, p.y, p.z);
    q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + q[3] * y * y
        + 2.0 * q[4] * y * z
        + q[5] * z * z
        + 2.0 * q[6] * x
        + 2.0 * q[7] * y
        + 2.0 * q[8] * z
        + q[9]
}

/// Position minimizing `pᵀ A p + 2 bᵀ p + c`, or the midpoint when `A` is
/// (near) singular.
fn optimal_position(q: &Quadric, a: Vec3F64, b: Vec3F64) -> Vec3F64 {
    let m = kornia_algebra::Mat3F64::from_cols(
        Vec3F64::new(q[0], q[1], q[2]),
        Vec3F64::new(q[1], q[3], q[4]),
        Vec3F64::new(q[2], q[4], q[5]),
    );
    let rhs = -Vec3F64::new(q[6], q[7], q[8]);
    let det = m.determinant();
    if det.abs() > 1e-12 && det.is_finite() {
        let p = m.inverse() * rhs;
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            return p;
        }
    }
    (a + b) * 0.5
}

/// Vertex ring (unique neighbours) of `v` over living faces.
fn neighbours(
    v: u32,
    faces: &[[u32; 3]],
    alive: &[bool],
    vertex_faces: &[Vec<u32>],
) -> HashSet<u32> {
    let mut out = HashSet::new();
    for &f in &vertex_faces[v as usize] {
        if !alive[f as usize] {
            continue;
        }
        for &u in &faces[f as usize] {
            if u != v {
                out.insert(u);
            }
        }
    }
    out
}

/// Simplify `mesh` to at most `target_faces` triangles.
///
/// Returns a new mesh; the input is left untouched. If the mesh already has
/// few enough faces, it is cloned unchanged.
pub fn simplify(mesh: &Mesh, target_faces: usize) -> Mesh {
    if mesh.faces.len() <= target_faces {
        return Mesh {
            positions: mesh.positions.clone(),
            faces: mesh.faces.clone(),
        };
    }

    let n_vertices = mesh.positions.len();
    let mut positions = mesh.positions.clone();
    let mut faces = mesh.faces.clone();
    let mut face_alive = vec![true; faces.len()];
    let mut vertex_alive = vec![true; n_vertices];
    let mut version = vec![0u32; n_vertices];

    let mut vertex_faces: Vec<Vec<u32>> = vec![Vec::new(); n_vertices];
    for (fi, f) in faces.iter().enumerate() {
        for &v in f {
            vertex_faces[v as usize].push(fi as u32);
        }
    }

    // Vertex quadrics from incident face planes.
    let mut quadrics: Vec<Quadric> = vec![[0.0; 10]; n_vertices];
    for f in &faces {
        let (a, b, c) = (
            positions[f[0] as usize],
            positions[f[1] as usize],
            positions[f[2] as usize],
        );
        if let Some(q) = plane_quadric(a, b, c) {
            for &v in f {
                quadrics[v as usize] = add_quadrics(&quadrics[v as usize], &q);
            }
        }
    }

    // Boundary vertices are locked so open meshes keep their borders.
    let mut edge_use: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for f in &faces {
        for (u, v) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            let key = if u < v { (u, v) } else { (v, u) };
            *edge_use.entry(key).or_default() += 1;
        }
    }
    let mut locked = vec![false; n_vertices];
    for ((u, v), count) in &edge_use {
        if *count != 2 {
            locked[*u as usize] = true;
            locked[*v as usize] = true;
        }
    }

    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let push_edge = |a: u32,
                     b: u32,
                     positions: &Vec<Vec3F64>,
                     quadrics: &Vec<Quadric>,
                     version: &Vec<u32>,
                     heap: &mut BinaryHeap<Candidate>| {
        if a == b || locked[a as usize] || locked[b as usize] {
            return;
        }
        let (pa, pb) = (positions[a as usize], positions[b as usize]);
        let q = add_quadrics(&quadrics[a as usize], &quadrics[b as usize]);
        let p = optimal_position(&q, pa, pb);
        heap.push(Candidate {
            cost: error(&q, p).max(0.0),
            a,
            b,
            version_a: version[a as usize],
            version_b: version[b as usize],
            position: p,
        });
    };

    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();
    for f in &faces {
        for (u, v) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            let key = if u < v { (u, v) } else { (v, u) };
            if seen_edges.insert(key) {
                push_edge(key.0, key.1, &positions, &quadrics, &version, &mut heap);
            }
        }
    }

    let mut living_faces = faces.len();
    let mut guard = 0usize;
    let max_steps = faces.len() * 4 + 1024;

    while living_faces > target_faces {
        guard += 1;
        if guard > max_steps {
            break;
        }
        let Some(cand) = heap.pop() else { break };
        let (a, b) = (cand.a, cand.b);
        if !vertex_alive[a as usize]
            || !vertex_alive[b as usize]
            || version[a as usize] != cand.version_a
            || version[b as usize] != cand.version_b
        {
            continue; // stale candidate
        }

        let shared: Vec<u32> = vertex_faces[a as usize]
            .iter()
            .chain(vertex_faces[b as usize].iter())
            .copied()
            .filter(|&f| {
                face_alive[f as usize]
                    && faces[f as usize].contains(&a)
                    && faces[f as usize].contains(&b)
            })
            .collect();
        let mut unique_shared = shared.clone();
        unique_shared.sort_unstable();
        unique_shared.dedup();
        if unique_shared.is_empty() || unique_shared.len() > 2 {
            continue;
        }

        // Link condition: the common neighbours of a and b must be exactly the
        // opposite vertices of their shared faces, else the collapse pinches.
        let common = {
            let mut c = neighbours(a, &faces, &face_alive, &vertex_faces);
            c.retain(|v| neighbours(b, &faces, &face_alive, &vertex_faces).contains(v));
            c
        };
        let mut opposite: HashSet<u32> = HashSet::new();
        for &f in &unique_shared {
            for &v in &faces[f as usize] {
                if v != a && v != b {
                    opposite.insert(v);
                }
            }
        }
        if common != opposite {
            continue;
        }

        // Reject collapses that would flip any surviving face.
        let new_pos = cand.position;
        let affected: HashSet<u32> = vertex_faces[a as usize]
            .iter()
            .chain(vertex_faces[b as usize].iter())
            .copied()
            .filter(|&f| face_alive[f as usize])
            .collect();
        let mut flips = false;
        for &fi in &affected {
            let f = faces[fi as usize];
            if f.contains(&a) && f.contains(&b) {
                continue; // removed below
            }
            let old = {
                let (p0, p1, p2) = (
                    positions[f[0] as usize],
                    positions[f[1] as usize],
                    positions[f[2] as usize],
                );
                (p1 - p0).cross(p2 - p0)
            };
            let new = {
                let p = |v: u32| {
                    if v == a || v == b {
                        new_pos
                    } else {
                        positions[v as usize]
                    }
                };
                let (p0, p1, p2) = (p(f[0]), p(f[1]), p(f[2]));
                (p1 - p0).cross(p2 - p0)
            };
            let old_len = old.length();
            let new_len = new.length();
            if new_len < 1e-14 {
                flips = true;
                break;
            }
            if old_len > 0.0 && old.dot(new) / (old_len * new_len) < 0.2 {
                flips = true;
                break;
            }
        }
        if flips {
            continue;
        }

        // Apply the collapse: b's non-shared faces move onto a.
        positions[a as usize] = new_pos;
        quadrics[a as usize] = add_quadrics(&quadrics[a as usize], &quadrics[b as usize]);
        for &fi in &unique_shared {
            face_alive[fi as usize] = false;
            living_faces -= 1;
        }
        let b_faces = std::mem::take(&mut vertex_faces[b as usize]);
        for fi in b_faces {
            if !face_alive[fi as usize] {
                continue;
            }
            let f = &mut faces[fi as usize];
            for v in f.iter_mut() {
                if *v == b {
                    *v = a;
                }
            }
            vertex_faces[a as usize].push(fi);
        }
        vertex_alive[b as usize] = false;
        version[a as usize] += 1;
        version[b as usize] += 1;

        // Re-push edges around the moved vertex. Sorting makes the push order
        // deterministic: `HashSet` iteration order varies per run and would
        // otherwise decide how equal-cost candidates are popped.
        let mut ring: Vec<u32> = neighbours(a, &faces, &face_alive, &vertex_faces)
            .into_iter()
            .collect();
        ring.sort_unstable();
        for v in ring {
            push_edge(a, v, &positions, &quadrics, &version, &mut heap);
        }
    }

    // Compact vertices and faces.
    let mut remap = vec![u32::MAX; n_vertices];
    let mut out_positions = Vec::new();
    for (v, alive) in vertex_alive.iter().enumerate() {
        if *alive {
            remap[v] = out_positions.len() as u32;
            out_positions.push(positions[v]);
        }
    }
    let mut out_faces = Vec::new();
    for (fi, f) in faces.iter().enumerate() {
        if !face_alive[fi] {
            continue;
        }
        let f = [
            remap[f[0] as usize],
            remap[f[1] as usize],
            remap[f[2] as usize],
        ];
        if f[0] != f[1] && f[1] != f[2] && f[0] != f[2] {
            out_faces.push(f);
        }
    }
    Mesh {
        positions: out_positions,
        faces: out_faces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::marching_tetrahedra;
    use crate::tsdf::Volume;

    fn sphere_mesh(n: usize, voxel: f64, radius: f64) -> Mesh {
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
                    if d.abs() <= 4.0 * voxel {
                        sdf[idx] = (d / (4.0 * voxel)).clamp(-1.0, 1.0) as f32;
                        weight[idx] = 1.0;
                    }
                }
            }
        }
        marching_tetrahedra(&Volume {
            origin,
            voxel_size: voxel,
            trunc: 4.0 * voxel,
            dims: [n, n, n],
            sdf,
            weight,
        })
    }

    #[test]
    fn simplification_reaches_the_target_and_stays_closed() {
        let mesh = sphere_mesh(40, 0.1, 1.23);
        let before = mesh.face_count();
        let simplified = simplify(&mesh, before / 4);
        assert!(
            simplified.face_count() <= before / 4 + 2,
            "{} -> {}",
            before,
            simplified.face_count()
        );
        assert!(simplified.face_count() > 0);
        assert!(simplified.is_closed(), "simplified sphere must stay closed");
        assert_eq!(simplified.euler_characteristic(), 2);
        // Surviving vertices stay on the sphere (no folds far off it).
        for p in &simplified.positions {
            assert!(
                (p.length() - 1.23).abs() < 0.15,
                "simplified vertex left the surface: {}",
                p.length()
            );
        }
    }

    #[test]
    fn no_op_when_already_below_target() {
        let mesh = sphere_mesh(24, 0.12, 1.0);
        let simplified = simplify(&mesh, mesh.face_count() + 100);
        assert_eq!(simplified.face_count(), mesh.face_count());
        assert_eq!(simplified.vertex_count(), mesh.vertex_count());
    }

    #[test]
    fn simplification_is_deterministic() {
        let mesh = sphere_mesh(32, 0.1, 1.0);
        let target = mesh.face_count() / 3;
        let a = simplify(&mesh, target);
        let b = simplify(&mesh, target);
        assert_eq!(a.face_count(), b.face_count());
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(&b.positions) {
            assert_eq!(pa.to_array(), pb.to_array());
        }
        assert_eq!(a.faces, b.faces);
    }
}
