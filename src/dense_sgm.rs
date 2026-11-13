//! Census transform (Zabih & Woodfill 1994) and semi-global matching
//! (Hirschmüller 2008).
//!
//! NCC needs intensity *variance* to match anything; smooth, low-texture
//! surfaces defeat it. Census compares intensity *orderings*, which survive on
//! gentle shading. One trap: on a perfectly flat patch every comparison is a
//! tie, so the census code is constant and the Hamming cost is **zero
//! everywhere** — winner-takes-all then picks an arbitrary depth. SGM's
//! smoothness prior resolves that by propagating evidence from nearby texture,
//! which is why plain census WTA reconstructs flat tabletops as objects.

use kornia_image::Image;

/// Census code image (24 bits per pixel).
pub struct CensusMap {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Census codes, row-major.
    pub codes: Vec<u32>,
}

/// The 24 centre-symmetric comparison pairs of a 7×7 window: `(dx, dy)` picks
/// the positive representative; each bit compares it against its mirror.
fn pairs() -> Vec<(isize, isize)> {
    let mut out = Vec::with_capacity(24);
    for dy in -3isize..=3 {
        for dx in -3isize..=3 {
            if dx == 0 && dy == 0 {
                continue;
            }
            if dx > 0 || (dx == 0 && dy > 0) {
                out.push((dx, dy));
            }
        }
    }
    assert_eq!(out.len(), 24);
    out
}

/// Build a census code image from a grayscale frame.
pub fn compute_census(gray: &Image<u8, 1>) -> CensusMap {
    let (w, h) = (gray.width(), gray.height());
    let pairs = pairs();
    let src = gray.as_slice();
    let mut codes = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut code = 0u32;
            for (i, &(dx, dy)) in pairs.iter().enumerate() {
                let a = sample(src, w, h, x, y, dx, dy);
                let b = sample(src, w, h, x, y, -dx, -dy);
                if a > b {
                    code |= 1 << i;
                }
            }
            codes[y * w + x] = code;
        }
    }
    CensusMap {
        width: w,
        height: h,
        codes,
    }
}

/// Nearest-neighbour census sample; `None` outside the image.
#[inline]
pub fn census_at(map: &CensusMap, x: f64, y: f64) -> Option<u32> {
    let (xi, yi) = (x.floor() as isize, y.floor() as isize);
    if xi < 0 || yi < 0 || xi >= map.width as isize || yi >= map.height as isize {
        return None;
    }
    Some(map.codes[yi as usize * map.width + xi as usize])
}

/// Hamming distance between two census codes, in bits.
#[inline]
pub fn hamming(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones()
}

#[inline]
fn sample(src: &[u8], w: usize, h: usize, x: usize, y: usize, dx: isize, dy: isize) -> u8 {
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
        // Border replication keeps the comparison well-defined.
        let cx = nx.clamp(0, w as isize - 1) as usize;
        let cy = ny.clamp(0, h as isize - 1) as usize;
        return src[cy * w + cx];
    }
    src[ny as usize * w + nx as usize]
}

/// Semi-global matching aggregation over a cost volume.
///
/// `cost` is row-major with `planes` entries per pixel (costs in `[0, 1]`).
/// `out` receives the aggregated volume: each directional pass adds its
/// smoothed path cost to the raw cost, penalising disparity jumps with `p1`
/// and large jumps with `p2`. Four paths (L→R, T→B, R→L, B→T) are the default;
/// `paths` clamps to the four cardinal directions.
#[allow(clippy::too_many_arguments)]
pub fn sgm_aggregate(
    cost: &[f32],
    width: usize,
    height: usize,
    planes: usize,
    p1: f32,
    p2: f32,
    paths: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(cost.len(), width * height * planes);
    out.copy_from_slice(cost);
    let mut buf = vec![0.0f32; planes];

    for dir in 0..paths.min(4) {
        match dir {
            0 => {
                for y in 0..height {
                    buf.fill(0.0);
                    for x in 0..width {
                        accumulate(cost, width, planes, x, y, p1, p2, &mut buf, out);
                    }
                }
            }
            1 => {
                for x in 0..width {
                    buf.fill(0.0);
                    for y in 0..height {
                        accumulate(cost, width, planes, x, y, p1, p2, &mut buf, out);
                    }
                }
            }
            2 => {
                for y in 0..height {
                    buf.fill(0.0);
                    for x in (0..width).rev() {
                        accumulate(cost, width, planes, x, y, p1, p2, &mut buf, out);
                    }
                }
            }
            _ => {
                for x in 0..width {
                    buf.fill(0.0);
                    for y in (0..height).rev() {
                        accumulate(cost, width, planes, x, y, p1, p2, &mut buf, out);
                    }
                }
            }
        }
    }
}

/// One SGM step: smooth the cost of pixel `(x, y)` along the current path.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate(
    cost: &[f32],
    width: usize,
    planes: usize,
    x: usize,
    y: usize,
    p1: f32,
    p2: f32,
    prev: &mut [f32],
    out: &mut [f32],
) {
    let base = (y * width + x) * planes;
    let mut min_prev = f32::INFINITY;
    for &l in prev.iter() {
        if l < min_prev {
            min_prev = l;
        }
    }
    let jump = min_prev + p2;
    for d in 0..planes {
        let mut m = prev[d];
        if d > 0 {
            m = m.min(prev[d - 1] + p1);
        }
        if d + 1 < planes {
            m = m.min(prev[d + 1] + p1);
        }
        let l = cost[base + d] + m.min(jump) - min_prev;
        prev[d] = l;
        out[base + d] += l;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_image(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Image<u8, 1> {
        let mut img = Image::<u8, 1>::from_size_val(
            kornia_image::ImageSize {
                width: w,
                height: h,
            },
            0,
        )
        .unwrap();
        let s = img.as_slice_mut();
        for y in 0..h {
            for x in 0..w {
                s[y * w + x] = f(x, y);
            }
        }
        img
    }

    #[test]
    fn census_is_constant_on_a_ramp_and_codes_edges() {
        // A pure linear ramp: every comparison has the same outcome, so the
        // census code is identical everywhere — census cannot localise depth
        // on a ramp either; that is SGM's job (regularization), not census's.
        let ramp = compute_census(&gray_image(32, 32, |x, _| (x * 8) as u8));
        let first = ramp.codes[0];
        assert!(
            ramp.codes.iter().all(|&c| c == first),
            "ramp codes identical"
        );
        assert_eq!(hamming(first, first), 0);

        // A step edge changes the codes: a pixel whose window straddles it sees
        // mixed comparisons, unlike a pixel sitting in a flat region.
        let edge = compute_census(&gray_image(32, 32, |x, _| if x < 16 { 0 } else { 255 }));
        let flat_code = edge.codes[16 * 32 + 5];
        let edge_code = edge.codes[16 * 32 + 16];
        assert!(
            flat_code != edge_code,
            "the border-straddling pixel must code the edge"
        );
        assert!(hamming(flat_code, edge_code) > 0);
    }

    #[test]
    fn census_at_is_nearest_neighbour_and_bounded() {
        let map = compute_census(&gray_image(8, 8, |x, _| (x * 32) as u8));
        assert_eq!(census_at(&map, 0.1, 0.1), Some(map.codes[0]));
        assert_eq!(census_at(&map, 7.9, 7.9), Some(map.codes[63]));
        assert_eq!(census_at(&map, -1.0, 0.0), None);
        assert_eq!(census_at(&map, 8.0, 8.0), None);
    }

    #[test]
    fn sgm_propagates_disparity_through_a_flat_gap() {
        // One image row, 5 pixels, 4 disparities. The two textured ends agree
        // on disparity 1; the middle is flat (all disparities equally costly).
        // Plain WTA would pick disparity 0 at the middle; SGM must carry the
        // neighbouring evidence through.
        let (w, h, dmax) = (5usize, 1usize, 4usize);
        let mut cost = vec![0.0f32; w * h * dmax];
        fn textured(cost: &mut [f32], base: usize) {
            cost[base] = 0.5;
            cost[base + 1] = 0.0;
            cost[base + 2] = 0.5;
            cost[base + 3] = 0.5;
        }
        fn flat(cost: &mut [f32], base: usize) {
            for d in 0..4 {
                cost[base + d] = 0.4;
            }
        }
        textured(&mut cost, 0);
        flat(&mut cost, dmax);
        flat(&mut cost, 2 * dmax);
        flat(&mut cost, 3 * dmax);
        textured(&mut cost, 4 * dmax);

        let mut out = vec![0.0f32; cost.len()];
        sgm_aggregate(&cost, w, h, dmax, 0.05, 0.3, 4, &mut out);
        let argmin = |x: usize| {
            let base = x * dmax;
            (0..dmax)
                .min_by(|&a, &b| out[base + a].total_cmp(&out[base + b]))
                .unwrap()
        };
        assert_eq!(argmin(0), 1);
        assert_eq!(argmin(2), 1, "SGM must fill the flat gap");
        assert_eq!(argmin(4), 1);
    }
}
