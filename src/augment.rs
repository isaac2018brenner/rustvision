//! Random distortions applied to training images, so the network learns digits
//! that are rotated, resized, shifted, slanted or drawn with a thicker or thinner
//! pen, like the ones people draw in the browser demo.

use crate::mnist::SIDE;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Affine {
    pub rotate: f32, // radians
    pub scale: f32,
    pub shear: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { rotate: 0.0, scale: 1.0, shear: 0.0, dx: 0.0, dy: 0.0 };

    pub fn random(rng: &mut Rng) -> Self {
        let mut r = |lo: f32, hi: f32| lo + (hi - lo) * rng.uniform();
        Affine {
            rotate: r(-15.0, 15.0).to_radians(),
            scale: r(0.85, 1.15),
            shear: r(-0.25, 0.25),
            dx: r(-2.5, 2.5),
            dy: r(-2.5, 2.5),
        }
    }
}

/// Warps `src` by `t` around the image centre with bilinear sampling.
pub fn warp(src: &[f32], t: Affine) -> Vec<f32> {
    // Forward map: p' = R * Sh * S * (p - c) + c + d. We sample with its inverse.
    let (sin, cos) = t.rotate.sin_cos();
    let (a, b, c, d) = (cos * t.scale, (cos * t.shear - sin) * t.scale, sin * t.scale, (sin * t.shear + cos) * t.scale);
    let det = a * d - b * c;
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    let centre = (SIDE as f32 - 1.0) / 2.0;
    let px = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= SIDE as i32 || y >= SIDE as i32 {
            0.0
        } else {
            src[y as usize * SIDE + x as usize]
        }
    };
    let mut out = vec![0.0; SIDE * SIDE];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let (u, v) = (x as f32 - centre - t.dx, y as f32 - centre - t.dy);
            let (sx, sy) = (ia * u + ib * v + centre, ic * u + id * v + centre);
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as i32, y0 as i32);
            out[y * SIDE + x] = (1.0 - fy) * ((1.0 - fx) * px(x0, y0) + fx * px(x0 + 1, y0))
                + fy * ((1.0 - fx) * px(x0, y0 + 1) + fx * px(x0 + 1, y0 + 1));
        }
    }
    out
}

/// Thickens (`grow = true`) or thins strokes by taking a soft max/min over each pixel's 4-neighbourhood.
pub fn morph(src: &[f32], grow: bool, strength: f32) -> Vec<f32> {
    let n = SIDE as i32;
    let at = |x: i32, y: i32| if x < 0 || y < 0 || x >= n || y >= n { 0.0 } else { src[(y * n + x) as usize] };
    let mut out = vec![0.0; SIDE * SIDE];
    for y in 0..n {
        for x in 0..n {
            let nb = [at(x - 1, y), at(x + 1, y), at(x, y - 1), at(x, y + 1)];
            let v = at(x, y);
            let m = if grow { nb.iter().fold(v, |a, &b| a.max(b)) } else { nb.iter().fold(v, |a, &b| a.min(b)) };
            out[(y * n + x) as usize] = v + strength * (m - v);
        }
    }
    out
}

pub fn augment(src: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut img = warp(src, Affine::random(rng));
    let r = rng.uniform();
    if r < 0.2 {
        img = morph(&img, true, 0.4 + 0.6 * rng.uniform());
    } else if r < 0.35 {
        img = morph(&img, false, 0.3 + 0.4 * rng.uniform());
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_warp_is_a_no_op() {
        let mut rng = Rng::new(1);
        let img: Vec<f32> = (0..SIDE * SIDE).map(|_| rng.uniform()).collect();
        let out = warp(&img, Affine::IDENTITY);
        assert!(img.iter().zip(&out).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn shift_moves_pixels() {
        let mut img = vec![0.0; SIDE * SIDE];
        img[10 * SIDE + 10] = 1.0;
        let out = warp(&img, Affine { dx: 3.0, dy: -2.0, ..Affine::IDENTITY });
        assert!((out[8 * SIDE + 13] - 1.0).abs() < 1e-5);
    }
}
