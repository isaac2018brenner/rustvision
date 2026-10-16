//! rustvision: a convolutional neural network built from scratch in pure Rust.

pub mod augment;
pub mod checkpoint;
pub mod mnist;
pub mod net;
pub mod optim;
pub mod parallel;
pub mod rng;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Dot product with 8 independent accumulators so the compiler can vectorize it.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}
