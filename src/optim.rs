//! Adam optimizer.

use crate::net::Net;

pub struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: i32,
}

impl Adam {
    pub fn new(net: &Net) -> Self {
        Adam { m: net.zero_grads(), v: net.zero_grads(), t: 0 }
    }

    /// Applies one update. `grad_scale` turns summed per-example gradients into a mean.
    pub fn step(&mut self, params: &mut [Vec<f32>], grads: &[Vec<f32>], grad_scale: f32, lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t);
        let bc2 = 1.0 - B2.powi(self.t);
        for (((p, g), m), v) in params.iter_mut().zip(grads).zip(&mut self.m).zip(&mut self.v) {
            for (((p, &g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
                let g = g * grad_scale;
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                *p -= lr * (*m / bc1) / ((*v / bc2).sqrt() + 1e-8);
            }
        }
    }
}
