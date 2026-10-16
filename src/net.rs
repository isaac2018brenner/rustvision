//! A small sequential CNN with hand-written forward and backward passes.
//!
//! Tensors are flat `Vec<f32>` in channel-major `[C, H, W]` order. Each layer
//! owns one flat parameter vector (weights followed by biases), so gradients and
//! optimizer state are just `Vec<Vec<f32>>` with the same shapes.

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Shape {
    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Stride-1 convolution with a square `k x k` kernel and zero padding `pad`.
    Conv {
        out_c: usize,
        k: usize,
        pad: usize,
    },
    /// 2x2 max pooling with stride 2.
    MaxPool,
    Relu,
    /// Fully connected layer; flattens its input.
    Dense {
        out: usize,
    },
}

pub struct Net {
    pub layers: Vec<Layer>,
    /// `shapes[i]` is the input shape of layer `i`; the last entry is the output shape.
    pub shapes: Vec<Shape>,
    pub params: Vec<Vec<f32>>,
}

/// Activations from one forward pass: `acts[0]` is the input, `acts[i + 1]` is layer `i`'s output.
pub type Acts = Vec<Vec<f32>>;

impl Net {
    pub fn new(input: Shape, layers: Vec<Layer>) -> Self {
        let mut shapes = vec![input];
        for layer in &layers {
            let s = *shapes.last().unwrap();
            shapes.push(match *layer {
                Layer::Conv { out_c, k, pad } => {
                    assert!(s.h + 2 * pad >= k && s.w + 2 * pad >= k, "kernel larger than input");
                    Shape { c: out_c, h: s.h + 2 * pad - k + 1, w: s.w + 2 * pad - k + 1 }
                }
                Layer::MaxPool => Shape { c: s.c, h: s.h / 2, w: s.w / 2 },
                Layer::Relu => s,
                Layer::Dense { out } => Shape { c: out, h: 1, w: 1 },
            });
        }
        let params = layers.iter().zip(&shapes).map(|(l, s)| vec![0.0; param_count(l, s)]).collect();
        Net { layers, shapes, params }
    }

    /// The network used for MNIST: two conv blocks and two dense layers (~207k parameters).
    pub fn mnist() -> Self {
        use Layer::*;
        Net::new(
            Shape { c: 1, h: 28, w: 28 },
            vec![
                Conv { out_c: 16, k: 5, pad: 2 },
                Relu,
                MaxPool,
                Conv { out_c: 32, k: 3, pad: 1 },
                Relu,
                MaxPool,
                Dense { out: 128 },
                Relu,
                Dense { out: 10 },
            ],
        )
    }

    /// He initialization for weights, zero biases.
    pub fn init(&mut self, rng: &mut Rng) {
        for ((layer, s), p) in self.layers.iter().zip(&self.shapes).zip(&mut self.params) {
            let (fan_in, n_weights) = match *layer {
                Layer::Conv { out_c, k, .. } => (s.c * k * k, out_c * s.c * k * k),
                Layer::Dense { out } => (s.len(), out * s.len()),
                _ => continue,
            };
            let std = (2.0 / fan_in as f32).sqrt();
            p[..n_weights].iter_mut().for_each(|w| *w = std * rng.normal());
            p[n_weights..].fill(0.0);
        }
    }

    pub fn num_params(&self) -> usize {
        self.params.iter().map(|p| p.len()).sum()
    }

    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        self.params.iter().map(|p| vec![0.0; p.len()]).collect()
    }

    pub fn forward(&self, input: &[f32]) -> Acts {
        assert_eq!(input.len(), self.shapes[0].len());
        let mut acts = Vec::with_capacity(self.layers.len() + 1);
        acts.push(input.to_vec());
        for (i, layer) in self.layers.iter().enumerate() {
            let (s, o) = (self.shapes[i], self.shapes[i + 1]);
            let x = &acts[i];
            let mut y = vec![0.0; o.len()];
            match *layer {
                Layer::Conv { out_c, k, pad } => conv_forward(&mut y, x, &self.params[i], s, out_c, k, pad),
                Layer::MaxPool => maxpool_forward(&mut y, x, s),
                Layer::Relu => y.iter_mut().zip(x).for_each(|(y, &x)| *y = x.max(0.0)),
                Layer::Dense { out } => dense_forward(&mut y, x, &self.params[i], out),
            }
            acts.push(y);
        }
        acts
    }

    /// Backpropagates `dout` (the gradient w.r.t. the network output) through the
    /// activations of one forward pass. Parameter gradients are added into `grads`;
    /// the gradient w.r.t. the input image is returned.
    pub fn backward(&self, acts: &Acts, dout: &[f32], grads: &mut [Vec<f32>]) -> Vec<f32> {
        let mut d = dout.to_vec();
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let s = self.shapes[i];
            let x = &acts[i];
            let mut dx = vec![0.0; s.len()];
            match *layer {
                Layer::Conv { out_c, k, pad } => {
                    conv_backward(&mut dx, &mut grads[i], &d, x, &self.params[i], s, out_c, k, pad)
                }
                Layer::MaxPool => maxpool_backward(&mut dx, &d, x, s),
                Layer::Relu => {
                    dx.iter_mut().zip(&d).zip(x).for_each(|((dx, &d), &x)| *dx = if x > 0.0 { d } else { 0.0 })
                }
                Layer::Dense { out } => dense_backward(&mut dx, &mut grads[i], &d, x, &self.params[i], out),
            }
            d = dx;
        }
        d
    }

    /// Forward + backward for one labelled example. Returns (loss, predicted class).
    pub fn train_step(&self, input: &[f32], label: usize, grads: &mut [Vec<f32>]) -> (f32, usize) {
        let acts = self.forward(input);
        let probs = softmax(acts.last().unwrap());
        let loss = -probs[label].max(1e-12).ln();
        let mut dlogits = probs.clone();
        dlogits[label] -= 1.0;
        self.backward(&acts, &dlogits, grads);
        (loss, argmax(&probs))
    }

    /// Gradient of one class's logit w.r.t. the input pixels: which pixels push the score up.
    pub fn saliency(&self, input: &[f32], class: usize) -> Vec<f32> {
        let acts = self.forward(input);
        let mut dlogits = vec![0.0; acts.last().unwrap().len()];
        dlogits[class] = 1.0;
        let mut scratch = self.zero_grads();
        self.backward(&acts, &dlogits, &mut scratch)
    }
}

fn param_count(layer: &Layer, s: &Shape) -> usize {
    match *layer {
        Layer::Conv { out_c, k, .. } => out_c * s.c * k * k + out_c,
        Layer::Dense { out } => out * s.len() + out,
        _ => 0,
    }
}

/// For kernel offset `kk`, the range of output positions whose input position `o + kk - pad` is in bounds.
#[inline]
fn valid_range(kk: usize, pad: usize, in_len: usize, out_len: usize) -> (usize, usize) {
    let lo = pad.saturating_sub(kk);
    let hi = (in_len + pad).saturating_sub(kk).min(out_len);
    (lo, hi.max(lo))
}

/// `y[oc] = b[oc] + sum_ic w[oc, ic] * x[ic]`, written as row-wise axpys so the inner loop vectorizes.
fn conv_forward(y: &mut [f32], x: &[f32], p: &[f32], s: Shape, out_c: usize, k: usize, pad: usize) {
    let (oh, ow) = (s.h + 2 * pad - k + 1, s.w + 2 * pad - k + 1);
    let (w, b) = p.split_at(out_c * s.c * k * k);
    for oc in 0..out_c {
        let yc = &mut y[oc * oh * ow..(oc + 1) * oh * ow];
        yc.fill(b[oc]);
        for ic in 0..s.c {
            let xc = &x[ic * s.h * s.w..(ic + 1) * s.h * s.w];
            for ky in 0..k {
                let (y0, y1) = valid_range(ky, pad, s.h, oh);
                for kx in 0..k {
                    let wv = w[((oc * s.c + ic) * k + ky) * k + kx];
                    let (x0, x1) = valid_range(kx, pad, s.w, ow);
                    for oy in y0..y1 {
                        let iy = oy + ky - pad;
                        let xrow = &xc[iy * s.w + x0 + kx - pad..iy * s.w + x1 + kx - pad];
                        for (yv, &xv) in yc[oy * ow + x0..oy * ow + x1].iter_mut().zip(xrow) {
                            *yv += wv * xv;
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_backward(
    dx: &mut [f32],
    g: &mut [f32],
    dy: &[f32],
    x: &[f32],
    p: &[f32],
    s: Shape,
    out_c: usize,
    k: usize,
    pad: usize,
) {
    let (oh, ow) = (s.h + 2 * pad - k + 1, s.w + 2 * pad - k + 1);
    let nw = out_c * s.c * k * k;
    let w = &p[..nw];
    let (gw, gb) = g.split_at_mut(nw);
    for oc in 0..out_c {
        let dyc = &dy[oc * oh * ow..(oc + 1) * oh * ow];
        gb[oc] += dyc.iter().sum::<f32>();
        for ic in 0..s.c {
            let xc = &x[ic * s.h * s.w..(ic + 1) * s.h * s.w];
            let dxc = &mut dx[ic * s.h * s.w..(ic + 1) * s.h * s.w];
            for ky in 0..k {
                let (y0, y1) = valid_range(ky, pad, s.h, oh);
                for kx in 0..k {
                    let wi = ((oc * s.c + ic) * k + ky) * k + kx;
                    let (x0, x1) = valid_range(kx, pad, s.w, ow);
                    let mut acc = 0.0;
                    for oy in y0..y1 {
                        let i0 = (oy + ky - pad) * s.w + x0 + kx - pad;
                        let drow = &dyc[oy * ow + x0..oy * ow + x1];
                        acc += crate::dot(drow, &xc[i0..i0 + drow.len()]);
                        for (dxv, &dv) in dxc[i0..i0 + drow.len()].iter_mut().zip(drow) {
                            *dxv += w[wi] * dv;
                        }
                    }
                    gw[wi] += acc;
                }
            }
        }
    }
}

fn maxpool_forward(y: &mut [f32], x: &[f32], s: Shape) {
    let (oh, ow) = (s.h / 2, s.w / 2);
    for c in 0..s.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let i = c * s.h * s.w + 2 * oy * s.w + 2 * ox;
                y[(c * oh + oy) * ow + ox] = x[i].max(x[i + 1]).max(x[i + s.w]).max(x[i + s.w + 1]);
            }
        }
    }
}

/// Routes each gradient to the input that won the max (the first one, on ties).
fn maxpool_backward(dx: &mut [f32], dy: &[f32], x: &[f32], s: Shape) {
    let (oh, ow) = (s.h / 2, s.w / 2);
    for c in 0..s.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let i = c * s.h * s.w + 2 * oy * s.w + 2 * ox;
                let window = [i, i + 1, i + s.w, i + s.w + 1];
                let best = window.into_iter().reduce(|a, b| if x[b] > x[a] { b } else { a }).unwrap();
                dx[best] += dy[(c * oh + oy) * ow + ox];
            }
        }
    }
}

fn dense_forward(y: &mut [f32], x: &[f32], p: &[f32], out: usize) {
    let n = x.len();
    let (w, b) = p.split_at(out * n);
    for (j, yv) in y.iter_mut().enumerate() {
        *yv = b[j] + crate::dot(&w[j * n..(j + 1) * n], x);
    }
}

fn dense_backward(dx: &mut [f32], g: &mut [f32], dy: &[f32], x: &[f32], p: &[f32], out: usize) {
    let n = x.len();
    let w = &p[..out * n];
    let (gw, gb) = g.split_at_mut(out * n);
    for (j, &d) in dy.iter().enumerate() {
        gb[j] += d;
        if d == 0.0 {
            continue; // common after ReLU, and saves a full row of work
        }
        for (gv, &xv) in gw[j * n..(j + 1) * n].iter_mut().zip(x) {
            *gv += d * xv;
        }
        for (dxv, &wv) in dx.iter_mut().zip(&w[j * n..(j + 1) * n]) {
            *dxv += d * wv;
        }
    }
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

pub fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Net, Vec<f32>) {
        use Layer::*;
        let mut net = Net::new(
            Shape { c: 2, h: 8, w: 8 },
            vec![
                Conv { out_c: 3, k: 3, pad: 1 },
                Relu,
                MaxPool,
                Conv { out_c: 4, k: 3, pad: 0 },
                Relu,
                Dense { out: 6 },
                Relu,
                Dense { out: 5 },
            ],
        );
        let mut rng = Rng::new(3);
        net.init(&mut rng);
        for p in &mut net.params {
            p.iter_mut().for_each(|v| *v += 0.1 * rng.normal()); // non-zero biases too
        }
        let x = (0..net.shapes[0].len()).map(|_| rng.uniform()).collect();
        (net, x)
    }

    fn loss(net: &Net, x: &[f32], label: usize) -> f64 {
        let probs = softmax(net.forward(x).last().unwrap());
        -(probs[label] as f64).ln()
    }

    #[test]
    fn shapes_are_inferred() {
        let net = Net::mnist();
        let s: Vec<_> = net.shapes.iter().map(|s| (s.c, s.h, s.w)).collect();
        assert_eq!(s[0], (1, 28, 28));
        assert_eq!(s[3], (16, 14, 14));
        assert_eq!(s[6], (32, 7, 7));
        assert_eq!(s[9], (10, 1, 1));
        assert_eq!(net.num_params(), 416 + 4640 + 200_832 + 1290);
    }

    /// Checks every parameter gradient and the input gradient against central finite differences.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gradients_match_finite_differences() {
        let (mut net, x) = tiny();
        let label = 2;
        let mut grads = net.zero_grads();
        let acts = net.forward(&x);
        let mut dlogits = softmax(acts.last().unwrap());
        dlogits[label] -= 1.0;
        let dx = net.backward(&acts, &dlogits, &mut grads);

        let eps = 1e-2;
        let close = |a: f32, n: f32| (a - n).abs() < 1e-3 + 0.03 * a.abs().max(n.abs());
        for li in 0..net.params.len() {
            for i in 0..net.params[li].len() {
                let orig = net.params[li][i];
                net.params[li][i] = orig + eps;
                let lp = loss(&net, &x, label);
                net.params[li][i] = orig - eps;
                let lm = loss(&net, &x, label);
                net.params[li][i] = orig;
                let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;
                assert!(close(grads[li][i], numeric), "layer {li} param {i}: {} vs {numeric}", grads[li][i]);
            }
        }
        let mut xp = x.clone();
        for i in 0..x.len() {
            xp[i] = x[i] + eps;
            let lp = loss(&net, &xp, label);
            xp[i] = x[i] - eps;
            let lm = loss(&net, &xp, label);
            xp[i] = x[i];
            let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;
            assert!(close(dx[i], numeric), "input {i}: {} vs {numeric}", dx[i]);
        }
    }

    #[test]
    fn can_overfit_a_few_examples() {
        let (mut net, _) = tiny();
        let mut rng = Rng::new(9);
        let data: Vec<(Vec<f32>, usize)> =
            (0..5).map(|c| ((0..net.shapes[0].len()).map(|_| rng.uniform()).collect(), c)).collect();
        let mut opt = crate::optim::Adam::new(&net);
        let mut total = 0.0;
        for _ in 0..200 {
            let mut grads = net.zero_grads();
            total = 0.0;
            for (x, y) in &data {
                total += net.train_step(x, *y, &mut grads).0;
            }
            opt.step(&mut net.params, &grads, 1.0 / data.len() as f32, 1e-2);
        }
        assert!(total / 5.0 < 0.05, "mean loss {}", total / 5.0);
    }
}
