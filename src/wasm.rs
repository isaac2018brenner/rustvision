//! A tiny C-style interface for the browser demo. No wasm-bindgen: JavaScript
//! writes pixels into a buffer in WebAssembly memory, calls `run`, and reads the
//! results back out of other buffers.

use crate::checkpoint;
use crate::mnist::PIXELS;
use crate::net::{argmax, softmax, Net};
use crate::rng::Rng;
use std::cell::RefCell;

struct State {
    net: Net,
    input: Vec<f32>,
    probs: Vec<f32>,
    conv1: Vec<f32>,
    conv2: Vec<f32>,
    saliency: Vec<f32>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new({
        let net = Net::mnist();
        // Feature maps after each conv layer's ReLU (layer outputs 2 and 5).
        let (c1, c2) = (net.shapes[2].len(), net.shapes[5].len());
        State { net, input: vec![0.0; PIXELS], probs: vec![0.0; 10], conv1: vec![0.0; c1], conv2: vec![0.0; c2], saliency: vec![0.0; PIXELS] }
    });
}

/// Pointers handed to JavaScript stay valid because these buffers are never resized.
fn ptr(f: impl Fn(&State) -> &Vec<f32>) -> *const f32 {
    STATE.with(|s| f(&s.borrow()).as_ptr())
}

/// Reserves `len` bytes that JavaScript can fill (used to pass in the model file).
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = vec![0u8; len];
    let p = buf.as_mut_ptr();
    std::mem::forget(buf);
    p
}

/// Loads weights previously copied to `ptr`. Returns 1 on success.
///
/// # Safety
/// `ptr` must come from `alloc(len)`.
#[no_mangle]
pub unsafe extern "C" fn load_model(ptr: *mut u8, len: usize) -> i32 {
    let bytes = Vec::from_raw_parts(ptr, len, len);
    STATE.with(|s| checkpoint::load_into(&mut s.borrow_mut().net, &bytes).is_ok() as i32)
}

#[no_mangle]
pub extern "C" fn input_ptr() -> *mut f32 {
    ptr(|s| &s.input) as *mut f32
}

#[no_mangle]
pub extern "C" fn probs_ptr() -> *const f32 {
    ptr(|s| &s.probs)
}

#[no_mangle]
pub extern "C" fn conv1_ptr() -> *const f32 {
    ptr(|s| &s.conv1)
}

#[no_mangle]
pub extern "C" fn conv2_ptr() -> *const f32 {
    ptr(|s| &s.conv2)
}

#[no_mangle]
pub extern "C" fn saliency_ptr() -> *const f32 {
    ptr(|s| &s.saliency)
}

/// Classifies the image in the input buffer and fills every output buffer. Returns the predicted digit.
#[no_mangle]
pub extern "C" fn run() -> u32 {
    STATE.with(|s| {
        let s = &mut *s.borrow_mut();
        let acts = s.net.forward(&s.input);
        s.probs.copy_from_slice(&softmax(acts.last().unwrap()));
        s.conv1.copy_from_slice(&acts[2]);
        s.conv2.copy_from_slice(&acts[5]);
        let pred = argmax(&s.probs);
        // SmoothGrad: average the gradient over noisy copies of the input, which
        // removes most of the speckle that a single raw gradient shows.
        const SAMPLES: usize = 16;
        let mut rng = Rng::new(1);
        s.saliency.fill(0.0);
        for _ in 0..SAMPLES {
            let noisy: Vec<f32> = s.input.iter().map(|&v| v + 0.15 * rng.normal()).collect();
            for (acc, g) in s.saliency.iter_mut().zip(s.net.saliency(&noisy, pred)) {
                *acc += g / SAMPLES as f32;
            }
        }
        pred as u32
    })
}
