//! Minimal data parallelism on top of `std::thread::scope`.

use std::sync::OnceLock;

pub fn num_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RUSTVISION_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .max(1)
    })
}

/// Splits `items` into one contiguous chunk per thread and maps `f` over the chunks in parallel.
pub fn par_chunks<T: Sync, R: Send, F: Fn(&[T]) -> R + Sync>(items: &[T], f: F) -> Vec<R> {
    let n = num_threads().min(items.len()).max(1);
    if n == 1 {
        return vec![f(items)];
    }
    let size = items.len().div_ceil(n);
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(size).map(|c| s.spawn(move || f(c))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
