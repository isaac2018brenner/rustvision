//! Reader for the MNIST IDX file format.

use std::io;
use std::path::Path;

pub const SIDE: usize = 28;
pub const PIXELS: usize = SIDE * SIDE;

pub struct Dataset {
    /// Row-major images scaled to [0, 1], `PIXELS` values each.
    pub images: Vec<f32>,
    pub labels: Vec<u8>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image(&self, i: usize) -> &[f32] {
        &self.images[i * PIXELS..(i + 1) * PIXELS]
    }

    /// Loads `{prefix}-images-idx3-ubyte` and `{prefix}-labels-idx1-ubyte` from `dir`.
    pub fn load(dir: &Path, prefix: &str) -> io::Result<Self> {
        let images = std::fs::read(dir.join(format!("{prefix}-images-idx3-ubyte")))?;
        let labels = std::fs::read(dir.join(format!("{prefix}-labels-idx1-ubyte")))?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let be = |b: &[u8], i: usize| u32::from_be_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
        if images.len() < 16 || be(&images, 0) != 2051 || be(&images, 2) != SIDE || be(&images, 3) != SIDE {
            return Err(bad("bad image file header"));
        }
        if labels.len() < 8 || be(&labels, 0) != 2049 {
            return Err(bad("bad label file header"));
        }
        let n = be(&images, 1);
        if be(&labels, 1) != n || images.len() != 16 + n * PIXELS || labels.len() != 8 + n {
            return Err(bad("image and label files do not match"));
        }
        Ok(Dataset { images: images[16..].iter().map(|&p| p as f32 / 255.0).collect(), labels: labels[8..].to_vec() })
    }
}
