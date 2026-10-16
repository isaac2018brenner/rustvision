//! Raw little-endian f32 weights behind a small header. The same bytes are
//! loaded by the command-line tool and by the WebAssembly demo.

use crate::net::Net;

const MAGIC: &[u8; 4] = b"RVIS";

pub fn to_bytes(net: &Net) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend((net.num_params() as u32).to_le_bytes());
    for p in &net.params {
        for v in p {
            out.extend(v.to_le_bytes());
        }
    }
    out
}

/// Loads weights into `net`, whose architecture must match the one that was saved.
pub fn load_into(net: &mut Net, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        return Err("not a rustvision checkpoint".into());
    }
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if n != net.num_params() || bytes.len() != 8 + 4 * n {
        return Err(format!("checkpoint has {n} parameters, network expects {}", net.num_params()));
    }
    let mut vals = bytes[8..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
    for p in &mut net.params {
        p.iter_mut().for_each(|v| *v = vals.next().unwrap());
    }
    Ok(())
}
