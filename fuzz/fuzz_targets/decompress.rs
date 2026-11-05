//! Arbitrary bytes must never panic any decoding entry point.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip streams that declare huge volumes; they are valid but slow.
    if compresso::Header::parse(data).is_ok_and(|h| h.nbytes() > 1 << 24) {
        return;
    }
    let _ = compresso::decompress_bytes(data);
    if let Ok(header) = compresso::Header::parse(data) {
        let sz = usize::from(header.shape[2]);
        let _ = compresso::decompress_z_range_bytes(data, sz / 3..sz);
    }
    let _ = compresso::labels(data);
    let _ = compresso::remap(data, |l| Some(l.wrapping_mul(3) & 0xFF), true);
    let _ = compresso::valid(data);
});
