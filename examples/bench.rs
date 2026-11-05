//! Times compression and decompression of a raw volume, and optionally checks
//! the results against reference streams written by Python `compresso`.
//!
//! ```sh
//! cargo run --release --example bench -- <volume.raw> <width> <sx> <sy> <sz> [reference_prefix]
//! ```
//!
//! `volume.raw` holds little-endian labels in Fortran (x-fastest) order. If
//! `reference_prefix` is given, `<prefix>.<config>.cpso` files (configs:
//! default, s881, s221, c6, v0) are compared byte for byte.

use compresso::{
    CompressOptions, Connectivity, Label, compress, decompress, decompress_z_range_bytes,
};
use std::time::Instant;

fn configs() -> Vec<(&'static str, CompressOptions)> {
    let default = CompressOptions::default();
    vec![
        ("default", default.clone()),
        ("s881", default.clone().steps([8, 8, 1])),
        ("s221", default.clone().steps([2, 2, 1])),
        ("c6", default.clone().connectivity(Connectivity::Six)),
        ("v0", default.random_access_z_index(false)),
    ]
}

fn run<T: Label>(raw: &[u8], shape: [usize; 3], reference: Option<&str>) {
    let data: Vec<T> = raw
        .chunks_exact(T::WIDTH)
        .map(|c| {
            let mut v = [0u8; 8];
            v[..c.len()].copy_from_slice(c);
            T::from_u64(u64::from_le_bytes(v))
        })
        .collect();
    let mb = raw.len() as f64 / 1e6;
    for (name, options) in configs() {
        let t = Instant::now();
        let stream = match compress(&data, shape, &options) {
            Ok(stream) => stream,
            Err(e) => {
                println!("{name:8} encode error: {e}");
                continue;
            }
        };
        let encode = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let decoded = decompress::<T>(&stream).expect("decode");
        let decode = t.elapsed().as_secs_f64();
        assert!(decoded == data, "{name}: round trip failed");
        let matches = reference.map(|prefix| {
            std::fs::read(format!("{prefix}.{name}.cpso"))
                .map(|expected| expected == stream)
                .unwrap_or(false)
        });
        if let (Some(prefix), "default") = (reference, name) {
            // `<prefix>.z<start>_<end>.raw`: Python's decompress(stream, z=(start, end)).
            if let Ok(expected) = std::fs::read(format!("{prefix}.z100_103.raw")) {
                let t = Instant::now();
                let part = decompress_z_range_bytes(&stream, 100..103).expect("z range");
                let ok = if part == expected { "==" } else { "!=" };
                println!(
                    "z 100..103 decode {:6.4}s  {ok} python",
                    t.elapsed().as_secs_f64()
                );
            }
        }
        println!(
            "{name:8} {:>11} bytes ({:5.1}x)  encode {encode:6.3}s ({:6.0} MB/s)  decode {decode:6.3}s ({:6.0} MB/s){}",
            stream.len(),
            raw.len() as f64 / stream.len() as f64,
            mb / encode,
            mb / decode,
            match matches {
                Some(true) => "  == python",
                Some(false) => "  != PYTHON",
                None => "",
            }
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!("usage: bench <volume.raw> <width> <sx> <sy> <sz> [reference_prefix]");
        std::process::exit(2);
    }
    let raw = std::fs::read(&args[1]).expect("read volume");
    let width: usize = args[2].parse().expect("width");
    let shape = [3, 4, 5].map(|i| args[i].parse::<usize>().expect("shape"));
    let reference = args.get(6).map(String::as_str);
    match width {
        1 => run::<u8>(&raw, shape, reference),
        2 => run::<u16>(&raw, shape, reference),
        4 => run::<u32>(&raw, shape, reference),
        8 => run::<u64>(&raw, shape, reference),
        _ => panic!("width must be 1, 2, 4 or 8"),
    }
}
