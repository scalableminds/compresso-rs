//! compress -> decompress must reproduce the input for any volume and options.
#![no_main]
use compresso::{CompressOptions, Connectivity, Error, compress, decompress};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let (params, labels) = data.split_at(6);
    let sx = usize::from(params[0] % 16) + 1;
    let sy = usize::from(params[1] % 16) + 1;
    let sz = usize::from(params[2] % 8) + 1;
    let steps = [params[3] % 9, params[4] % 9, params[5] % 5];
    let options = CompressOptions {
        steps: if params[5] & 0x80 != 0 { None } else { Some(steps) },
        connectivity: if params[4] & 0x80 != 0 { Connectivity::Six } else { Connectivity::Four },
        random_access_z_index: params[3] & 0x80 != 0,
    };
    let volume: Vec<u16> = (0..sx * sy * sz)
        .map(|i| labels.get(i % labels.len().max(1)).map_or(0, |&b| u16::from(b % 4) * 16000))
        .collect();
    match compress(&volume, [sx, sy, sz], &options) {
        Ok(stream) => assert_eq!(decompress::<u16>(&stream).unwrap(), volume),
        Err(Error::InvalidSteps { .. } | Error::TooManyWindows) => {}
        Err(e) => panic!("{e}"),
    }
});
