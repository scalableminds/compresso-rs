//! Malformed, truncated and corrupted streams must return errors (never panic),
//! and the errors that upstream raises must have the same meaning.

use compresso::{
    CompressOptions, Connectivity, Error, Header, compress, decompress, decompress_bytes,
    decompress_z_range, labels, remap, valid,
};
use std::path::PathBuf;

/// A 1×1×1 u8 volume whose single voxel is a boundary voxel and therefore
/// decoded from the given location instructions.
fn single_boundary_voxel(locations: &[u8]) -> Vec<u8> {
    let mut s = b"cpso".to_vec();
    s.extend([0, 1]); // version 0, u8 labels
    s.extend([1, 0, 1, 0, 1, 0]); // 1x1x1
    s.extend([1, 1, 1]); // steps
    s.extend(0u64.to_le_bytes()); // ids
    s.extend(1u32.to_le_bytes()); // values
    s.extend((locations.len() as u64).to_le_bytes());
    s.push(4); // connectivity
    s.push(1); // window value: boundary
    s.extend(locations);
    s.push(0); // windows: block 0 -> value 0
    s
}

#[test]
fn location_instructions_match_upstream() {
    assert_eq!(
        decompress::<u8>(&single_boundary_voxel(&[6, 255])),
        Ok(vec![255])
    );
    assert_eq!(
        decompress::<u8>(&single_boundary_voxel(&[42 + 7])),
        Ok(vec![42])
    );
    for offset in 0..6u8 {
        let err = decompress::<u8>(&single_boundary_voxel(&[offset])).unwrap_err();
        assert_eq!(err, Error::InvalidLocation { offset });
        assert_eq!(
            err.to_string(),
            format!("compresso: unable to decode indeterminate locations. (offset {offset})")
        );
    }
    let err = decompress::<u8>(&single_boundary_voxel(&[])).unwrap_err();
    assert_eq!(err, Error::NoLocations);
    assert_eq!(
        err.to_string(),
        "compresso: unable to decode indeterminate locations. (no locations)"
    );
    // A literal escape without its label.
    assert!(matches!(
        decompress::<u8>(&single_boundary_voxel(&[6])),
        Err(Error::Corrupt(_))
    ));
}

#[test]
fn invalid_headers() {
    let good = single_boundary_voxel(&[6, 255]);
    for n in [0, 1, 35] {
        assert_eq!(
            decompress_bytes(&good[..n]),
            Err(Error::TooShort { len: n })
        );
        assert_eq!(Header::parse(&good[..n]), Err(Error::TooShort { len: n }));
    }
    let with = |offset: usize, value: u8| {
        let mut s = good.clone();
        s[offset] = value;
        s
    };
    assert_eq!(decompress_bytes(&with(0, b'x')), Err(Error::InvalidMagic));
    assert_eq!(
        decompress_bytes(&with(4, 2)),
        Err(Error::UnsupportedVersion(2))
    );
    assert_eq!(
        decompress_bytes(&with(5, 3)),
        Err(Error::InvalidDataWidth(3))
    );
    assert_eq!(
        decompress_bytes(&with(5, 0)),
        Err(Error::InvalidDataWidth(0))
    );
    assert_eq!(
        decompress_bytes(&with(35, 5)),
        Err(Error::InvalidConnectivity(5))
    );
    // Zero-sized volumes decode to nothing, like Python.
    assert_eq!(decompress_bytes(&with(6, 0)), Ok(vec![]));
    // Width mismatch for typed decoding.
    assert_eq!(
        decompress::<u32>(&good),
        Err(Error::LabelWidthMismatch {
            stream: 1,
            requested: 4
        })
    );
    // Steps that divide by zero or exceed 64 bit windows.
    for steps in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [8, 8, 2], [255, 255, 255]] {
        let mut s = good.clone();
        s[12..15].copy_from_slice(&steps);
        assert_eq!(decompress_bytes(&s), Err(Error::InvalidSteps { steps }));
    }
    // Section sizes beyond the input.
    for (range, value) in [
        (15..23, u64::MAX),
        (15..23, 1u64 << 40),
        (23..27, u64::from(u32::MAX)),
        (27..35, u64::MAX),
        (27..35, 100),
    ] {
        let mut s = good.clone();
        let len = range.len();
        s[range].copy_from_slice(&value.to_le_bytes()[..len]);
        assert_eq!(decompress_bytes(&s), Err(Error::Truncated));
        assert!(!valid(&s));
    }
    assert!(valid(&good));
}

#[test]
fn corrupt_windows() {
    // Window value index out of range.
    let mut s = single_boundary_voxel(&[6, 255]);
    *s.last_mut().unwrap() = 2;
    assert!(matches!(decompress_bytes(&s), Err(Error::Corrupt(_))));
    // A window written outside the grid.
    let mut s = single_boundary_voxel(&[6, 255]);
    s.push(0);
    assert!(matches!(decompress_bytes(&s), Err(Error::Corrupt(_))));
    // Like upstream, a zero run past the end of the grid is harmless.
    let mut s = single_boundary_voxel(&[6, 255]);
    s.push(255);
    assert_eq!(decompress_bytes(&s), Ok(vec![255]));
}

#[test]
fn z_range_errors() {
    let volume: Vec<u16> = (0..4 * 4 * 5).map(|i| (i / 7) as u16).collect();
    let shape = [4, 4, 5];
    let v1 = compress(&volume, shape, &CompressOptions::default()).unwrap();
    let v0 = compress(
        &volume,
        shape,
        &CompressOptions::default().random_access_z_index(false),
    )
    .unwrap();
    let c6 = compress(
        &volume,
        shape,
        &CompressOptions::default().connectivity(Connectivity::Six),
    )
    .unwrap();
    assert_eq!(Header::parse(&c6).unwrap().format_version, 0);

    for stream in [&v1, &v0, &c6] {
        assert_eq!(decompress_z_range::<u16>(stream, 0..5).unwrap(), volume);
        assert_eq!(decompress_z_range::<u16>(stream, 5..5).unwrap(), vec![]);
        assert_eq!(
            decompress_z_range::<u16>(stream, 0..6),
            Err(Error::InvalidZRange {
                start: 0,
                end: 6,
                sz: 5
            })
        );
        #[allow(clippy::reversed_empty_ranges)]
        let reversed = decompress_z_range::<u16>(stream, 3..2);
        assert!(matches!(reversed, Err(Error::InvalidZRange { .. })));
    }
    assert_eq!(
        decompress_z_range::<u16>(&v1, 1..3).unwrap(),
        volume[16..48]
    );
    assert_eq!(
        decompress_z_range::<u16>(&v0, 1..3),
        Err(Error::RandomAccessUnavailable)
    );
    assert_eq!(
        decompress_z_range::<u16>(&c6, 1..3),
        Err(Error::RandomAccessUnavailable)
    );
    // A version 1 stream claiming 6-connectivity.
    let mut s = v1.clone();
    s[35] = 6;
    assert_eq!(
        decompress_z_range::<u16>(&s, 1..3),
        Err(Error::RandomAccessConnectivity6)
    );
}

fn fixture_streams() -> Vec<Vec<u8>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cpso"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect()
}

/// Decodes a possibly corrupt stream through every entry point, skipping
/// headers that declare more than 16 MiB of output.
fn exercise(stream: &[u8]) {
    if Header::parse(stream).is_ok_and(|h| h.nbytes() > 16 << 20) {
        return;
    }
    let _ = decompress_bytes(stream);
    if let Ok(h) = Header::parse(stream) {
        let sz = usize::from(h.shape[2]);
        let _ = compresso::decompress_z_range_bytes(stream, sz / 2..sz);
    }
    let _ = labels(stream);
    let _ = remap(stream, |l| Some(l ^ 1), true);
    let _ = valid(stream);
}

#[test]
fn truncated_streams_never_panic() {
    for stream in fixture_streams().iter().step_by(4) {
        for len in 0..stream.len().min(160) {
            exercise(&stream[..len]);
        }
        for len in (0..stream.len()).step_by(97) {
            exercise(&stream[..len]);
        }
    }
}

#[test]
fn mutated_streams_never_panic() {
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for stream in fixture_streams() {
        if stream.len() > 20_000 {
            continue;
        }
        for _ in 0..16 {
            let mut s = stream.clone();
            for _ in 0..1 + next() % 4 {
                let i = (next() as usize) % s.len();
                s[i] = next() as u8;
            }
            exercise(&s);
        }
    }
}

#[test]
fn huge_declared_volumes_fail_without_aborting() {
    let mut s = single_boundary_voxel(&[6, 255]);
    s[5] = 8; // u64 labels
    s[6..12].fill(0xFF); // 65535^3 voxels, about 2 PB
    let err = decompress_bytes(&s).unwrap_err();
    assert!(
        matches!(err, Error::AllocationFailed { .. } | Error::Truncated),
        "{err}"
    );
    assert_eq!(Header::parse(&s).unwrap().nbytes(), 65535u64.pow(3) * 8);
}
