use compresso::{
    CompressOptions, Connectivity, Error, Header, Label, compress, decompress, decompress_bytes,
    decompress_z_range, labels, remap,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A volume of axis-aligned boxes over a noisy background.
fn volume<T: Label>(rng: &mut Rng, [sx, sy, sz]: [usize; 3], max_label: u64) -> Vec<T> {
    let mut v = vec![T::from_u64(rng.below(3)); sx * sy * sz];
    for _ in 0..rng.below(12) {
        let label = T::from_u64(max_label - rng.below(max_label.min(20) + 1));
        let lo = [
            rng.below(sx as u64),
            rng.below(sy as u64),
            rng.below(sz as u64),
        ];
        let hi = [
            lo[0] + rng.below(sx as u64),
            lo[1] + rng.below(sy as u64),
            lo[2] + rng.below(sz as u64),
        ];
        for z in lo[2] as usize..(hi[2] as usize).min(sz) {
            for y in lo[1] as usize..(hi[1] as usize).min(sy) {
                for x in lo[0] as usize..(hi[0] as usize).min(sx) {
                    v[x + sx * (y + sy * z)] = label;
                }
            }
        }
    }
    for _ in 0..rng.below(8) {
        let i = rng.below(v.len() as u64) as usize;
        v[i] = T::from_u64(rng.next() & max_label);
    }
    v
}

fn roundtrip<T: Label>(rng: &mut Rng) {
    for _ in 0..150 {
        let shape = [
            1 + rng.below(30) as usize,
            1 + rng.below(30) as usize,
            1 + rng.below(6) as usize,
        ];
        let data = volume::<T>(rng, shape, T::MAX);
        let connectivity = if rng.below(2) == 0 {
            Connectivity::Four
        } else {
            Connectivity::Six
        };
        let steps = loop {
            let s = [
                1 + rng.below(8) as u8,
                1 + rng.below(8) as u8,
                1 + rng.below(3) as u8,
            ];
            if s.iter().map(|&v| usize::from(v)).product::<usize>() <= 64 {
                break s;
            }
        };
        let options = CompressOptions::default()
            .steps(steps)
            .connectivity(connectivity)
            .random_access_z_index(rng.below(2) == 0);
        let stream = match compress(&data, shape, &options) {
            Ok(stream) => stream,
            Err(Error::TooManyWindows) => continue,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(decompress::<T>(&stream).unwrap(), data);
        let sxy = shape[0] * shape[1];
        if options.random_access_z_index && connectivity == Connectivity::Four {
            let start = rng.below(shape[2] as u64) as usize;
            let end = start + 1 + rng.below((shape[2] - start) as u64) as usize;
            assert_eq!(
                decompress_z_range::<T>(&stream, start..end).unwrap(),
                data[start * sxy..end * sxy]
            );
        }
        let mut expected: Vec<u64> = data.iter().map(|v| v.to_u64()).collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(labels(&stream).unwrap(), expected);
    }
}

#[test]
fn roundtrip_all_label_types() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    roundtrip::<u8>(&mut rng);
    roundtrip::<u16>(&mut rng);
    roundtrip::<u32>(&mut rng);
    roundtrip::<u64>(&mut rng);
}

#[test]
fn encode_errors() {
    let data = vec![0u32; 24];
    let opts = CompressOptions::default();
    assert_eq!(
        compress(&data, [2, 3, 5], &opts),
        Err(Error::ShapeMismatch {
            shape: [2, 3, 5],
            len: 24
        })
    );
    assert!(matches!(
        compress(&data, [usize::MAX, 2, 2], &opts),
        Err(Error::ShapeMismatch { .. })
    ));
    let big = vec![0u8; 70_000];
    assert_eq!(
        compress(&big, [70_000, 1, 1], &opts),
        Err(Error::DimensionTooLarge {
            shape: [70_000, 1, 1]
        })
    );
    for steps in [[0, 4, 1], [8, 8, 2], [4, 4, 5]] {
        assert_eq!(
            compress(&data, [2, 3, 4], &opts.clone().steps(steps)),
            Err(Error::InvalidSteps { steps })
        );
    }
}

#[test]
fn empty_volume_is_a_bare_header() {
    let stream = compress::<u16>(&[], [0, 5, 7], &CompressOptions::default()).unwrap();
    assert_eq!(stream.len(), Header::SIZE);
    let header = Header::parse(&stream).unwrap();
    assert_eq!(header.format_version, 0);
    assert_eq!(header.shape, [0, 5, 7]);
    assert_eq!(header.steps, [4, 4, 1]);
    assert_eq!(decompress::<u16>(&stream).unwrap(), vec![]);
}

#[test]
fn header_roundtrip_and_accessors() {
    let data: Vec<u64> = (0..10 * 300 * 2).map(|i| i / 13).collect();
    let stream = compress(
        &data,
        [10, 300, 2],
        &CompressOptions::default().steps([8, 8, 1]),
    )
    .unwrap();
    let header = Header::parse(&stream).unwrap();
    assert_eq!(header.to_bytes(), stream[..Header::SIZE]);
    assert_eq!(header.voxels(), 6000);
    assert_eq!(header.nbytes(), 48_000);
    assert_eq!(header.window_width(), 8);
    assert_eq!(header.index_width(), 2); // 2 * 10 * 300 = 6000 < 65535
    assert!(header.has_z_index());
    assert_eq!(decompress_bytes(&stream).unwrap().len(), 48_000);
}

#[test]
fn remap_rewrites_ids_and_literal_locations() {
    // Labels near u8::MAX use literal escapes (code 6) in the locations.
    let data: Vec<u8> = vec![250, 251, 252, 253, 254, 255, 1, 2, 3];
    let stream = compress(&data, [9, 1, 1], &CompressOptions::default()).unwrap();
    let swapped = remap(&stream, |l| Some(255 - l), false).unwrap();
    let expected: Vec<u8> = data.iter().map(|l| 255 - l).collect();
    assert_eq!(decompress::<u8>(&swapped).unwrap(), expected);

    assert_eq!(
        remap(&stream, |l| (l != 3).then_some(l), false),
        Err(Error::MissingLabel(3))
    );
    assert_eq!(
        decompress::<u8>(&remap(&stream, |l| (l != 3).then_some(0), true).unwrap()).unwrap(),
        vec![0, 0, 0, 0, 0, 0, 0, 0, 3]
    );
    assert_eq!(
        remap(&stream, |_| Some(256), false),
        Err(Error::LabelOverflow(256))
    );
}

#[test]
fn python_reference_stream() {
    // compresso.compress(np.arange(24, dtype=np.uint32).reshape(2, 3, 4))
    // with compresso 3.3.3 (header only; the full stream is 144 bytes).
    let data: Vec<u32> = {
        // NumPy C-order arange reshaped to (2, 3, 4); flatten in Fortran order.
        let mut v = Vec::new();
        for z in 0..4 {
            for y in 0..3 {
                for x in 0..2 {
                    v.push((x * 12 + y * 4 + z) as u32);
                }
            }
        }
        v
    };
    let stream = compress(&data, [2, 3, 4], &CompressOptions::default()).unwrap();
    assert_eq!(stream.len(), 144);
    let hex: String = stream[..36].iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "6370736f0104020003000400040401040000000000000001000000140000000000000004"
    );
}

#[test]
fn remap_to_labels_that_need_a_literal_escape_keeps_the_z_index_valid() {
    // Per-slice isolated voxels with small labels are stored inline (label + 7).
    let shape = [5, 4, 6];
    let mut data = vec![0u8; 5 * 4 * 6];
    for z in 0..6 {
        for (i, x) in [(0, 0), (1, 2), (2, 4)] {
            data[x + 5 * (i + 4 * z)] = (10 + i + z) as u8;
        }
    }
    let stream = compress(&data, shape, &CompressOptions::default()).unwrap();
    let mapping = |l: u64| Some(if l == 0 { 0 } else { 250 + l % 5 });
    let remapped = remap(&stream, mapping, false).unwrap();
    assert!(
        remapped.len() > stream.len(),
        "literal escapes were inserted"
    );
    let expected: Vec<u8> = data
        .iter()
        .map(|&l| mapping(u64::from(l)).unwrap() as u8)
        .collect();
    assert_eq!(decompress::<u8>(&remapped).unwrap(), expected);
    let sxy = 20;
    for (start, end) in [(0, 6), (1, 2), (3, 6), (5, 6)] {
        assert_eq!(
            decompress_z_range::<u8>(&remapped, start..end).unwrap(),
            expected[start * sxy..end * sxy]
        );
    }
    // Mapping back yields the original stream contents.
    let back = remap(
        &remapped,
        |l| {
            Some(match l {
                0 => 0,
                l => l,
            })
        },
        false,
    )
    .unwrap();
    assert_eq!(decompress::<u8>(&back).unwrap(), expected);
}
