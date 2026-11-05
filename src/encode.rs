use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use crate::ccl::{Runs, UnionFind, alloc_zeroed, label_runs, next_set};
use crate::error::{Error, Result};
use crate::header::{Header, index_width, window_width};
use crate::label::{Label, write_le};

/// Connectivity of the connected components of the non-boundary voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Connectivity {
    /// 2D (in-plane) 4-connectivity: each z slice is labeled independently.
    /// Required for the random access z index.
    #[default]
    Four,
    /// 3D 6-connectivity.
    Six,
}

impl Connectivity {
    /// The numeric value stored in the header (4 or 6).
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            Connectivity::Four => 4,
            Connectivity::Six => 6,
        }
    }
}

/// Options for [`compress`](crate::compress).
///
/// The defaults match `compresso.compress(data)` in Python: steps `(4, 4, 1)`
/// with automatic fallback to `(8, 8, 1)`, 4-connectivity and the random
/// access z index enabled.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Boundary window size `[xstep, ystep, zstep]`. The product must be
    /// between 1 and 64.
    ///
    /// `None` uses `[4, 4, 1]`. If that window cannot be run-length encoded
    /// because there are too many distinct windows, `None` falls back to
    /// `[8, 8, 1]`, as Python does when `steps` is not given.
    pub steps: Option<[u8; 3]>,
    /// Connectivity used for the connected components.
    pub connectivity: Connectivity,
    /// Append a z index that allows decoding z slices independently
    /// (format version 1). It is ignored (treated as `false`) for
    /// [`Connectivity::Six`], as in Python.
    pub random_access_z_index: bool,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            steps: None,
            connectivity: Connectivity::Four,
            random_access_z_index: true,
        }
    }
}

impl CompressOptions {
    /// Sets explicit grid steps (disables the `(8, 8, 1)` fallback).
    #[must_use]
    pub fn steps(mut self, steps: [u8; 3]) -> Self {
        self.steps = Some(steps);
        self
    }

    /// Sets the connectivity.
    #[must_use]
    pub fn connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Enables or disables the random access z index.
    #[must_use]
    pub fn random_access_z_index(mut self, enabled: bool) -> Self {
        self.random_access_z_index = enabled;
        self
    }
}

const DEFAULT_STEPS: [u8; 3] = [4, 4, 1];
const FALLBACK_STEPS: [u8; 3] = [8, 8, 1];

/// Compresses a 3D label volume into a Compresso stream.
///
/// `data` holds `shape[0] * shape[1] * shape[2]` labels in Fortran order:
/// x varies fastest, then y, then z. This is the memory layout of
/// `np.asfortranarray(data)` for an array of shape `(sx, sy, sz)`.
///
/// The output is byte-identical to
/// `compresso.compress(data, steps, connectivity, random_access_z_index)`.
///
/// # Errors
///
/// Returns an error if `data.len()` does not match `shape`
/// ([`Error::ShapeMismatch`]), if a dimension exceeds 65535
/// ([`Error::DimensionTooLarge`]), if the steps are invalid
/// ([`Error::InvalidSteps`]), or if explicit steps produce too many distinct
/// windows ([`Error::TooManyWindows`]).
///
/// ```
/// use compresso::{compress, decompress, CompressOptions};
///
/// let shape = [4, 3, 2];
/// let labels: Vec<u32> = (0..24).map(|i| i / 5).collect();
/// let stream = compress(&labels, shape, &CompressOptions::default())?;
/// assert_eq!(decompress::<u32>(&stream)?, labels);
/// # Ok::<(), compresso::Error>(())
/// ```
pub fn compress<T: Label>(
    data: &[T],
    shape: [usize; 3],
    options: &CompressOptions,
) -> Result<Vec<u8>> {
    let voxels = shape
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s))
        .filter(|&v| v == data.len())
        .ok_or(Error::ShapeMismatch {
            shape,
            len: data.len(),
        })?;
    if shape.iter().any(|&s| s > usize::from(u16::MAX)) {
        return Err(Error::DimensionTooLarge { shape });
    }
    let steps = options.steps.unwrap_or(DEFAULT_STEPS);
    let connectivity = options.connectivity;
    let random_access = options.random_access_z_index && connectivity == Connectivity::Four;
    let dims = shape.map(|s| s as u16);

    if voxels == 0 {
        let header = Header {
            format_version: 0,
            data_width: T::WIDTH as u8,
            shape: dims,
            steps,
            id_size: 0,
            value_size: 0,
            location_size: 0,
            connectivity: connectivity.as_u8(),
        };
        return Ok(header.to_bytes().to_vec());
    }
    validate_steps(steps)?;
    let boundaries = extract_boundaries(data, shape, connectivity)?;
    let (ids, components_per_slice) = component_ids(data, &boundaries, shape, connectivity)?;
    let (locations, z_index) =
        encode_indeterminate_locations(data, &boundaries, shape, connectivity, random_access)?;
    let windows = match encode_windows(&boundaries, shape, steps) {
        Err(Error::TooManyWindows) if options.steps.is_none() => {
            encode_windows(&boundaries, shape, FALLBACK_STEPS)?
        }
        other => other?,
    };
    drop(boundaries);

    let header = Header {
        format_version: u8::from(random_access),
        data_width: T::WIDTH as u8,
        shape: dims,
        steps: windows.steps,
        id_size: ids.len() as u64,
        value_size: u32::try_from(windows.values.len()).map_err(|_| Error::TooManyWindows)?,
        location_size: locations.len() as u64,
        connectivity: connectivity.as_u8(),
    };

    let ww = windows.width;
    let iw = index_width(shape[0], shape[1]);
    let index_bytes = if random_access { 2 * shape[2] * iw } else { 0 };
    let total = Header::SIZE
        + (ids.len() + locations.len()) * T::WIDTH
        + (windows.values.len() + windows.rle.len()) * ww
        + index_bytes;
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| Error::AllocationFailed { bytes: total })?;
    out.extend_from_slice(&header.to_bytes());
    for id in &ids {
        write_le(&mut out, id.to_u64(), T::WIDTH);
    }
    for &value in &windows.values {
        write_le(&mut out, value, ww);
    }
    for location in &locations {
        write_le(&mut out, location.to_u64(), T::WIDTH);
    }
    for &window in &windows.rle {
        write_le(&mut out, window, ww);
    }
    if random_access {
        // The integer casts intentionally truncate like `static_cast` upstream.
        for &count in &components_per_slice {
            write_le(&mut out, count, iw);
        }
        for &offset in &z_index {
            write_le(&mut out, offset, iw);
        }
    }
    debug_assert_eq!(out.len(), total);
    Ok(out)
}

pub(crate) fn validate_steps(steps: [u8; 3]) -> Result<()> {
    let product: usize = steps.iter().map(|&s| usize::from(s)).product();
    if product == 0 || product > 64 {
        return Err(Error::InvalidSteps { steps });
    }
    Ok(())
}

/// A voxel is a boundary if its label differs from its +x or +y neighbor
/// (or its +z neighbor with 6-connectivity).
fn extract_boundaries<T: Label>(
    data: &[T],
    [sx, sy, sz]: [usize; 3],
    connectivity: Connectivity,
) -> Result<Vec<u8>> {
    let sxy = sx * sy;
    let six = connectivity == Connectivity::Six;
    let mut boundaries = alloc_zeroed::<u8>(data.len())?;
    // Written as whole-row slice comparisons so that they vectorize.
    for (r, out) in boundaries.chunks_exact_mut(sx).enumerate() {
        let (y, z) = (r % sy, r / sy);
        let base = r * sx;
        let row = &data[base..base + sx];
        for ((o, a), b) in out.iter_mut().zip(row).zip(&row[1..]) {
            *o = u8::from(a != b);
        }
        if y + 1 < sy {
            let below = &data[base + sx..base + 2 * sx];
            for ((o, a), b) in out.iter_mut().zip(row).zip(below) {
                *o |= u8::from(a != b);
            }
        }
        if six && z + 1 < sz {
            let behind = &data[base + sxy..base + sxy + sx];
            for ((o, a), b) in out.iter_mut().zip(row).zip(behind) {
                *o |= u8::from(a != b);
            }
        }
    }
    Ok(boundaries)
}

/// Labels the components of the non-boundary voxels and returns the label of
/// each component (in first-occurrence order) and the number of components
/// per z slice (only meaningful for 4-connectivity).
fn component_ids<T: Label>(
    data: &[T],
    boundaries: &[u8],
    [sx, sy, sz]: [usize; 3],
    connectivity: Connectivity,
) -> Result<(Vec<T>, Vec<u64>)> {
    let sxy = sx * sy;
    let mut uf = UnionFind::new();
    let mut ids = Vec::new();
    let mut per_slice = vec![0u64; sz];
    let mut first_labels: Vec<T> = Vec::new();
    let mut runs = Runs::default();
    match connectivity {
        Connectivity::Four => {
            for (z, count) in per_slice.iter_mut().enumerate() {
                let slice = &data[z * sxy..(z + 1) * sxy];
                uf.reset();
                first_labels.clear();
                label_runs(
                    &boundaries[z * sxy..(z + 1) * sxy],
                    [sx, sy, 1],
                    false,
                    &mut runs,
                    &mut uf,
                    |i| first_labels.push(slice[i]),
                )?;
                let before = ids.len();
                uf.for_each_root(|root| ids.push(first_labels[root as usize - 1]));
                *count = (ids.len() - before) as u64;
            }
        }
        Connectivity::Six => {
            label_runs(boundaries, [sx, sy, sz], true, &mut runs, &mut uf, |i| {
                first_labels.push(data[i])
            })?;
            uf.for_each_root(|root| ids.push(first_labels[root as usize - 1]));
        }
    }
    Ok((ids, per_slice))
}

/// Boundary voxels whose label cannot be inferred from an already decoded
/// non-boundary neighbor get an explicit instruction. 0-5 copy a neighbor
/// (-x, +x, -y, +y, -z, +z), 6 is followed by a literal label, and any
/// larger value is `label + 7`.
///
/// Returns the instructions and the difference-coded offset of the first
/// instruction of each z slice.
fn encode_indeterminate_locations<T: Label>(
    labels: &[T],
    boundaries: &[u8],
    [sx, sy, sz]: [usize; 3],
    connectivity: Connectivity,
    random_access: bool,
) -> Result<(Vec<T>, Vec<u64>)> {
    let sxy = sx * sy;
    let six = connectivity == Connectivity::Six;
    let mut locations: Vec<T> = Vec::new();
    let mut z_index = Vec::with_capacity(sz);
    let is_candidate = |n: usize, loc: usize| boundaries[n] == 0 && labels[n] == labels[loc];

    for z in 0..sz {
        z_index.push(locations.len() as u64);
        for y in 0..sy {
            let base = sx * y + sxy * z;
            let row = &boundaries[base..base + sx];
            let mut x = 0;
            while let Some(found) = next_set(row, x) {
                x = found + 1;
                let loc = base + found;
                let x = found;
                if (x > 0 && boundaries[loc - 1] == 0)
                    || (y > 0 && boundaries[loc - sx] == 0)
                    || (six && z > 0 && boundaries[loc - sxy] == 0)
                {
                    continue;
                }
                let code = if x > 0 && is_candidate(loc - 1, loc) {
                    0
                } else if x + 1 < sx && is_candidate(loc + 1, loc) {
                    1
                } else if y > 0 && is_candidate(loc - sx, loc) {
                    2
                } else if y + 1 < sy && is_candidate(loc + sx, loc) {
                    3
                } else if !random_access && z > 0 && is_candidate(loc - sxy, loc) {
                    4
                } else if !random_access && z + 1 < sz && is_candidate(loc + sxy, loc) {
                    5
                } else {
                    let label = labels[loc].to_u64();
                    if label > T::MAX - 7 {
                        locations.push(T::from_u64(6));
                        locations.push(labels[loc]);
                    } else {
                        locations.push(T::from_u64(label + 7));
                    }
                    continue;
                };
                locations.push(T::from_u64(code));
            }
        }
    }
    for i in (1..z_index.len()).rev() {
        z_index[i] -= z_index[i - 1];
    }
    Ok((locations, z_index))
}

struct Windows {
    steps: [u8; 3],
    width: usize,
    /// Sorted distinct window bit masks.
    values: Vec<u64>,
    /// Run-length encoded indices into `values`.
    rle: Vec<u64>,
}

fn encode_windows(boundaries: &[u8], shape: [usize; 3], steps: [u8; 3]) -> Result<Windows> {
    let [sx, sy, sz] = shape;
    let [xs, ys, zs] = steps.map(usize::from);
    let nx = sx.div_ceil(xs);
    let ny = sy.div_ceil(ys);
    let nz = sz.div_ceil(zs);
    let mut windows = alloc_zeroed::<u64>(nx * ny * nz)?;
    for (r, row) in boundaries.chunks_exact(sx).enumerate() {
        let (y, z) = (r % sy, r / sy);
        let block = nx * (y / ys) + nx * ny * (z / zs);
        let shift = xs * (y % ys) + xs * ys * (z % zs);
        for (window, chunk) in windows[block..block + nx].iter_mut().zip(row.chunks(xs)) {
            let mut bits = 0u64;
            for (i, &b) in chunk.iter().enumerate() {
                bits |= u64::from(b) << i;
            }
            *window |= bits << shift;
        }
    }

    // Distinct window values in ascending order and their ranks.
    let mut ranks: HashMap<u64, u64, BuildMix> = HashMap::default();
    let mut last = None;
    for &window in &windows {
        if last != Some(window) {
            ranks.entry(window).or_insert(0);
            last = Some(window);
        }
    }
    let mut values: Vec<u64> = ranks.keys().copied().collect();
    values.sort_unstable();

    // Window indices are stored shifted left by one bit (the low bit flags
    // a zero run), so they must not exceed half the window integer range.
    let width = window_width(xs * ys * zs);
    let max_run = (u64::MAX >> (64 - 8 * width)) / 2;
    if values.len() as u64 - 1 > max_run {
        return Err(Error::TooManyWindows);
    }
    for (rank, value) in values.iter().enumerate() {
        ranks.insert(*value, rank as u64);
    }

    let mut rle = Vec::with_capacity(windows.len() / 4);
    let mut zero_run = 0u64;
    let mut last = None;
    for &window in &windows {
        // Consecutive windows are frequently identical; reuse the last lookup.
        let index = match last {
            Some((w, i)) if w == window => i,
            _ => {
                let i = ranks[&window];
                last = Some((window, i));
                i
            }
        };
        if index == 0 {
            zero_run += 1;
            if zero_run == max_run {
                rle.push((zero_run << 1) | 1);
                zero_run = 0;
            }
            continue;
        }
        if zero_run > 0 {
            rle.push((zero_run << 1) | 1);
            zero_run = 0;
        }
        rle.push(index << 1);
    }
    if zero_run > 0 {
        rle.push((zero_run << 1) | 1);
    }
    Ok(Windows {
        steps,
        width,
        values,
        rle,
    })
}

/// A fast hasher for `u64` keys (the splitmix64 finalizer).
#[derive(Default, Clone, Copy)]
struct BuildMix;

impl BuildHasher for BuildMix {
    type Hasher = Mix;
    fn build_hasher(&self) -> Mix {
        Mix(0)
    }
}

struct Mix(u64);

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | u64::from(b);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}
