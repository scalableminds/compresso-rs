use std::ops::Range;

use crate::ccl::{Runs, UnionFind, alloc_zeroed, label_runs, next_set};
use crate::error::{Error, Result};
use crate::header::Header;
use crate::label::{Label, Table};

/// The sections of a stream, as laid out by the upstream encoder.
pub(crate) struct Sections<'a> {
    pub header: Header,
    pub ids: Table<'a>,
    pub values: Table<'a>,
    pub locations: Table<'a>,
    /// Run-length encoded windows.
    pub windows: Table<'a>,
    /// Per-slice component counts and difference-coded location offsets
    /// (format version 1 only).
    pub z_index: Option<(Table<'a>, Table<'a>)>,
    /// Byte offset of the windows section (everything after the locations).
    pub tail_start: usize,
    /// Byte offset of the z index (the end of the last complete window).
    pub index_start: usize,
}

impl<'a> Sections<'a> {
    pub fn parse(stream: &'a [u8]) -> Result<Self> {
        let header = Header::parse(stream)?;
        let label_width = usize::from(header.data_width);
        let window_width = header.window_width();
        let mut offset = Header::SIZE;
        let mut take = |count: u64, width: usize| -> Result<Table<'a>> {
            let len = usize::try_from(count)
                .ok()
                .and_then(|c| c.checked_mul(width))
                .ok_or(Error::Truncated)?;
            let end = offset.checked_add(len).ok_or(Error::Truncated)?;
            let bytes = stream.get(offset..end).ok_or(Error::Truncated)?;
            offset = end;
            Ok(Table { bytes, width })
        };
        let ids = take(header.id_size, label_width)?;
        let values = take(u64::from(header.value_size), window_width)?;
        let locations = take(header.location_size, label_width)?;
        let tail_start = offset;

        let sz = usize::from(header.shape[2]);
        let index_width = header.index_width();
        let index_bytes = if header.has_z_index() {
            2 * sz * index_width
        } else {
            0
        };
        let window_bytes = (stream.len() - tail_start)
            .checked_sub(index_bytes)
            .ok_or(Error::Truncated)?;
        // Like upstream, a trailing partial window is ignored and the z index
        // is read directly after the last complete window.
        let window_end = tail_start + window_bytes / window_width * window_width;
        let windows = Table {
            bytes: &stream[tail_start..window_end],
            width: window_width,
        };
        let z_index = header.has_z_index().then(|| {
            let split = window_end + sz * index_width;
            (
                Table {
                    bytes: &stream[window_end..split],
                    width: index_width,
                },
                Table {
                    bytes: &stream[split..split + sz * index_width],
                    width: index_width,
                },
            )
        });
        Ok(Sections {
            header,
            ids,
            values,
            locations,
            windows,
            z_index,
            tail_start,
            index_start: window_end,
        })
    }
}

/// Decompresses a stream into a volume of labels of type `T`.
///
/// `T` must match the label width of the stream (see [`Header::data_width`]);
/// use [`decompress_bytes`] if the width is only known at runtime.
/// The result holds `sx * sy * sz` labels with x varying fastest.
///
/// The output matches `compresso.decompress(stream)` in Python. Malformed
/// streams return an error and never panic or read out of bounds.
/// Before decoding untrusted input, check [`Header::nbytes`]: a small
/// stream can declare a very large volume.
///
/// # Errors
///
/// Returns an error if the header is invalid, if `T` does not match the
/// label width of the stream, or if the stream is truncated or corrupt.
pub fn decompress<T: Label>(stream: &[u8]) -> Result<Vec<T>> {
    let header = Header::parse(stream)?;
    decode(stream, 0..usize::from(header.shape[2]))
}

/// Decompresses only the z slices in `z`. The result holds
/// `sx * sy * z.len()` labels.
///
/// Partial ranges require a format version 1 stream (encoded with the random
/// access z index) and 4-connectivity. The full range `0..sz` works for every
/// stream. Matches `compresso.decompress(stream, z=(start, end))`.
///
/// # Errors
///
/// The errors of [`decompress`], plus [`Error::InvalidZRange`] for ranges
/// outside `0..=sz`, and [`Error::RandomAccessUnavailable`] /
/// [`Error::RandomAccessConnectivity6`] for partial ranges of streams
/// without a usable z index.
pub fn decompress_z_range<T: Label>(stream: &[u8], z: Range<usize>) -> Result<Vec<T>> {
    decode(stream, z)
}

/// Decompresses a stream of any label width into little-endian bytes,
/// `sx * sy * sz * data_width` bytes with x varying fastest.
///
/// # Errors
///
/// See [`decompress`].
pub fn decompress_bytes(stream: &[u8]) -> Result<Vec<u8>> {
    let header = Header::parse(stream)?;
    decompress_z_range_bytes(stream, 0..usize::from(header.shape[2]))
}

/// Like [`decompress_bytes`], restricted to the z slices in `z`.
///
/// # Errors
///
/// See [`decompress_z_range`].
pub fn decompress_z_range_bytes(stream: &[u8], z: Range<usize>) -> Result<Vec<u8>> {
    fn to_bytes<T: Label>(labels: Vec<T>) -> Result<Vec<u8>> {
        let len = labels.len() * T::WIDTH;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| Error::AllocationFailed { bytes: len })?;
        for label in labels {
            out.extend_from_slice(&label.to_u64().to_le_bytes()[..T::WIDTH]);
        }
        Ok(out)
    }
    match Header::parse(stream)?.data_width {
        1 => decode::<u8>(stream, z),
        2 => to_bytes(decode::<u16>(stream, z)?),
        4 => to_bytes(decode::<u32>(stream, z)?),
        _ => to_bytes(decode::<u64>(stream, z)?),
    }
}

fn decode<T: Label>(stream: &[u8], z: Range<usize>) -> Result<Vec<T>> {
    let header = Header::parse(stream)?;
    if usize::from(header.data_width) != T::WIDTH {
        return Err(Error::LabelWidthMismatch {
            stream: header.data_width,
            requested: T::WIDTH,
        });
    }
    // On 32-bit targets the volume may not be addressable; all products of
    // the dimensions below are bounded by this one.
    if usize::try_from(header.nbytes()).is_err() {
        return Err(Error::AllocationFailed { bytes: usize::MAX });
    }
    let [sx, sy, sz] = header.shape_usize();
    let Range {
        start: zstart,
        end: zend,
    } = z;
    if zstart > sz || zend > sz || zend < zstart {
        return Err(Error::InvalidZRange {
            start: zstart,
            end: zend,
            sz,
        });
    }
    if sx * sy * sz == 0 || zstart == zend {
        return Ok(Vec::new());
    }
    let partial = zstart > 0 || zend < sz;
    if partial && !header.has_z_index() {
        return Err(Error::RandomAccessUnavailable);
    }
    if partial && header.connectivity == 6 {
        return Err(Error::RandomAccessConnectivity6);
    }
    let window_voxels = header.window_voxels();
    if window_voxels == 0 || (window_voxels > 64 && header.value_size > 0) {
        return Err(Error::InvalidSteps {
            steps: header.steps,
        });
    }

    let sections = Sections::parse(stream)?;
    let real_sz = zend - zstart;
    let sxy = sx * sy;
    let voxels = sxy * real_sz;

    // Cumulative z index: components before / first location of each slice.
    let (component_offset, location_start) = match &sections.z_index {
        Some((components, locations)) => {
            let prefix = |table: &Table<'_>, n: usize| {
                table
                    .iter()
                    .take(n)
                    .fold(0u64, |acc, v| acc.wrapping_add(v))
            };
            let components_before = if zstart > 0 {
                prefix(components, zstart)
            } else {
                0
            };
            (components_before, prefix(locations, zstart + 1))
        }
        None => (0, 0),
    };
    let boundaries = decode_boundaries(&sections, zstart, zend)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(voxels)
        .map_err(|_| Error::AllocationFailed {
            bytes: voxels.saturating_mul(T::WIDTH),
        })?;
    decode_components(
        &sections,
        &boundaries,
        [sx, sy, real_sz],
        component_offset,
        &mut output,
    )?;
    decode_indeterminate_locations(
        &sections,
        &boundaries,
        [sx, sy, real_sz],
        location_start,
        &mut output,
    )?;
    Ok(output)
}

/// Expands the run-length encoded boundary windows of slices `zstart..zend`
/// into one byte per voxel (1 = boundary).
fn decode_boundaries(sections: &Sections<'_>, zstart: usize, zend: usize) -> Result<Vec<u8>> {
    let header = &sections.header;
    let [sx, sy, sz] = header.shape_usize();
    let [xs, ys, zs] = header.steps.map(usize::from);
    let sxy = sx * sy;
    let voxels = sxy * (zend - zstart);
    if sections.values.is_empty() {
        return alloc_zeroed::<u8>(voxels);
    }

    let nx = sx.div_ceil(xs);
    let ny = sy.div_ceil(ys);
    let nz = sz.div_ceil(zs);
    let nblocks = nx * ny * nz;

    // Indices into the window value table. Values above u32::MAX are clamped,
    // since the table never has that many entries.
    let mut windows = alloc_zeroed::<u32>(nblocks)?;
    let mut index = 0usize;
    for word in sections.windows.iter() {
        if word & 1 == 1 {
            index = index.saturating_add(usize::try_from(word >> 1).unwrap_or(usize::MAX));
        } else {
            let slot = windows
                .get_mut(index)
                .ok_or(Error::Corrupt("boundary window outside of the grid"))?;
            *slot = u32::try_from(word >> 1).unwrap_or(u32::MAX);
            index += 1;
        }
    }

    let values: Vec<u64> = sections.values.iter().collect();
    let mut boundaries = Vec::new();
    boundaries
        .try_reserve_exact(voxels)
        .map_err(|_| Error::AllocationFailed { bytes: voxels })?;
    for z in zstart..zend {
        let zblock = nx * ny * (z / zs);
        let zoffset = xs * ys * (z % zs);
        for y in 0..sy {
            let block = nx * (y / ys) + zblock;
            let shift = xs * (y % ys) + zoffset;
            for (xb, &value_index) in windows[block..block + nx].iter().enumerate() {
                let value = values
                    .get(value_index as usize)
                    .ok_or(Error::Corrupt("boundary window value out of range"))?
                    >> shift;
                // Expand the row's bits of the window to one byte per voxel.
                let n = xs.min(sx - xb * xs);
                let mut k = 0;
                while k < n {
                    let bytes = SPREAD[((value >> k) & 0xFF) as usize].to_le_bytes();
                    let m = (n - k).min(8);
                    boundaries.extend_from_slice(&bytes[..m]);
                    k += 8;
                }
            }
        }
    }
    Ok(boundaries)
}

/// `SPREAD[b]` holds bit `i` of `b` in byte `i` (little endian).
const SPREAD: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut i = 0;
        while i < 8 {
            table[b] |= (((b >> i) & 1) as u64) << (8 * i);
            i += 1;
        }
        b += 1;
    }
    table
};

/// Writes the component labels of all non-boundary voxels. Boundary voxels
/// receive the label of component `component_offset`, which the C++
/// implementation also writes before the locations pass overwrites them.
fn decode_components<T: Label>(
    sections: &Sections<'_>,
    boundaries: &[u8],
    [sx, sy, sz]: [usize; 3],
    component_offset: u64,
    output: &mut Vec<T>,
) -> Result<()> {
    let ids = &sections.ids;
    let id = |component: u64| -> Option<T> {
        if component == 0 {
            Some(T::default())
        } else {
            ids.get(usize::try_from(component - 1).ok()?)
                .map(T::from_u64)
        }
    };
    let boundary_label = id(component_offset);
    let missing = Error::Corrupt("more components than component ids");
    let mut uf = UnionFind::new();
    let mut runs = Runs::default();
    let mut numbering = Vec::new();
    let mut lut: Vec<T> = Vec::new();

    // Labels one block (a slice, or the whole volume with 6-connectivity) and
    // writes it. Returns the number of components in the block.
    let mut fill =
        |boundaries: &[u8], shape: [usize; 3], six: bool, out: &mut Vec<T>, base: u64| {
            uf.reset();
            label_runs(boundaries, shape, six, &mut runs, &mut uf, |_| {})?;
            let count = uf.resolve(&mut numbering)?;
            lut.clear();
            lut.push(T::default());
            for &number in &numbering[1..] {
                lut.push(id(base + u64::from(number)).ok_or(missing.clone())?);
            }
            let sx = shape[0];
            for r in 0..runs.rows.len() - 1 {
                let mut x = 0;
                for run in runs.row(r) {
                    let (start, end) = (run.start as usize, run.end as usize);
                    if start > x {
                        let label = boundary_label.ok_or_else(|| missing.clone())?;
                        out.resize(out.len() + start - x, label);
                    }
                    out.resize(out.len() + end - start, lut[run.label as usize]);
                    x = end;
                }
                if x < sx {
                    let label = boundary_label.ok_or_else(|| missing.clone())?;
                    out.resize(out.len() + sx - x, label);
                }
            }
            Ok::<u64, Error>(u64::from(count))
        };

    if sections.header.connectivity == 6 {
        fill(boundaries, [sx, sy, sz], true, output, component_offset)?;
    } else {
        let sxy = sx * sy;
        let mut base = component_offset;
        for z in 0..sz {
            let range = z * sxy..(z + 1) * sxy;
            base += fill(&boundaries[range], [sx, sy, 1], false, output, base)?;
        }
    }
    Ok(())
}

/// Resolves the labels of the boundary voxels: either copied from the
/// preceding non-boundary neighbor or read from the locations instructions.
fn decode_indeterminate_locations<T: Label>(
    sections: &Sections<'_>,
    boundaries: &[u8],
    [sx, sy, sz]: [usize; 3],
    location_start: u64,
    labels: &mut [T],
) -> Result<()> {
    let six = sections.header.connectivity == 6;
    let locations = &sections.locations;
    let sxy = sx * sy;
    let mut index = usize::try_from(location_start).unwrap_or(usize::MAX);
    let read = |index: usize| {
        locations
            .get(index)
            .ok_or(Error::Corrupt("location index out of range"))
    };

    for z in 0..sz {
        for y in 0..sy {
            let base = sx * y + sxy * z;
            let row = &boundaries[base..base + sx];
            let mut next = 0;
            while let Some(x) = next_set(row, next) {
                next = x + 1;
                let loc = base + x;
                if x > 0 && boundaries[loc - 1] == 0 {
                    labels[loc] = labels[loc - 1];
                    continue;
                }
                if y > 0 && boundaries[loc - sx] == 0 {
                    labels[loc] = labels[loc - sx];
                    continue;
                }
                if six && z > 0 && boundaries[loc - sxy] == 0 {
                    labels[loc] = labels[loc - sxy];
                    continue;
                }
                if locations.is_empty() {
                    return Err(Error::NoLocations);
                }
                let offset = read(index)?;
                labels[loc] = match offset {
                    0 if x > 0 => labels[loc - 1],
                    1 if x + 1 < sx => labels[loc + 1],
                    2 if y > 0 => labels[loc - sx],
                    3 if y + 1 < sy => labels[loc + sx],
                    4 if z > 0 => labels[loc - sxy],
                    5 if z + 1 < sz => labels[loc + sxy],
                    0..=5 => {
                        return Err(Error::InvalidLocation {
                            offset: offset as u8,
                        });
                    }
                    6 => {
                        index += 1;
                        T::from_u64(read(index)?)
                    }
                    _ => T::from_u64(offset - 7),
                };
                index += 1;
            }
        }
    }
    Ok(())
}
