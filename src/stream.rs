//! Operations on encoded streams that do not need to decompress the volume.

use crate::decode::Sections;
use crate::error::{Error, Result};
use crate::header::Header;
use crate::label::{Table, write_le};

/// Decodes the labels carried by the location instructions
/// (6 is followed by a literal label, values above 6 encode `label + 7`).
fn location_labels(locations: &Table<'_>, mut f: impl FnMut(usize, u64, bool)) -> Result<()> {
    let mut i = 0;
    while i < locations.len() {
        let code = locations.get(i).unwrap_or(0);
        if code == 6 {
            i += 1;
            let label = locations
                .get(i)
                .ok_or(Error::Corrupt("literal location without a label"))?;
            f(i, label, true);
        } else if code > 6 {
            f(i, code - 7, false);
        }
        i += 1;
    }
    Ok(())
}

/// Returns the sorted, distinct labels in the stream without decompressing it.
///
/// Corresponds to `compresso.labels(stream)`.
///
/// # Errors
///
/// Returns an error if the header is invalid or the sections are truncated.
pub fn labels(stream: &[u8]) -> Result<Vec<u64>> {
    let sections = Sections::parse(stream)?;
    let mut labels: Vec<u64> = sections.ids.iter().collect();
    location_labels(&sections.locations, |_, label, _| labels.push(label))?;
    labels.sort_unstable();
    labels.dedup();
    Ok(labels)
}

/// Rewrites the labels of a stream without decompressing it.
///
/// `mapping` returns the new label for an existing one. If it returns `None`,
/// the label is kept when `preserve_missing_labels` is `true`, otherwise
/// [`Error::MissingLabel`] is returned. Mapped labels must fit into the label
/// width of the stream ([`Error::LabelOverflow`]).
///
/// Corresponds to `compresso.remap(stream, mapping, preserve_missing_labels)`
/// and produces the same bytes. One difference: when a label stored inline
/// as `label + 7` is mapped above `MAX - 7`, Python fails. This function
/// rewrites that entry as a literal escape and adjusts the header and z index.
///
/// # Errors
///
/// [`Error::MissingLabel`] and [`Error::LabelOverflow`] as described above,
/// or an error if the stream is invalid.
///
/// ```
/// use compresso::{compress, decompress, remap, CompressOptions};
///
/// let labels = vec![1u16, 1, 2, 2, 3, 3];
/// let stream = compress(&labels, [6, 1, 1], &CompressOptions::default())?;
/// let remapped = remap(&stream, |l| (l == 2).then_some(20), true)?;
/// assert_eq!(decompress::<u16>(&remapped)?, vec![1, 1, 20, 20, 3, 3]);
/// # Ok::<(), compresso::Error>(())
/// ```
pub fn remap(
    stream: &[u8],
    mut mapping: impl FnMut(u64) -> Option<u64>,
    preserve_missing_labels: bool,
) -> Result<Vec<u8>> {
    let sections = Sections::parse(stream)?;
    let width = usize::from(sections.header.data_width);
    let max = u64::MAX >> (64 - 8 * width);
    let mut map = |label: u64| -> Result<u64> {
        match mapping(label) {
            Some(new) if new > max => Err(Error::LabelOverflow(new)),
            Some(new) => Ok(new),
            None if preserve_missing_labels => Ok(label),
            None => Err(Error::MissingLabel(label)),
        }
    };

    let mut out = Vec::with_capacity(stream.len());
    out.extend_from_slice(&stream[..Header::SIZE]);
    for id in sections.ids.iter() {
        write_le(&mut out, map(id)?, width);
    }
    out.extend_from_slice(sections.values.bytes);

    // Rewrite the locations. A label whose `label + 7` no longer fits inline
    // becomes a literal escape (6, label), which lengthens the section, so
    // the header's location_size and the z index are updated as well.
    // (Python raises an OverflowError in that case.)
    let source = sections.locations;
    let mut literal_at = vec![false; source.len()];
    location_labels(&source, |i, _, literal| literal_at[i] = literal)?;
    let mut locations: Vec<u64> = Vec::with_capacity(source.len());
    // Number of entries inserted before each source index.
    let mut inserted_before: Vec<u64> = Vec::with_capacity(source.len() + 1);
    for (i, code) in source.iter().enumerate() {
        inserted_before.push((locations.len() - i) as u64);
        if literal_at[i] {
            locations.push(map(code)?);
        } else if code > 6 {
            let new = map(code - 7)?;
            if new <= max - 7 {
                locations.push(new + 7);
            } else {
                locations.push(6);
                locations.push(new);
            }
        } else {
            locations.push(code);
        }
    }
    inserted_before.push((locations.len() - source.len()) as u64);

    let mut header = sections.header;
    header.location_size = locations.len() as u64;
    out[..Header::SIZE].copy_from_slice(&header.to_bytes());
    for location in &locations {
        write_le(&mut out, *location, width);
    }
    match &sections.z_index {
        Some((components, z_index)) if locations.len() != source.len() => {
            out.extend_from_slice(&stream[sections.tail_start..sections.index_start]);
            out.extend_from_slice(components.bytes);
            // Shift each slice's first location by the entries inserted before it.
            let mut old_start = 0u64;
            let mut previous_new = 0u64;
            let index_width = z_index.width;
            for stored in z_index.iter() {
                old_start = old_start.wrapping_add(stored);
                let shift = usize::try_from(old_start)
                    .ok()
                    .and_then(|i| inserted_before.get(i))
                    .copied()
                    .ok_or(Error::Corrupt("z index points past the locations"))?;
                let new_start = old_start + shift;
                write_le(&mut out, new_start.wrapping_sub(previous_new), index_width);
                previous_new = new_start;
            }
            let index_end = sections.index_start + components.bytes.len() + z_index.bytes.len();
            out.extend_from_slice(&stream[index_end..]);
        }
        _ => out.extend_from_slice(&stream[sections.tail_start..]),
    }
    Ok(out)
}

/// Checks whether `stream` looks like a valid Compresso stream: a valid header
/// and enough bytes for the sections it declares.
///
/// Corresponds to `compresso.valid(stream)`, but returns `false` instead of
/// raising for unsupported format versions.
#[must_use]
pub fn valid(stream: &[u8]) -> bool {
    Sections::parse(stream).is_ok()
}
