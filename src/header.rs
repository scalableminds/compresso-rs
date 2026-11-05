use crate::error::{Error, Result};
use crate::label::read_le;

/// The Compresso stream header (36 bytes, little endian).
///
/// | offset | size | field             |
/// |-------:|-----:|-------------------|
/// |      0 |    4 | magic `cpso`      |
/// |      4 |    1 | format version    |
/// |      5 |    1 | label width (bytes) |
/// |      6 |  2×3 | sx, sy, sz        |
/// |     12 |  1×3 | xstep, ystep, zstep |
/// |     15 |    8 | id_size           |
/// |     23 |    4 | value_size        |
/// |     27 |    8 | location_size     |
/// |     35 |    1 | connectivity      |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header {
    /// 0: no z index, 1: random access z index appended to the stream.
    pub format_version: u8,
    /// Label width in bytes (1, 2, 4 or 8).
    pub data_width: u8,
    /// Volume shape `[sx, sy, sz]` (x is the fastest varying axis).
    pub shape: [u16; 3],
    /// Boundary window size `[xstep, ystep, zstep]`.
    pub steps: [u8; 3],
    /// Number of connected component labels.
    pub id_size: u64,
    /// Number of distinct boundary window values.
    pub value_size: u32,
    /// Number of indeterminate location entries.
    pub location_size: u64,
    /// Connectivity used for connected components (4 or 6).
    pub connectivity: u8,
}

impl Header {
    /// Size of the serialized header in bytes.
    pub const SIZE: usize = 36;
    /// Magic number at the start of every stream.
    pub const MAGIC: [u8; 4] = *b"cpso";

    /// Parses and validates the header at the start of `stream`.
    ///
    /// Validates the magic number, format version, label width and
    /// connectivity. The grid steps are validated during decoding.
    ///
    /// # Errors
    ///
    /// [`Error::TooShort`], [`Error::InvalidMagic`],
    /// [`Error::UnsupportedVersion`], [`Error::InvalidDataWidth`] or
    /// [`Error::InvalidConnectivity`].
    pub fn parse(stream: &[u8]) -> Result<Self> {
        if stream.len() < Self::SIZE {
            return Err(Error::TooShort { len: stream.len() });
        }
        if stream[..4] != Self::MAGIC {
            return Err(Error::InvalidMagic);
        }
        let header = Self::parse_unchecked(stream);
        if header.format_version > 1 {
            return Err(Error::UnsupportedVersion(header.format_version));
        }
        if !matches!(header.data_width, 1 | 2 | 4 | 8) {
            return Err(Error::InvalidDataWidth(header.data_width));
        }
        if !matches!(header.connectivity, 4 | 6) {
            return Err(Error::InvalidConnectivity(header.connectivity));
        }
        Ok(header)
    }

    fn parse_unchecked(b: &[u8]) -> Self {
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        Header {
            format_version: b[4],
            data_width: b[5],
            shape: [u16_at(6), u16_at(8), u16_at(10)],
            steps: [b[12], b[13], b[14]],
            id_size: read_le(&b[15..23]),
            value_size: read_le(&b[23..27]) as u32,
            location_size: read_le(&b[27..35]),
            connectivity: b[35],
        }
    }

    /// Serializes the header into its 36 byte representation.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[..4].copy_from_slice(&Self::MAGIC);
        out[4] = self.format_version;
        out[5] = self.data_width;
        for (i, s) in self.shape.iter().enumerate() {
            out[6 + 2 * i..8 + 2 * i].copy_from_slice(&s.to_le_bytes());
        }
        out[12..15].copy_from_slice(&self.steps);
        out[15..23].copy_from_slice(&self.id_size.to_le_bytes());
        out[23..27].copy_from_slice(&self.value_size.to_le_bytes());
        out[27..35].copy_from_slice(&self.location_size.to_le_bytes());
        out[35] = self.connectivity;
        out
    }

    /// Volume shape as `usize`s.
    #[must_use]
    pub fn shape_usize(&self) -> [usize; 3] {
        self.shape.map(usize::from)
    }

    /// Number of voxels in the volume.
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.shape.iter().map(|&s| u64::from(s)).product()
    }

    /// Number of bytes of the fully decompressed volume.
    #[must_use]
    pub fn nbytes(&self) -> u64 {
        self.voxels() * u64::from(self.data_width)
    }

    /// Whether the stream carries the random access z index (format version 1).
    #[must_use]
    pub fn has_z_index(&self) -> bool {
        self.format_version == 1
    }

    /// Number of voxels covered by one boundary window.
    #[must_use]
    pub fn window_voxels(&self) -> usize {
        self.steps.iter().map(|&s| usize::from(s)).product()
    }

    /// Width in bytes of one boundary window (1, 2, 4 or 8).
    #[must_use]
    pub fn window_width(&self) -> usize {
        window_width(self.window_voxels())
    }

    /// Width in bytes of each z index entry (format version 1 only).
    #[must_use]
    pub fn index_width(&self) -> usize {
        index_width(usize::from(self.shape[0]), usize::from(self.shape[1]))
    }
}

pub(crate) fn window_width(window_voxels: usize) -> usize {
    match window_voxels {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
}

/// Mirrors `CompressoHeader::index_byte_width` (note the strict comparison
/// against each integer maximum).
pub(crate) fn index_width(sx: usize, sy: usize) -> usize {
    let worst_case = 2 * (sx as u64) * (sy as u64);
    if worst_case < u64::from(u8::MAX) {
        1
    } else if worst_case < u64::from(u16::MAX) {
        2
    } else if worst_case < u64::from(u32::MAX) {
        4
    } else {
        8
    }
}
