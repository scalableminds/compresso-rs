use std::fmt;

/// Errors returned while encoding or decoding a Compresso stream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The label slice does not contain `shape[0] * shape[1] * shape[2]` elements.
    ShapeMismatch {
        /// The requested volume shape.
        shape: [usize; 3],
        /// The number of labels that were supplied.
        len: usize,
    },
    /// A dimension does not fit into the 16-bit size fields of the header.
    DimensionTooLarge {
        /// The requested volume shape.
        shape: [usize; 3],
    },
    /// The grid steps contain a zero or describe a window larger than 64 voxels.
    InvalidSteps {
        /// The offending steps.
        steps: [u8; 3],
    },
    /// The number of distinct boundary windows exceeds what the window integer
    /// type can run-length encode. Use a window of 64 voxels such as `[8, 8, 1]`.
    TooManyWindows,
    /// The volume has more connected components than can be represented.
    TooManyComponents,
    /// The input is shorter than the 36 byte header.
    TooShort {
        /// Length of the input.
        len: usize,
    },
    /// The stream does not start with the `cpso` magic number.
    InvalidMagic,
    /// The stream uses a format version other than 0 or 1.
    UnsupportedVersion(u8),
    /// The stream declares a label width other than 1, 2, 4 or 8 bytes.
    InvalidDataWidth(u8),
    /// The stream declares a connectivity other than 4 or 6.
    InvalidConnectivity(u8),
    /// The requested label type does not match the label width of the stream.
    LabelWidthMismatch {
        /// Label width in bytes declared by the stream.
        stream: u8,
        /// Label width in bytes of the requested type.
        requested: usize,
    },
    /// The requested z range is outside the volume.
    InvalidZRange {
        /// First requested slice.
        start: usize,
        /// One past the last requested slice.
        end: usize,
        /// Number of slices in the volume.
        sz: usize,
    },
    /// Partial z decoding requires a format version 1 stream (with z index).
    RandomAccessUnavailable,
    /// Partial z decoding is not supported for 6-connected streams.
    RandomAccessConnectivity6,
    /// The section sizes declared in the header exceed the input.
    Truncated,
    /// A boundary voxel requires an indeterminate location but the stream has none.
    NoLocations,
    /// An indeterminate location points to a neighbor outside the volume.
    ///
    /// `offset` is the location code: 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z.
    InvalidLocation {
        /// The offending location code.
        offset: u8,
    },
    /// The stream is internally inconsistent.
    Corrupt(&'static str),
    /// A label that should be remapped is missing from the mapping.
    MissingLabel(u64),
    /// A remapped label does not fit into the label type of the stream.
    LabelOverflow(u64),
    /// Allocating the output or a working buffer failed.
    AllocationFailed {
        /// Number of bytes that could not be allocated.
        bytes: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { shape, len } => write!(
                f,
                "compresso: shape {shape:?} requires {} labels, got {len}",
                shape.iter().map(|&s| s as u128).product::<u128>()
            ),
            Error::DimensionTooLarge { shape } => write!(
                f,
                "compresso: dimensions must not exceed 65535, got {shape:?}"
            ),
            Error::InvalidSteps { steps } => write!(
                f,
                "compresso: steps must be non-zero and span at most 64 voxels, got {steps:?}"
            ),
            Error::TooManyWindows => f.write_str(
                "compresso: Unable to RLE encode. Too many windows. Use 64-bit steps e.g. (8,8,1) instead.",
            ),
            Error::TooManyComponents => f.write_str("compresso: too many connected components"),
            Error::TooShort { len } => write!(
                f,
                "compresso: Input too small to be a valid stream. Bytes: {len}"
            ),
            Error::InvalidMagic => f.write_str("compresso: Data stream is not valid (bad magic)."),
            Error::UnsupportedVersion(v) => write!(
                f,
                "compresso: Unable to decode format version {v}. Only versions 0 and 1 are supported."
            ),
            Error::InvalidDataWidth(w) => write!(
                f,
                "compresso: Invalid data width in stream. Unable to decompress. Got: {w}"
            ),
            Error::InvalidConnectivity(c) => write!(
                f,
                "compresso: Invalid connectivity in stream. Unable to decompress. Got: {c}"
            ),
            Error::LabelWidthMismatch { stream, requested } => write!(
                f,
                "compresso: stream has {stream} byte labels, requested {requested} byte labels"
            ),
            Error::InvalidZRange { start, end, sz } => write!(
                f,
                "compresso: Invalid decode range: {start}-{end} (sz = {sz})"
            ),
            Error::RandomAccessUnavailable => {
                f.write_str("compresso: Cannot random access z slices without the index.")
            }
            Error::RandomAccessConnectivity6 => {
                f.write_str("compresso: Connectivity 6 does not support random access.")
            }
            Error::Truncated => f.write_str("compresso: section sizes exceed the input"),
            Error::NoLocations => f.write_str(
                "compresso: unable to decode indeterminate locations. (no locations)",
            ),
            Error::InvalidLocation { offset } => write!(
                f,
                "compresso: unable to decode indeterminate locations. (offset {offset})"
            ),
            Error::Corrupt(reason) => write!(f, "compresso: corrupt stream: {reason}"),
            Error::MissingLabel(label) => write!(f, "compresso: label {label} missing from mapping"),
            Error::LabelOverflow(label) => write!(
                f,
                "compresso: label {label} does not fit into the label type of the stream"
            ),
            Error::AllocationFailed { bytes } => {
                write!(f, "compresso: unable to allocate {bytes} bytes")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias for results of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
