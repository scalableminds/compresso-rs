use std::fmt::Debug;
use std::hash::Hash;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Unsigned integer types that can be used as segmentation labels:
/// `u8`, `u16`, `u32` and `u64`.
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait Label:
    sealed::Sealed + Copy + Default + Eq + Ord + Hash + Debug + Send + Sync + 'static
{
    /// Width of the label in bytes.
    const WIDTH: usize;
    /// Largest representable label.
    const MAX: u64;
    /// Widens the label to `u64`.
    fn to_u64(self) -> u64;
    /// Narrows a `u64` to the label type, truncating like a C cast.
    fn from_u64(value: u64) -> Self;
}

macro_rules! impl_label {
    ($($t:ty),*) => {$(
        impl Label for $t {
            const WIDTH: usize = std::mem::size_of::<$t>();
            const MAX: u64 = <$t>::MAX as u64;
            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline]
            fn from_u64(value: u64) -> Self {
                value as $t
            }
        }
    )*};
}
impl_label!(u8, u16, u32, u64);

/// Reads a little-endian unsigned integer of `bytes.len()` (at most 8) bytes.
#[inline]
pub(crate) fn read_le(bytes: &[u8]) -> u64 {
    let mut value = [0u8; 8];
    value[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(value)
}

/// Appends the lowest `width` bytes of `value` in little-endian order.
#[inline]
pub(crate) fn write_le(out: &mut Vec<u8>, value: u64, width: usize) {
    out.extend_from_slice(&value.to_le_bytes()[..width]);
}

/// A view of a little-endian integer table inside a stream.
#[derive(Clone, Copy)]
pub(crate) struct Table<'a> {
    pub bytes: &'a [u8],
    pub width: usize,
}

impl<'a> Table<'a> {
    pub fn len(&self) -> usize {
        self.bytes.len() / self.width
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        let start = index.checked_mul(self.width)?;
        let end = start.checked_add(self.width)?;
        self.bytes.get(start..end).map(read_le)
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> + 'a {
        self.bytes.chunks_exact(self.width).map(read_le)
    }
}
