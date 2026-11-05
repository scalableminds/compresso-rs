//! A pure Rust implementation of the **Compresso** codec for segmentation
//! volumes.
//!
//! > B. Matejek, D. Haehn, F. Lekschas, M. Mitzenmacher, H. Pfister.
//! > *Compresso: Efficient Compression of Segmentation Data for
//! > Connectomics.* MICCAI 2017.
//!
//! This crate reads and writes the stream format of the
//! [`compresso`](https://pypi.org/project/compresso/) Python package
//! (formats 0 and 1, magic `cpso`), which is also used by Neuroglancer
//! precomputed volumes with `"encoding": "compresso"`. [`compress`] produces
//! byte-identical output to `compresso.compress`, and [`decompress`] returns the
//! same labels as `compresso.decompress`. There is no C/C++ code and no
//! `unsafe`.
//!
//! # Layout
//!
//! Volumes are flat slices in Fortran order, with x varying fastest, then y,
//! then z. This matches `np.asfortranarray(volume).ravel(order="F")` for a
//! NumPy array of shape `(sx, sy, sz)`, and the layout of Neuroglancer chunks.
//!
//! # Example
//!
//! ```
//! use compresso::{compress, decompress, CompressOptions, Connectivity, Header};
//!
//! let shape = [64, 64, 8];
//! let volume: Vec<u64> = (0..64 * 64 * 8u64).map(|i| (i % 64) / 20 + 100 * (i / 4096)).collect();
//!
//! // Same defaults as `compresso.compress(volume)` in Python.
//! let stream = compress(&volume, shape, &CompressOptions::default())?;
//! assert_eq!(decompress::<u64>(&stream)?, volume);
//!
//! // Explicit parameters, as in `compresso.compress(volume, steps=(8, 8, 1), connectivity=6)`.
//! let options = CompressOptions::default().steps([8, 8, 1]).connectivity(Connectivity::Six);
//! let stream = compress(&volume, shape, &options)?;
//! let header = Header::parse(&stream)?;
//! assert_eq!(header.shape, [64, 64, 8]);
//! assert_eq!(header.data_width, 8);
//! # Ok::<(), compresso::Error>(())
//! ```
//!
//! # How it works
//!
//! A voxel is a *boundary* voxel if its label differs from its +x or +y
//! neighbor (or its +z neighbor with 6-connectivity). The binary boundary map
//! is cut into small windows (e.g. 4×4×1) that are stored as bit masks,
//! deduplicated and run-length encoded. The connected components of the
//! non-boundary voxels are numbered in raster order and stored as one label per
//! component. Boundary voxels are mostly decoded from an already decoded
//! neighbor. The remaining ones get an explicit "indeterminate location"
//! instruction.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod ccl;
mod decode;
mod encode;
mod error;
mod header;
mod label;
mod stream;

pub use decode::{decompress, decompress_bytes, decompress_z_range, decompress_z_range_bytes};
pub use encode::{CompressOptions, Connectivity, compress};
pub use error::{Error, Result};
pub use header::Header;
pub use label::Label;
pub use stream::{labels, remap, valid};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
