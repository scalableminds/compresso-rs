//! Connected component labeling of the non-boundary voxels.
//!
//! Compresso numbers the connected components of the non-boundary voxels in
//! the order of their first occurrence in a raster scan (x fastest, then y,
//! then z). With 4-connectivity, each z slice is labeled independently and the
//! numbering continues across slices. With 6-connectivity, the whole volume is
//! labeled at once.
//!
//! This is a two-pass union-find labeling. Provisional labels are allocated in
//! raster order, and a union always links the larger root under the smaller
//! one. Each set's root is therefore its first provisional label, so resolving
//! the roots in increasing order gives first-occurrence numbering.

use crate::error::{Error, Result};

pub(crate) struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    pub fn new() -> Self {
        UnionFind { parent: vec![0] }
    }

    pub fn reset(&mut self) {
        self.parent.clear();
        self.parent.push(0);
    }

    /// Number of provisional labels allocated so far.
    pub fn len(&self) -> usize {
        self.parent.len() - 1
    }

    #[inline]
    fn make(&mut self) -> Result<u32> {
        let label = u32::try_from(self.parent.len()).map_err(|_| Error::TooManyComponents)?;
        if label == u32::MAX {
            return Err(Error::TooManyComponents);
        }
        self.parent
            .try_reserve(1)
            .map_err(|_| Error::AllocationFailed {
                bytes: (self.parent.len() + 1) * 4,
            })?;
        self.parent.push(label);
        Ok(label)
    }

    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize];
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
    }

    #[inline]
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra < rb {
            self.parent[rb as usize] = ra;
        } else if rb < ra {
            self.parent[ra as usize] = rb;
        }
    }

    /// Fills `out[l]` with the final 1-based component number of every
    /// provisional label `l` and returns the number of components.
    pub fn resolve(&mut self, out: &mut Vec<u32>) -> Result<u32> {
        let n = self.len();
        out.clear();
        out.try_reserve(n + 1)
            .map_err(|_| Error::AllocationFailed { bytes: (n + 1) * 4 })?;
        out.push(0);
        let mut next = 0u32;
        for label in 1..=n as u32 {
            let root = self.find(label);
            if root == label {
                next += 1;
                out.push(next);
            } else {
                // The root is smaller than `label`, so it is already numbered.
                let number = out[root as usize];
                out.push(number);
            }
        }
        Ok(next)
    }

    /// Calls `f` with each root label in increasing order (the components in
    /// first-occurrence order).
    pub fn for_each_root(&mut self, mut f: impl FnMut(u32)) {
        for label in 1..=self.len() as u32 {
            if self.find(label) == label {
                f(label);
            }
        }
    }
}

/// A run `start..end` of non-boundary voxels within one row (x range).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub start: u32,
    pub end: u32,
    pub label: u32,
}

/// The runs of non-boundary voxels of a block, row by row. The runs of row
/// `r = y + sy * z` are `runs[rows[r]..rows[r + 1]]`, sorted by x.
#[derive(Default)]
pub(crate) struct Runs {
    pub runs: Vec<Run>,
    pub rows: Vec<usize>,
}

impl Runs {
    pub fn row(&self, r: usize) -> &[Run] {
        &self.runs[self.rows[r]..self.rows[r + 1]]
    }
}

/// Connects the run `start..end` to the overlapping runs in `candidates`
/// (a previous row, sorted by x). `cursor` is advanced monotonically.
#[inline]
fn connect(
    candidates: &[Run],
    cursor: &mut usize,
    start: u32,
    end: u32,
    label: &mut u32,
    uf: &mut UnionFind,
) {
    while *cursor < candidates.len() && candidates[*cursor].end <= start {
        *cursor += 1;
    }
    let mut j = *cursor;
    while j < candidates.len() && candidates[j].start < end {
        let other = candidates[j].label;
        if *label == 0 {
            *label = other;
        } else if *label != other {
            uf.union(*label, other);
        }
        j += 1;
    }
}

/// First pass of the labeling. It collects the runs of non-boundary voxels
/// (`boundaries[i] == 0`) of a block of `sz` slices and gives them
/// provisional labels. Runs that overlap a run of the previous row (or, with
/// `six`, the same row of the previous slice) are joined. With
/// `six == false`, z neighbors are ignored, so pass a single slice.
///
/// `on_new(i)` is called with the voxel index of the first voxel of every
/// run that creates a new provisional label. Labels are created in raster
/// order of the runs' first voxels, so the smallest label of each component
/// belongs to its first voxel in raster order.
pub(crate) fn label_runs(
    boundaries: &[u8],
    [sx, sy, sz]: [usize; 3],
    six: bool,
    out: &mut Runs,
    uf: &mut UnionFind,
    mut on_new: impl FnMut(usize),
) -> Result<()> {
    debug_assert_eq!(boundaries.len(), sx * sy * sz);
    out.runs.clear();
    out.rows.clear();
    out.rows.push(0);
    for r in 0..sy * sz {
        let (y, z) = (r % sy, r / sy);
        let row = &boundaries[r * sx..(r + 1) * sx];
        let up = if y > 0 {
            out.rows[r - 1]..out.rows[r]
        } else {
            0..0
        };
        let back = if six && z > 0 {
            out.rows[r - sy]..out.rows[r - sy + 1]
        } else {
            0..0
        };
        let (mut up_cursor, mut back_cursor) = (0, 0);
        let mut x = 0;
        while let Some(start) = next_clear(row, x) {
            x = next_set(row, start).unwrap_or(sx);
            let (s, e) = (start as u32, x as u32);
            let mut label = 0;
            connect(&out.runs[up.clone()], &mut up_cursor, s, e, &mut label, uf);
            connect(
                &out.runs[back.clone()],
                &mut back_cursor,
                s,
                e,
                &mut label,
                uf,
            );
            if label == 0 {
                label = uf.make()?;
                on_new(r * sx + start);
            }
            out.runs.push(Run {
                start: s,
                end: e,
                label,
            });
        }
        out.rows.push(out.runs.len());
    }
    Ok(())
}

const LOW_BITS: u64 = 0x0101_0101_0101_0101;

#[inline]
fn word(row: &[u8], x: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&row[x..x + 8]);
    u64::from_le_bytes(bytes)
}

/// Index of the first boundary voxel (`1`) at or after `x`. `row` holds only
/// zeros and ones.
#[inline]
pub(crate) fn next_set(row: &[u8], mut x: usize) -> Option<usize> {
    while x + 8 <= row.len() {
        let m = word(row, x) & LOW_BITS;
        if m != 0 {
            return Some(x + (m.trailing_zeros() / 8) as usize);
        }
        x += 8;
    }
    (x..row.len()).find(|&i| row[i] != 0)
}

/// Index of the first non-boundary voxel (`0`) at or after `x`. `row` holds
/// only zeros and ones.
#[inline]
pub(crate) fn next_clear(row: &[u8], mut x: usize) -> Option<usize> {
    while x + 8 <= row.len() {
        let m = !word(row, x) & LOW_BITS;
        if m != 0 {
            return Some(x + (m.trailing_zeros() / 8) as usize);
        }
        x += 8;
    }
    (x..row.len()).find(|&i| row[i] == 0)
}

pub(crate) fn alloc_zeroed<T: Copy + Default>(len: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::AllocationFailed {
            bytes: len.saturating_mul(std::mem::size_of::<T>()),
        })?;
    v.resize(len, T::default());
    Ok(v)
}
