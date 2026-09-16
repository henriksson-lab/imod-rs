//! Owned translation of `IMOD/raptor/lasik/svl/lib/base/svlBitArray.{h,cpp}`.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

const INT_BIT_SIZE: usize = u32::BITS as usize;

/// SVL's packed bitmap.  `Vec<u32>` replaces the source's manually owned
/// `int *`; padding bits retain the source behaviour for `set_all`, `flip_all`,
/// and `count`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SvlBitArray {
    size: usize,
    map: Vec<u32>,
}

impl SvlBitArray {
    /// C++ default constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ size constructor.
    pub fn with_size(size: usize) -> Self {
        Self {
            size,
            map: vec![0; size.div_ceil(INT_BIT_SIZE)],
        }
    }

    /// C++ `size`.
    pub fn size(&self) -> usize {
        self.size
    }

    /// C++ `operator[]` and `get`.
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.size, "bit index is out of bounds");
        self.map[index / INT_BIT_SIZE] & (1_u32 << (index % INT_BIT_SIZE)) != 0
    }

    /// C++ `set`.
    pub fn set(&mut self, index: usize) {
        assert!(index < self.size, "bit index is out of bounds");
        self.map[index / INT_BIT_SIZE] |= 1_u32 << (index % INT_BIT_SIZE);
    }

    /// C++ `clear`.
    pub fn clear(&mut self, index: usize) {
        assert!(index < self.size, "bit index is out of bounds");
        self.map[index / INT_BIT_SIZE] &= !(1_u32 << (index % INT_BIT_SIZE));
    }

    /// C++ `flip`.
    pub fn flip(&mut self, index: usize) {
        assert!(index < self.size, "bit index is out of bounds");
        self.map[index / INT_BIT_SIZE] ^= 1_u32 << (index % INT_BIT_SIZE);
    }

    /// C++ `setAll`. This deliberately sets padding bits too.
    pub fn set_all(&mut self) {
        self.map.fill(u32::MAX);
    }

    /// C++ `clearAll`.
    pub fn clear_all(&mut self) {
        self.map.fill(0);
    }

    /// C++ `flipAll`. This deliberately flips padding bits too.
    pub fn flip_all(&mut self) {
        for word in &mut self.map {
            *word = !*word;
        }
    }

    /// C++ `copy`. Rust assignment/clone is the source equivalent without
    /// retaining the C implementation's stale-size allocation bug.
    pub fn copy(&mut self, source: &Self) {
        self.size = source.size;
        self.map.clone_from(&source.map);
    }

    /// C++ `count`, including allocated padding bits after `set_all` or
    /// `flip_all`, exactly as its lookup-table loop does.
    pub fn count(&self) -> usize {
        self.map.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// C++ `save`, writing one native boolean byte per logical bit.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = File::create(path)?;
        for index in 0..self.size {
            file.write_all(&[u8::from(self.get(index))])?;
        }
        Ok(())
    }

    /// C++ `print`, returning the exact text that its stream overload emits.
    pub fn print(&self, stride: Option<usize>) -> String {
        let mut output = String::new();
        for index in 0..self.size {
            output.push(if self.get(index) { '1' } else { '0' });
            if stride.is_some_and(|stride| stride > 0 && (index + 1) % stride == 0) {
                output.push('\n');
            }
        }
        if !stride.is_some_and(|stride| stride > 0 && self.size % stride == 0) {
            output.push('\n');
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::SvlBitArray;

    #[test]
    fn source_bit_operations_and_padding_count_match() {
        let mut bits = SvlBitArray::with_size(35);
        bits.set(0);
        bits.set(34);
        bits.flip(1);
        bits.clear(1);
        assert_eq!(bits.count(), 2);
        bits.set_all();
        assert_eq!(bits.count(), 64);
        bits.flip_all();
        assert_eq!(bits.count(), 0);
    }

    #[test]
    fn copy_save_and_print_follow_source_layout() {
        let mut bits = SvlBitArray::with_size(5);
        bits.set(0);
        bits.set(3);
        let mut copied = SvlBitArray::new();
        copied.copy(&bits);
        assert_eq!(copied, bits);
        assert_eq!(bits.print(Some(2)), "10\n01\n0\n");
        assert_eq!(bits.print(None), "10010\n");
    }
}
