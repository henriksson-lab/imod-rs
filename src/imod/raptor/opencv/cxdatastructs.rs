//! Owned Rust translation of `IMOD/raptor/opencv/cxdatastructs.cpp`.
//!
//! The original 4,000-line unit is an allocator plus pointer-based sequence,
//! set, graph, DFS scanner, and tree implementation.  Rust ownership removes
//! its block/link/free-list machinery: the public data structures below retain
//! their observable operations while `Vec`, `Option`, and stable slot indices
//! replace raw addresses and manual memory storage.

use std::collections::{HashSet, VecDeque};

/// `cvAlignLeft` (`cxdatastructs.cpp:45`).  As in the C macro family, callers
/// provide a positive power-of-two alignment and this rounds down rather than
/// up.  `wrapping_neg` preserves the C two's-complement bit operation.
pub fn cv_align_left(size: i32, align: i32) -> i32 {
    size & align.wrapping_neg()
}

/// Source `CV_STORAGE_BLOCK_SIZE`; `icvInitMemStorage` selects it for a
/// nonpositive requested size, then aligns the allocation boundary.
pub const CV_STORAGE_BLOCK_SIZE: usize = 64 * 1024;
const CV_STRUCT_ALIGN: usize = 8;

/// `icvInitMemStorage` (`cxdatastructs.cpp:83`).  The C function initializes
/// links, signature, and free-space counters; owned blocks make those links
/// unnecessary, while resetting the block policy has the same observable
/// allocation behavior.
pub fn icv_init_mem_storage(storage: &mut CvMemStorage, block_size: i32) {
    let requested = usize::try_from(block_size)
        .ok()
        .filter(|&size| size > 0)
        .unwrap_or(CV_STORAGE_BLOCK_SIZE);
    storage.block_size = requested.saturating_add(CV_STRUCT_ALIGN - 1) & !(CV_STRUCT_ALIGN - 1);
    storage.blocks.clear();
}

/// `icvDestroyMemStorage` (`cxdatastructs.cpp:153`).  Parent-block transfer is
/// a raw-pointer ownership optimization in C; each Rust storage owns its own
/// blocks, so destruction simply drops them.
pub fn icv_destroy_mem_storage(storage: &mut CvMemStorage) {
    storage.blocks.clear();
}

/// `icvGoNextMemBlock` (`cxdatastructs.cpp:258`).  The C allocator links a
/// fresh block as `top` and resets its free-space counter.  Each owned vector
/// is already a separate block here, so appending a source-sized zeroed block
/// is the equivalent state transition.
pub fn icv_go_next_mem_block(storage: &mut CvMemStorage) -> &mut [u8] {
    storage.allocate(storage.block_size)
}

/// C `CvMemStorage`, retaining byte allocations and checkpoint restoration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CvMemStorage {
    pub block_size: usize,
    blocks: Vec<Vec<u8>>,
}

/// C `CvMemStoragePos`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvMemStoragePos {
    blocks: usize,
}

impl CvMemStorage {
    /// C `cvCreateMemStorage`.
    pub fn new(block_size: usize) -> Self {
        let mut storage = Self::default();
        icv_init_mem_storage(&mut storage, i32::try_from(block_size).unwrap_or(i32::MAX));
        storage
    }

    /// C `cvCreateChildMemStorage`; Rust child storage owns its allocations.
    pub fn child(&self) -> Self {
        Self::new(self.block_size)
    }

    /// C `cvClearMemStorage` and `cvReleaseMemStorage`.
    pub fn clear(&mut self) {
        icv_destroy_mem_storage(self);
    }

    /// C `cvSaveMemStoragePos`.
    pub fn save_position(&self) -> CvMemStoragePos {
        CvMemStoragePos {
            blocks: self.blocks.len(),
        }
    }

    /// C `cvRestoreMemStoragePos`.
    pub fn restore_position(&mut self, position: CvMemStoragePos) {
        self.blocks.truncate(position.blocks);
    }

    /// C `cvMemStorageAlloc`; returns owned bytes rather than an invalidatable pointer.
    pub fn allocate(&mut self, size: usize) -> &mut [u8] {
        self.blocks.push(vec![0; size]);
        self.blocks.last_mut().unwrap()
    }

    /// C `cvMemStorageAllocString`.
    pub fn allocate_string(&mut self, value: &str) -> String {
        value.to_owned()
    }
}

/// C `CvSlice`, with negative offsets resolved against sequence length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvSlice {
    pub start: isize,
    pub end: isize,
}

/// Owned generic replacement for C `CvSeq` and its reader/writer APIs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CvSeq<T> {
    pub values: Vec<T>,
    pub block_size: usize,
}

/// Borrowed non-growing equivalent of the `CvSeq` header constructed by
/// `cvMakeSeqHeaderForArray`.  The C header stores a non-owning data pointer;
/// this lifetime-bound slice prevents the header surviving its backing array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CvSeqArrayHeader<'a, T> {
    pub values: &'a [T],
}

impl<'a, T> CvSeqArrayHeader<'a, T> {
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn get(&self, index: isize) -> Option<&'a T> {
        let index = if index < 0 {
            self.values.len().checked_sub(index.unsigned_abs())?
        } else {
            index as usize
        };
        self.values.get(index)
    }
}

/// `cvMakeSeqHeaderForArray` (`cxdatastructs.cpp:685`).  The C operation
/// makes a sequence header over external memory and explicitly prevents it
/// growing; this borrowed view preserves both properties.
pub fn cv_make_seq_header_for_array<T>(array: &[T]) -> CvSeqArrayHeader<'_, T> {
    CvSeqArrayHeader { values: array }
}

/// `icvGrowSeq` (`cxdatastructs.cpp:738`).  C prepares a free sequence block
/// before a writer stores the next item.  Rust does the same capacity step;
/// insertion direction only affects C block links, not contiguous `Vec`
/// storage.  `false` corresponds to the C allocation failure path.
pub fn icv_grow_seq<T>(sequence: &mut CvSeq<T>, _in_front: bool) -> bool {
    if sequence.values.len() < sequence.values.capacity() {
        return true;
    }
    sequence
        .values
        .try_reserve(sequence.block_size.max(1))
        .is_ok()
}

/// `icvFreeSeqBlock` (`cxdatastructs.cpp:867`).  The native implementation
/// unlinks an empty block and chains it on `free_blocks` for later reuse.  A
/// `Vec` retains its allocation after elements are removed, which is precisely
/// that reusable-block state; no pointer/link mutation is needed here.
pub fn icv_free_seq_block<T>(_sequence: &mut CvSeq<T>, _in_front_of: bool) {}

/// Typed equivalent of `CvSeqWriter`.  C exposes writable byte pointers and
/// relies on `cvFlushSeqWriter` to update sequence totals; staging values gives
/// the same explicit commit point without invalid references during growth.
pub struct CvSeqWriter<'a, T> {
    sequence: &'a mut CvSeq<T>,
    pending: Vec<T>,
}

impl<'a, T> CvSeqWriter<'a, T> {
    pub fn push(&mut self, value: T) {
        self.pending.push(value);
    }
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

/// `cvStartAppendToSeq` (`cxdatastructs.cpp:932`).
pub fn cv_start_append_to_seq<T>(sequence: &mut CvSeq<T>) -> CvSeqWriter<'_, T> {
    CvSeqWriter {
        sequence,
        pending: Vec::new(),
    }
}

/// Owning form of a source `cvStartWriteSeq` writer: C creates the sequence
/// in a storage arena and hands its writer back to the caller, whereas Rust
/// can own both until `finish` transfers the completed sequence.
pub struct CvSeqOwnedWriter<T> {
    sequence: CvSeq<T>,
    pending: Vec<T>,
}

impl<T> CvSeqOwnedWriter<T> {
    pub fn push(&mut self, value: T) {
        self.pending.push(value);
    }
    pub fn flush(&mut self) {
        for value in self.pending.drain(..) {
            self.sequence.push(value);
        }
    }
    pub fn finish(mut self) -> CvSeq<T> {
        self.flush();
        self.sequence
    }
}

/// `cvStartWriteSeq` (`cxdatastructs.cpp:955`).  C accepts type/header/element
/// byte metadata; Rust's generic `T` supplies that information, leaving the
/// source block-growth parameter as the relevant runtime setting.
pub fn cv_start_write_seq<T>(block_size: i32) -> Result<CvSeqOwnedWriter<T>, ()> {
    if block_size < 0 {
        return Err(());
    }
    let mut sequence = CvSeq::new();
    cv_set_seq_block_size(&mut sequence, block_size)?;
    Ok(CvSeqOwnedWriter {
        sequence,
        pending: Vec::new(),
    })
}

/// `cvFlushSeqWriter` (`cxdatastructs.cpp:976`).
pub fn cv_flush_seq_writer<T>(writer: &mut CvSeqWriter<'_, T>) {
    for value in writer.pending.drain(..) {
        writer.sequence.push(value);
    }
}

/// `cvCreateSeqBlock` (`cxdatastructs.cpp:1056`).  The C writer obtains its
/// next linked storage block here.  An owned sequence has one contiguous
/// backing allocation, so reserving the configured block is the equivalent
/// fallible growth transition before staging more elements.
pub fn cv_create_seq_block<T>(writer: &mut CvSeqWriter<'_, T>) -> Result<(), ()> {
    let needed = writer.sequence.block_size.max(1);
    writer.sequence.values.try_reserve(needed).map_err(|_| ())
}

/// `cvEndWriteSeq` (`cxdatastructs.cpp:1015`).  C returns the sequence after
/// flushing its final block; Rust returns the same borrowed sequence.
pub fn cv_end_write_seq<T>(mut writer: CvSeqWriter<'_, T>) -> &mut CvSeq<T> {
    cv_flush_seq_writer(&mut writer);
    writer.sequence
}

impl<T> CvSeq<T> {
    /// C `cvCreateSeq`.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            block_size: 1024,
        }
    }
    pub fn set_block_size(&mut self, elements: usize) {
        self.block_size = elements.max(1);
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// C `cvGetSeqElem`.
    pub fn get(&self, index: isize) -> Option<&T> {
        let index = if index < 0 {
            self.values.len().checked_sub(index.unsigned_abs())?
        } else {
            index as usize
        };
        self.values.get(index)
    }
    /// C `cvSliceLength`/`cvSeqSlice`.
    pub fn slice(&self, slice: CvSlice) -> Vec<&T> {
        let length = self.values.len() as isize;
        let normalize = |index: isize| if index < 0 { (length + index).max(0) } else { index.min(length) } as usize;
        let start = normalize(slice.start);
        let end = normalize(slice.end);
        if start <= end {
            self.values[start..end].iter().collect()
        } else {
            self.values[start..]
                .iter()
                .chain(self.values[..end].iter())
                .collect()
        }
    }
    pub fn push(&mut self, value: T) {
        let _ = icv_grow_seq(self, false);
        self.values.push(value);
    }
    pub fn push_front(&mut self, value: T) {
        let _ = icv_grow_seq(self, true);
        self.values.insert(0, value);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.values.pop()
    }
    pub fn pop_front(&mut self) -> Option<T> {
        (!self.values.is_empty()).then(|| self.values.remove(0))
    }
    pub fn insert(&mut self, index: usize, value: T) {
        self.values.insert(index.min(self.values.len()), value);
    }
    pub fn remove(&mut self, index: usize) -> Option<T> {
        (index < self.values.len()).then(|| self.values.remove(index))
    }
    pub fn clear(&mut self) {
        self.values.clear();
    }
    pub fn invert(&mut self) {
        self.values.reverse();
    }
    pub fn reader(&self, reverse: bool) -> CvSeqReader<'_, T> {
        CvSeqReader {
            sequence: self,
            position: if reverse {
                self.values.len() as isize - 1
            } else {
                0
            },
            reverse,
        }
    }
}

/// `cvSetSeqBlockSize` (`cxdatastructs.cpp:488`).  `CvSeq` owns typed
/// elements instead of byte blocks, so this records the same future-growth
/// count directly.  A zero request has the source default of 1024 elements;
/// negative requests are rejected just as C reports `CV_StsOutOfRange`.
pub fn cv_set_seq_block_size<T>(sequence: &mut CvSeq<T>, delta_elements: i32) -> Result<(), ()> {
    if delta_elements < 0 {
        return Err(());
    }
    sequence.set_block_size(if delta_elements == 0 {
        1024
    } else {
        delta_elements as usize
    });
    Ok(())
}

/// `cvSeqElemIdx` (`cxdatastructs.cpp:566`).  C determines the index by
/// locating a byte pointer in a sequence block.  A Rust reference can only be
/// valid for one of the owned elements when pointer identity matches; `-1`
/// retains the source result for an unrelated element.
pub fn cv_seq_elem_idx<T>(sequence: &CvSeq<T>, element: &T) -> i32 {
    sequence
        .values
        .iter()
        .position(|item| core::ptr::eq(item, element))
        .and_then(|index| i32::try_from(index).ok())
        .unwrap_or(-1)
}

/// `cvCvtSeqToArray` (`cxdatastructs.cpp:639`).  The C function copies its
/// possibly wrapped block range into caller-owned contiguous bytes.  Returning
/// an owned vector provides that same copy for typed Rust sequences.
pub fn cv_cvt_seq_to_array<T: Clone>(sequence: &CvSeq<T>, slice: CvSlice) -> Vec<T> {
    sequence.slice(slice).into_iter().cloned().collect()
}

/// `cvSeqPush`/`cvSeqPop` (`cxdatastructs.cpp:1312`).  Source returns a byte
/// pointer only to let callers fill the stored slot; typed Rust accepts and
/// returns the value while preserving end-of-sequence behavior.
pub fn cv_seq_push<T>(sequence: &mut CvSeq<T>, value: T) -> Result<(), ()> {
    if !icv_grow_seq(sequence, false) {
        return Err(());
    }
    sequence.values.push(value);
    Ok(())
}
pub fn cv_seq_pop<T>(sequence: &mut CvSeq<T>) -> Result<T, ()> {
    sequence.values.pop().ok_or(())
}

/// `cvSeqPushFront`/`cvSeqPopFront` (`cxdatastructs.cpp:1380`).
pub fn cv_seq_push_front<T>(sequence: &mut CvSeq<T>, value: T) -> Result<(), ()> {
    if !icv_grow_seq(sequence, true) {
        return Err(());
    }
    sequence.values.insert(0, value);
    Ok(())
}
pub fn cv_seq_pop_front<T>(sequence: &mut CvSeq<T>) -> Result<T, ()> {
    (!sequence.values.is_empty())
        .then(|| sequence.values.remove(0))
        .ok_or(())
}

/// `cvSeqInsert`/`cvSeqRemove` (`cxdatastructs.cpp:1430`).  The historical
/// API permits one wrapped negative/too-large index, then rejects anything
/// still outside the sequence.
pub fn cv_seq_insert<T>(
    sequence: &mut CvSeq<T>,
    mut before_index: isize,
    value: T,
) -> Result<(), ()> {
    let total = sequence.values.len() as isize;
    if before_index < 0 {
        before_index += total;
    }
    if before_index > total {
        before_index -= total;
    }
    if !(0..=total).contains(&before_index) {
        return Err(());
    }
    if !icv_grow_seq(sequence, before_index == 0) {
        return Err(());
    }
    sequence.values.insert(before_index as usize, value);
    Ok(())
}
pub fn cv_seq_remove<T>(sequence: &mut CvSeq<T>, mut index: isize) -> Result<T, ()> {
    let total = sequence.values.len() as isize;
    if index < 0 {
        index += total;
    }
    if index >= total {
        index -= total;
    }
    if !(0..total).contains(&index) {
        return Err(());
    }
    Ok(sequence.values.remove(index as usize))
}

/// `cvSeqPushMulti`/`cvSeqPopMulti` (`cxdatastructs.cpp:1630`).
pub fn cv_seq_push_multi<T: Clone>(
    sequence: &mut CvSeq<T>,
    values: &[T],
    front: bool,
) -> Result<(), ()> {
    if !sequence.values.try_reserve(values.len()).is_ok() {
        return Err(());
    }
    sequence.push_multi(values, front);
    Ok(())
}
pub fn cv_seq_pop_multi<T: Clone>(
    sequence: &mut CvSeq<T>,
    count: i32,
    front: bool,
) -> Result<Vec<T>, ()> {
    if count < 0 {
        return Err(());
    }
    Ok(sequence.pop_multi(count as usize, front))
}

/// `cvClearSeq` (`cxdatastructs.cpp:1765`).
pub fn cv_clear_seq<T>(sequence: &mut CvSeq<T>) {
    sequence.clear();
}

/// `cvSeqSlice` (`cxdatastructs.cpp:1833`).  A C non-copy slice aliases arena
/// blocks, an alias which cannot safely outlive a mutable Rust sequence; this
/// returns the equivalent typed, independently owned subsequence.
pub fn cv_seq_slice<T: Clone>(sequence: &CvSeq<T>, slice: CvSlice) -> CvSeq<T> {
    sequence.clone_slice(slice)
}

/// `cvSeqRemoveSlice` (`cxdatastructs.cpp:1916`).
pub fn cv_seq_remove_slice<T: Clone>(sequence: &mut CvSeq<T>, slice: CvSlice) -> Result<(), ()> {
    if sequence.values.is_empty() {
        return Err(());
    }
    sequence.remove_slice(slice);
    Ok(())
}

/// `cvSeqInsertSlice` (`cxdatastructs.cpp:1993`).  The C function accepts a
/// sequence or a one-dimensional matrix; a typed slice covers both safe Rust
/// representations without pointer/header reinterpretation.
pub fn cv_seq_insert_slice<T: Clone>(
    sequence: &mut CvSeq<T>,
    mut index: isize,
    values: &[T],
) -> Result<(), ()> {
    let total = sequence.values.len() as isize;
    if index < 0 {
        index += total;
    }
    if index > total {
        index -= total;
    }
    if !(0..=total).contains(&index) {
        return Err(());
    }
    sequence.values.try_reserve(values.len()).map_err(|_| ())?;
    sequence.insert_slice(index as usize, values);
    Ok(())
}

/// `cvSeqSort` (`cxdatastructs.cpp:2154`), expressed with Rust's stable
/// comparator-driven sort instead of swapping untyped sequence bytes.
pub fn cv_seq_sort<T, F>(sequence: &mut CvSeq<T>, mut compare: F)
where
    F: FnMut(&T, &T) -> core::cmp::Ordering,
{
    sequence.values.sort_by(|left, right| compare(left, right));
}

/// `cvSeqSearch` (`cxdatastructs.cpp:2440`).  For sorted input, `Err` is the
/// insertion position emitted by the native `_idx` output when no match exists.
pub fn cv_seq_search<T, F>(
    sequence: &CvSeq<T>,
    value: &T,
    sorted: bool,
    mut compare: F,
) -> Result<usize, usize>
where
    F: FnMut(&T, &T) -> core::cmp::Ordering,
{
    if sorted {
        sequence
            .values
            .binary_search_by(|item| compare(item, value))
    } else {
        sequence
            .values
            .iter()
            .position(|item| compare(value, item).is_eq())
            .ok_or(sequence.values.len())
    }
}

/// `icvMed3` (`cxdatastructs.cpp:2146`), used by the source quick-sort pivot
/// selection.  It returns the median input without moving typed elements.
pub fn icv_med3<'a, T, F>(a: &'a T, b: &'a T, c: &'a T, mut compare: F) -> &'a T
where
    F: FnMut(&T, &T) -> core::cmp::Ordering,
{
    if compare(a, b).is_lt() {
        if compare(b, c).is_lt() {
            b
        } else if compare(a, c).is_lt() {
            c
        } else {
            a
        }
    } else if compare(a, c).is_lt() {
        a
    } else if compare(b, c).is_lt() {
        c
    } else {
        b
    }
}

/// Typed replacement for `icvSeqElemsClearFlags` (`cxdatastructs.cpp:3342`).
/// The source computes an integer field by byte offset; a closure selects that
/// field directly, preserving the bit-clearing operation without casts.
pub fn icv_seq_elems_clear_flags<T, F>(sequence: &mut CvSeq<T>, clear_mask: i32, mut flags: F)
where
    F: FnMut(&mut T) -> &mut i32,
{
    for element in &mut sequence.values {
        *flags(element) &= !clear_mask;
    }
}

/// Typed replacement for `icvSeqFindNextElem` (`cxdatastructs.cpp:3375`).
/// It scans once from the caller's wrapped start position and returns the
/// matching stable index while updating that start position as the C routine
/// does through its output pointer.
pub fn icv_seq_find_next_elem<T, F>(
    sequence: &CvSeq<T>,
    mask: i32,
    value: i32,
    start_index: &mut isize,
    mut flags: F,
) -> Option<usize>
where
    F: FnMut(&T) -> i32,
{
    let total = sequence.values.len();
    if total == 0 {
        return None;
    }
    let start = start_index.rem_euclid(total as isize) as usize;
    for step in 0..total {
        let index = (start + step) % total;
        if flags(&sequence.values[index]) & mask == value {
            *start_index = index as isize;
            return Some(index);
        }
    }
    None
}

/// `cvSeqInvert` (`cxdatastructs.cpp:2548`).
pub fn cv_seq_invert<T>(sequence: &mut CvSeq<T>) {
    sequence.values.reverse();
}

/// `cvSeqPartition` (`cxdatastructs.cpp:2591`).  It returns the number of
/// equivalence classes and one zero-based class label per sequence element.
pub fn cv_seq_partition<T, F>(sequence: &CvSeq<T>, mut is_equal: F) -> (usize, Vec<i32>)
where
    F: FnMut(&T, &T) -> bool,
{
    let total = sequence.values.len();
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank = vec![0_u8; total];
    fn root(parent: &mut [usize], node: usize) -> usize {
        if parent[node] != node {
            let ancestor = parent[node];
            parent[node] = root(parent, ancestor);
        }
        parent[node]
    }
    for left in 0..total {
        for right in left + 1..total {
            if is_equal(&sequence.values[left], &sequence.values[right]) {
                let left_root = root(&mut parent, left);
                let right_root = root(&mut parent, right);
                if left_root != right_root {
                    if rank[left_root] < rank[right_root] {
                        parent[left_root] = right_root;
                    } else {
                        parent[right_root] = left_root;
                        if rank[left_root] == rank[right_root] {
                            rank[left_root] += 1;
                        }
                    }
                }
            }
        }
    }
    let mut labels = vec![0_i32; total];
    let mut root_labels = std::collections::HashMap::new();
    for (index, label) in labels.iter_mut().enumerate() {
        let class = root(&mut parent, index);
        let next = root_labels.len() as i32;
        *label = *root_labels.entry(class).or_insert(next);
    }
    (root_labels.len(), labels)
}

impl<T: Clone> CvSeq<T> {
    pub fn push_multi(&mut self, values: &[T], front: bool) {
        if front {
            self.values.splice(0..0, values.iter().cloned());
        } else {
            self.values.extend_from_slice(values);
        }
    }
    pub fn pop_multi(&mut self, count: usize, front: bool) -> Vec<T> {
        let count = count.min(self.values.len());
        if front {
            self.values.drain(..count).collect()
        } else {
            self.values.drain(self.values.len() - count..).collect()
        }
    }
    pub fn remove_slice(&mut self, slice: CvSlice) {
        let length = self.values.len() as isize;
        let normalize = |index: isize| if index < 0 { (length + index).max(0) } else { index.min(length) } as usize;
        let start = normalize(slice.start);
        let end = normalize(slice.end);
        if start <= end {
            self.values.drain(start..end);
        } else {
            self.values.drain(start..);
            self.values.drain(..end);
        }
    }
    pub fn insert_slice(&mut self, index: usize, values: &[T]) {
        self.values.splice(
            index.min(self.values.len())..index.min(self.values.len()),
            values.iter().cloned(),
        );
    }
    pub fn clone_slice(&self, slice: CvSlice) -> Self {
        Self {
            values: self.slice(slice).into_iter().cloned().collect(),
            block_size: self.block_size,
        }
    }
}

impl<T: Ord> CvSeq<T> {
    pub fn sort(&mut self) {
        self.values.sort();
    }
    pub fn search(&self, value: &T) -> Result<usize, usize> {
        self.values.binary_search(value)
    }
}

/// C `CvSeqReader`.
pub struct CvSeqReader<'a, T> {
    sequence: &'a CvSeq<T>,
    position: isize,
    reverse: bool,
}
impl<'a, T> CvSeqReader<'a, T> {
    pub fn position(&self) -> isize {
        self.position
    }
    pub fn set_position(&mut self, position: isize, relative: bool) {
        let _ = cv_set_seq_reader_pos(self, position, relative);
    }
}
impl<'a, T> Iterator for CvSeqReader<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.sequence.get(self.position);
        self.position += if self.reverse { -1 } else { 1 };
        value
    }
}

/// `cvStartReadSeq` (`cxdatastructs.cpp:1071`).  Typed sequences retain their
/// elements contiguously, while `CvSeqReader` preserves the source reader's
/// current-position and reverse traversal contract.
pub fn cv_start_read_seq<T>(sequence: &CvSeq<T>, reverse: bool) -> CvSeqReader<'_, T> {
    sequence.reader(reverse)
}

/// `cvChangeSeqBlock` (`cxdatastructs.cpp:1130`).  Rust stores a sequence in
/// one contiguous allocation; `block_size` therefore defines logical blocks
/// for callers that used the C block-boundary reader operation.
pub fn cv_change_seq_block<T>(reader: &mut CvSeqReader<'_, T>, direction: i32) -> Result<(), ()> {
    let total = reader.sequence.values.len();
    if total == 0 || direction == 0 {
        return Err(());
    }
    let block_size = reader.sequence.block_size.max(1);
    let current = reader.position.rem_euclid(total as isize) as usize;
    let block_count = (total - 1) / block_size + 1;
    reader.position = if direction > 0 {
        (((current / block_size + 1) % block_count) * block_size) as isize
    } else {
        let block = current / block_size;
        let previous = (block + block_count - 1) % block_count;
        ((previous + 1) * block_size).min(total).saturating_sub(1) as isize
    };
    Ok(())
}

/// `cvGetSeqReaderPos` (`cxdatastructs.cpp:1160`).
pub fn cv_get_seq_reader_pos<T>(reader: &CvSeqReader<'_, T>) -> Result<i32, ()> {
    let total = reader.sequence.values.len();
    if total == 0 {
        return Err(());
    }
    i32::try_from(reader.position.rem_euclid(total as isize)).map_err(|_| ())
}

/// `cvSetSeqReaderPos` (`cxdatastructs.cpp:1190`).  The native function wraps
/// one sequence length for absolute positions and follows blocks for relative
/// positions.  With contiguous typed storage, modular indexing provides the
/// same observable element selection for both directions.
pub fn cv_set_seq_reader_pos<T>(
    reader: &mut CvSeqReader<'_, T>,
    index: isize,
    relative: bool,
) -> Result<(), ()> {
    let total = reader.sequence.values.len();
    if total == 0 {
        return Err(());
    }
    let total = total as isize;
    if !relative && (index < -total || index >= total * 2) {
        return Err(());
    }
    let base = if relative { reader.position } else { 0 };
    reader.position = (base + index).rem_euclid(total);
    Ok(())
}

/// C `CvSet`: removed values leave reusable stable slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CvSet<T> {
    slots: Vec<Option<T>>,
    free: Vec<usize>,
}
impl<T> Default for CvSet<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}
impl<T> CvSet<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, value: T) -> usize {
        if let Some(index) = self.free.pop() {
            self.slots[index] = Some(value);
            index
        } else {
            self.slots.push(Some(value));
            self.slots.len() - 1
        }
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.as_ref()
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.slots.get_mut(index)?.as_mut()
    }
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let value = self.slots.get_mut(index)?.take()?;
        self.free.push(index);
        Some(value)
    }
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
    }
    pub fn active_count(&self) -> usize {
        self.slots.len() - self.free.len()
    }
}

/// Safe typed forms of `cvCreateSet`, `cvSetAdd`, `cvSetRemove`, and
/// `cvClearSet` (`cxdatastructs.cpp:2745`).  Slot indices replace native
/// element pointers while retaining reuse of removed positions.
pub fn cv_create_set<T>() -> CvSet<T> {
    CvSet::new()
}
pub fn cv_set_add<T>(set: &mut CvSet<T>, value: T) -> usize {
    set.add(value)
}
pub fn cv_set_remove<T>(set: &mut CvSet<T>, index: usize) -> Result<T, ()> {
    set.remove(index).ok_or(())
}
pub fn cv_clear_set<T>(set: &mut CvSet<T>) {
    set.clear();
}

/// Owned undirected C `CvGraph`, with stable `CvSet` vertex/edge indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CvGraph<V, E> {
    pub vertices: CvSet<V>,
    pub edges: CvSet<(usize, usize, E)>,
}
impl<V, E> Default for CvGraph<V, E> {
    fn default() -> Self {
        Self {
            vertices: CvSet::new(),
            edges: CvSet::new(),
        }
    }
}
impl<V, E> CvGraph<V, E> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_vertex(&mut self, vertex: V) -> usize {
        self.vertices.add(vertex)
    }
    pub fn add_edge(&mut self, start: usize, end: usize, edge: E) -> Result<bool, E> {
        if self.vertices.get(start).is_none() || self.vertices.get(end).is_none() {
            return Err(edge);
        }
        if self.find_edge(start, end).is_some() {
            return Ok(false);
        }
        self.edges.add((start, end, edge));
        Ok(true)
    }
    pub fn find_edge(&self, start: usize, end: usize) -> Option<usize> {
        self.edges.slots.iter().position(|edge| {
            edge.as_ref()
                .is_some_and(|(a, b, _)| (*a == start && *b == end) || (*a == end && *b == start))
        })
    }
    pub fn remove_edge(&mut self, start: usize, end: usize) {
        if let Some(index) = self.find_edge(start, end) {
            self.edges.remove(index);
        }
    }
    pub fn remove_vertex(&mut self, vertex: usize) -> Option<V> {
        let incident: Vec<_> = self
            .edges
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, edge)| {
                edge.as_ref()
                    .filter(|(a, b, _)| *a == vertex || *b == vertex)
                    .map(|_| index)
            })
            .collect();
        for edge in incident {
            self.edges.remove(edge);
        }
        self.vertices.remove(vertex)
    }
    pub fn degree(&self, vertex: usize) -> usize {
        self.edges
            .slots
            .iter()
            .filter(|edge| {
                edge.as_ref()
                    .is_some_and(|(a, b, _)| *a == vertex || *b == vertex)
            })
            .count()
    }
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
    }
    pub fn breadth_first(&self, start: usize) -> Vec<usize> {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([start]);
        let mut result = Vec::new();
        while let Some(vertex) = queue.pop_front() {
            if !seen.insert(vertex) || self.vertices.get(vertex).is_none() {
                continue;
            }
            result.push(vertex);
            for edge in self.edges.slots.iter().flatten() {
                if edge.0 == vertex {
                    queue.push_back(edge.1);
                } else if edge.1 == vertex {
                    queue.push_back(edge.0);
                }
            }
        }
        result
    }
}

/// Typed graph entry points corresponding to `cvCreateGraph` through
/// `cvGraphRemoveEdgeByPtr` (`cxdatastructs.cpp:2868`).  Rust stable indices
/// are the safe substitute for the C vertex/edge pointers.
pub fn cv_create_graph<V, E>() -> CvGraph<V, E> {
    CvGraph::new()
}
pub fn cv_clear_graph<V, E>(graph: &mut CvGraph<V, E>) {
    graph.clear();
}
pub fn cv_graph_add_vtx<V, E>(graph: &mut CvGraph<V, E>, vertex: V) -> usize {
    graph.add_vertex(vertex)
}
pub fn cv_graph_remove_vtx_by_ptr<V, E>(graph: &mut CvGraph<V, E>, vertex: usize) -> Result<V, ()> {
    graph.remove_vertex(vertex).ok_or(())
}
pub fn cv_graph_remove_vtx<V, E>(graph: &mut CvGraph<V, E>, index: usize) -> Result<V, ()> {
    cv_graph_remove_vtx_by_ptr(graph, index)
}
pub fn cv_find_graph_edge_by_ptr<V, E>(
    graph: &CvGraph<V, E>,
    start: usize,
    end: usize,
) -> Option<usize> {
    graph.find_edge(start, end)
}
pub fn cv_find_graph_edge<V, E>(graph: &CvGraph<V, E>, start: usize, end: usize) -> Option<usize> {
    cv_find_graph_edge_by_ptr(graph, start, end)
}
pub fn cv_graph_add_edge_by_ptr<V, E>(
    graph: &mut CvGraph<V, E>,
    start: usize,
    end: usize,
    edge: E,
) -> Result<bool, E> {
    graph.add_edge(start, end, edge)
}
pub fn cv_graph_add_edge<V, E>(
    graph: &mut CvGraph<V, E>,
    start: usize,
    end: usize,
    edge: E,
) -> Result<bool, E> {
    cv_graph_add_edge_by_ptr(graph, start, end, edge)
}
pub fn cv_graph_remove_edge_by_ptr<V, E>(graph: &mut CvGraph<V, E>, start: usize, end: usize) {
    graph.remove_edge(start, end);
}
pub fn cv_graph_remove_edge<V, E>(graph: &mut CvGraph<V, E>, start: usize, end: usize) {
    cv_graph_remove_edge_by_ptr(graph, start, end);
}
pub fn cv_graph_vtx_degree_by_ptr<V, E>(graph: &CvGraph<V, E>, vertex: usize) -> Result<usize, ()> {
    graph
        .vertices
        .get(vertex)
        .map(|_| graph.degree(vertex))
        .ok_or(())
}
pub fn cv_graph_vtx_degree<V, E>(graph: &CvGraph<V, E>, vertex: usize) -> Result<usize, ()> {
    cv_graph_vtx_degree_by_ptr(graph, vertex)
}

/// `cvCloneGraph` (`cxdatastructs.cpp:3665`).  Both vertex and edge slot
/// stores are owned, so cloning preserves stable indices and their adjacency
/// relation without the native pointer-remapping pass.
pub fn cv_clone_graph<V: Clone, E: Clone>(graph: &CvGraph<V, E>) -> CvGraph<V, E> {
    graph.clone()
}

/// `CvGraphScanner` event masks from `cxcore.h`.  The negative `OVER` value
/// is returned once the owned event stream has been exhausted.
pub const CV_GRAPH_VERTEX: i32 = 1;
pub const CV_GRAPH_TREE_EDGE: i32 = 2;
pub const CV_GRAPH_BACK_EDGE: i32 = 4;
pub const CV_GRAPH_FORWARD_EDGE: i32 = 8;
pub const CV_GRAPH_CROSS_EDGE: i32 = 16;
pub const CV_GRAPH_NEW_TREE: i32 = 32;
pub const CV_GRAPH_BACKTRACKING: i32 = 64;
pub const CV_GRAPH_OVER: i32 = -1;

/// One safe, stable-index replacement for the mutable fields in C's
/// `CvGraphScanner`.  The native scanner publishes its current vertex, edge,
/// and destination through raw pointers; this item makes each publication an
/// immutable value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvGraphItem {
    pub code: i32,
    pub vertex: Option<usize>,
    pub edge: Option<usize>,
    pub destination: Option<usize>,
}

/// Owned depth-first scanner for [`CvGraph`].  Its event queue lets callers
/// release the scanner or mutate unrelated Rust values without invalidating
/// borrowed graph pointers, unlike the C arena-backed cursor.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CvGraphScanner {
    events: Vec<CvGraphItem>,
    position: usize,
    released: bool,
}

/// `cvCreateGraphScanner` (`cxdatastructs.cpp:3434`).  `start` corresponds to
/// the optional C start vertex.  This graph type is undirected, so the C-only
/// forward-edge state is never emitted; visited non-tree edges are classified
/// as back edges when they target the active DFS path and cross edges otherwise.
pub fn cv_create_graph_scanner<V, E>(
    graph: &CvGraph<V, E>,
    start: Option<usize>,
    mask: i32,
) -> Result<CvGraphScanner, ()> {
    if start.is_some_and(|vertex| graph.vertices.get(vertex).is_none()) {
        return Err(());
    }
    fn visit<V, E>(
        graph: &CvGraph<V, E>,
        vertex: usize,
        mask: i32,
        visited_vertices: &mut HashSet<usize>,
        active: &mut HashSet<usize>,
        visited_edges: &mut HashSet<usize>,
        events: &mut Vec<CvGraphItem>,
    ) {
        visited_vertices.insert(vertex);
        active.insert(vertex);
        if mask & CV_GRAPH_VERTEX != 0 {
            events.push(CvGraphItem {
                code: CV_GRAPH_VERTEX,
                vertex: Some(vertex),
                edge: None,
                destination: None,
            });
        }
        for (edge_index, edge) in graph.edges.slots.iter().enumerate() {
            let Some((left, right, _)) = edge else {
                continue;
            };
            if *left != vertex && *right != vertex {
                continue;
            }
            if !visited_edges.insert(edge_index) {
                continue;
            }
            let destination = if *left == vertex { *right } else { *left };
            if !visited_vertices.contains(&destination) {
                if mask & CV_GRAPH_TREE_EDGE != 0 {
                    events.push(CvGraphItem {
                        code: CV_GRAPH_TREE_EDGE,
                        vertex: Some(vertex),
                        edge: Some(edge_index),
                        destination: Some(destination),
                    });
                }
                visit(
                    graph,
                    destination,
                    mask,
                    visited_vertices,
                    active,
                    visited_edges,
                    events,
                );
                if mask & CV_GRAPH_BACKTRACKING != 0 {
                    events.push(CvGraphItem {
                        code: CV_GRAPH_BACKTRACKING,
                        vertex: Some(vertex),
                        edge: Some(edge_index),
                        destination: Some(destination),
                    });
                }
            } else {
                let code = if active.contains(&destination) {
                    CV_GRAPH_BACK_EDGE
                } else {
                    CV_GRAPH_CROSS_EDGE
                };
                if mask & code != 0 {
                    events.push(CvGraphItem {
                        code,
                        vertex: Some(vertex),
                        edge: Some(edge_index),
                        destination: Some(destination),
                    });
                }
            }
        }
        active.remove(&vertex);
    }

    let mut scanner = CvGraphScanner::default();
    let mut visited_vertices = HashSet::new();
    let mut active = HashSet::new();
    let mut visited_edges = HashSet::new();
    let mut roots: Vec<usize> = start.into_iter().collect();
    roots.extend(
        graph
            .vertices
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.as_ref().map(|_| index)),
    );
    for root in roots {
        if visited_vertices.contains(&root) {
            continue;
        }
        if mask & CV_GRAPH_NEW_TREE != 0 {
            scanner.events.push(CvGraphItem {
                code: CV_GRAPH_NEW_TREE,
                vertex: None,
                edge: None,
                destination: Some(root),
            });
        }
        visit(
            graph,
            root,
            mask,
            &mut visited_vertices,
            &mut active,
            &mut visited_edges,
            &mut scanner.events,
        );
    }
    Ok(scanner)
}

/// `cvReleaseGraphScanner` (`cxdatastructs.cpp:3483`).  Dropping a Rust value
/// is normally sufficient; explicit release makes the C lifecycle observable.
pub fn cv_release_graph_scanner(scanner: &mut CvGraphScanner) {
    scanner.events.clear();
    scanner.position = 0;
    scanner.released = true;
}

/// `cvNextGraphItem` (`cxdatastructs.cpp:3504`).  Once the scanner is released
/// or its events are consumed, the source's `CV_GRAPH_OVER` result is returned.
pub fn cv_next_graph_item(scanner: &mut CvGraphScanner) -> CvGraphItem {
    if scanner.released || scanner.position == scanner.events.len() {
        return CvGraphItem {
            code: CV_GRAPH_OVER,
            vertex: None,
            edge: None,
            destination: None,
        };
    }
    let item = scanner.events[scanner.position];
    scanner.position += 1;
    item
}

/// Owned C `cvTreeToNodeSeq`/tree iterator equivalent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CvTree<T> {
    pub value: T,
    pub children: Vec<CvTree<T>>,
}
impl<T> CvTree<T> {
    pub fn preorder(&self) -> CvSeq<&T> {
        let mut result = CvSeq::new();
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            result.push(&node.value);
            stack.extend(node.children.iter().rev());
        }
        result
    }
    pub fn insert_child(&mut self, child: CvTree<T>) {
        self.children.push(child);
    }
    pub fn remove_child(&mut self, index: usize) -> Option<CvTree<T>> {
        (index < self.children.len()).then(|| self.children.remove(index))
    }
}

/// `cvTreeToNodeSeq` (`cxdatastructs.cpp:3760`), returning the traversal's
/// borrowed node values rather than native node addresses.
pub fn cv_tree_to_node_seq<T>(tree: &CvTree<T>) -> CvSeq<&T> {
    tree.preorder()
}

/// `cvInsertNodeIntoTree` / `cvRemoveNodeFromTree` (`cxdatastructs.cpp:3811`).
/// Owned children make parent/frame pointer rewiring unnecessary.
pub fn cv_insert_node_into_tree<T>(parent: &mut CvTree<T>, node: CvTree<T>) {
    parent.insert_child(node);
}
pub fn cv_remove_node_from_tree<T>(parent: &mut CvTree<T>, index: usize) -> Option<CvTree<T>> {
    parent.remove_child(index)
}

/// Borrowed equivalent of `CvTreeNodeIterator`.  The native representation
/// walks horizontal/vertical pointer links; the owned tree precomputes that
/// same depth-first order, bounded by `max_level`.
pub struct CvTreeNodeIterator<'a, T> {
    nodes: Vec<&'a T>,
    current: isize,
}
pub fn cv_init_tree_node_iterator<T>(
    tree: &CvTree<T>,
    max_level: usize,
) -> CvTreeNodeIterator<'_, T> {
    fn visit<'a, T>(node: &'a CvTree<T>, level: usize, max_level: usize, nodes: &mut Vec<&'a T>) {
        nodes.push(&node.value);
        if level < max_level {
            for child in &node.children {
                visit(child, level + 1, max_level, nodes);
            }
        }
    }
    let mut nodes = Vec::new();
    visit(tree, 0, max_level, &mut nodes);
    CvTreeNodeIterator { nodes, current: 0 }
}
pub fn cv_next_tree_node<'a, T>(iterator: &mut CvTreeNodeIterator<'a, T>) -> Option<&'a T> {
    let node = iterator
        .nodes
        .get(iterator.current.max(0) as usize)
        .copied();
    if node.is_some() {
        iterator.current += 1;
    }
    node
}
pub fn cv_prev_tree_node<'a, T>(iterator: &mut CvTreeNodeIterator<'a, T>) -> Option<&'a T> {
    let node = iterator
        .nodes
        .get(iterator.current.max(0) as usize)
        .copied();
    if node.is_some() {
        iterator.current -= 1;
    }
    node
}

#[cfg(test)]
mod tests {
    use super::{
        CV_GRAPH_BACKTRACKING, CV_GRAPH_NEW_TREE, CV_GRAPH_OVER, CV_GRAPH_TREE_EDGE,
        CV_GRAPH_VERTEX, CV_STORAGE_BLOCK_SIZE, CvGraph, CvMemStorage, CvSeq, CvSet, CvSlice,
        CvTree, cv_align_left, cv_change_seq_block, cv_clear_seq, cv_clear_set, cv_create_graph,
        cv_create_graph_scanner, cv_create_seq_block, cv_create_set, cv_cvt_seq_to_array,
        cv_end_write_seq, cv_find_graph_edge, cv_flush_seq_writer, cv_get_seq_reader_pos,
        cv_graph_add_edge, cv_graph_add_vtx, cv_graph_remove_vtx, cv_make_seq_header_for_array,
        cv_next_graph_item, cv_release_graph_scanner, cv_seq_elem_idx, cv_seq_insert,
        cv_seq_insert_slice, cv_seq_invert, cv_seq_partition, cv_seq_pop, cv_seq_pop_front,
        cv_seq_pop_multi, cv_seq_push, cv_seq_push_front, cv_seq_push_multi, cv_seq_remove,
        cv_seq_remove_slice, cv_seq_search, cv_seq_slice, cv_seq_sort, cv_set_add, cv_set_remove,
        cv_set_seq_block_size, cv_set_seq_reader_pos, cv_start_append_to_seq, cv_start_read_seq,
        cv_start_write_seq, icv_destroy_mem_storage, icv_free_seq_block, icv_go_next_mem_block,
        icv_grow_seq, icv_init_mem_storage, icv_med3, icv_seq_elems_clear_flags,
        icv_seq_find_next_elem,
    };
    #[test]
    fn align_left_rounds_down_with_the_native_mask() {
        assert_eq!(cv_align_left(31, 8), 24);
        assert_eq!(cv_align_left(32, 8), 32);
        assert_eq!(cv_align_left(7, 8), 0);
    }
    #[test]
    fn storage_lifecycle_uses_source_default_and_alignment() {
        let mut storage = CvMemStorage::new(0);
        assert_eq!(storage.block_size, CV_STORAGE_BLOCK_SIZE);
        storage.allocate(4);
        icv_destroy_mem_storage(&mut storage);
        assert_eq!(storage.save_position().blocks, 0);
        icv_init_mem_storage(&mut storage, 9);
        assert_eq!(storage.block_size, 16);
    }
    #[test]
    fn next_memory_block_has_the_source_configured_size() {
        let mut storage = CvMemStorage::new(16);
        assert_eq!(icv_go_next_mem_block(&mut storage).len(), 16);
        assert_eq!(icv_go_next_mem_block(&mut storage).len(), 16);
        assert_eq!(storage.save_position().blocks, 2);
    }
    #[test]
    fn sequence_block_size_uses_default_and_rejects_negative_values() {
        let mut sequence = CvSeq::<i32>::new();
        assert_eq!(cv_set_seq_block_size(&mut sequence, 0), Ok(()));
        assert_eq!(sequence.block_size, 1024);
        assert_eq!(cv_set_seq_block_size(&mut sequence, 7), Ok(()));
        assert_eq!(sequence.block_size, 7);
        assert_eq!(cv_set_seq_block_size(&mut sequence, -1), Err(()));
    }
    #[test]
    fn sequence_element_index_uses_owned_element_identity() {
        let mut sequence = CvSeq::new();
        sequence.push(4);
        sequence.push(4);
        let second = sequence.get(1).unwrap();
        assert_eq!(cv_seq_elem_idx(&sequence, second), 1);
        assert_eq!(cv_seq_elem_idx(&sequence, &4), -1);
    }
    #[test]
    fn sequence_array_copy_keeps_wrapped_slice_order() {
        let mut sequence = CvSeq::new();
        sequence.push_multi(&[1, 2, 3, 4], false);
        assert_eq!(
            cv_cvt_seq_to_array(&sequence, CvSlice { start: 2, end: 1 }),
            [3, 4, 1]
        );
    }
    #[test]
    fn array_sequence_header_borrows_without_copying_or_growth() {
        let values = [3, 5, 8];
        let header = cv_make_seq_header_for_array(&values);
        assert_eq!(header.len(), 3);
        assert_eq!(header.get(-1), Some(&8));
        assert!(core::ptr::eq(header.values.as_ptr(), values.as_ptr()));
    }
    #[test]
    fn sequence_growth_reserves_its_configured_block_before_insertion() {
        let mut sequence = CvSeq::<i32>::new();
        sequence.set_block_size(3);
        assert!(icv_grow_seq(&mut sequence, false));
        assert!(sequence.values.capacity() >= 3);
        sequence.push(1);
        sequence.push_front(0);
        icv_free_seq_block(&mut sequence, false);
        assert_eq!(sequence.values, [0, 1]);
    }
    #[test]
    fn sequence_writer_commits_only_at_flush_and_finishes_pending_values() {
        let mut sequence = CvSeq::new();
        let mut writer = cv_start_append_to_seq(&mut sequence);
        writer.push(1);
        writer.push(2);
        assert_eq!(writer.pending_len(), 2);
        cv_flush_seq_writer(&mut writer);
        assert_eq!(writer.pending_len(), 0);
        writer.push(3);
        let sequence = cv_end_write_seq(writer);
        assert_eq!(sequence.values, [1, 2, 3]);
    }
    #[test]
    fn source_sequence_block_pivot_and_flag_helpers_are_typed() {
        let mut sequence = CvSeq::new();
        sequence.set_block_size(4);
        let mut writer = cv_start_append_to_seq(&mut sequence);
        cv_create_seq_block(&mut writer).unwrap();
        writer.push(3_i32);
        writer.push(1);
        writer.push(2);
        let sequence = cv_end_write_seq(writer);
        assert_eq!(
            *icv_med3(
                &sequence.values[0],
                &sequence.values[1],
                &sequence.values[2],
                i32::cmp
            ),
            2
        );

        let mut flags = CvSeq::new();
        flags.push(0b111_i32);
        flags.push(0b010_i32);
        icv_seq_elems_clear_flags(&mut flags, 0b101, |value| value);
        assert_eq!(flags.values, [0b010, 0b010]);
        let mut start = -1;
        assert_eq!(
            icv_seq_find_next_elem(&flags, 0b010, 0b010, &mut start, |value| *value),
            Some(1)
        );
        assert_eq!(start, 1);
    }
    #[test]
    fn owned_sequence_writer_creates_and_finishes_a_new_sequence() {
        let mut writer = cv_start_write_seq::<i32>(2).unwrap();
        writer.push(4);
        writer.push(5);
        assert_eq!(writer.finish().values, [4, 5]);
        assert!(cv_start_write_seq::<i32>(-1).is_err());
    }
    #[test]
    fn sequence_reader_positions_wrap_and_change_logical_blocks() {
        let mut sequence = CvSeq::new();
        sequence.set_block_size(3);
        sequence.push_multi(&[0, 1, 2, 3, 4, 5, 6, 7], false);
        let mut reader = cv_start_read_seq(&sequence, false);
        assert_eq!(cv_get_seq_reader_pos(&reader), Ok(0));
        assert_eq!(cv_set_seq_reader_pos(&mut reader, -1, false), Ok(()));
        assert_eq!(cv_get_seq_reader_pos(&reader), Ok(7));
        assert_eq!(cv_change_seq_block(&mut reader, 1), Ok(()));
        assert_eq!(cv_get_seq_reader_pos(&reader), Ok(0));
        assert_eq!(cv_set_seq_reader_pos(&mut reader, 4, false), Ok(()));
        assert_eq!(cv_change_seq_block(&mut reader, -1), Ok(()));
        assert_eq!(cv_get_seq_reader_pos(&reader), Ok(2));
        assert_eq!(cv_set_seq_reader_pos(&mut reader, -3, true), Ok(()));
        assert_eq!(cv_get_seq_reader_pos(&reader), Ok(7));
    }
    #[test]
    fn source_sequence_mutators_preserve_front_back_and_wrapped_indices() {
        let mut sequence = CvSeq::new();
        cv_seq_push(&mut sequence, 2).unwrap();
        cv_seq_push_front(&mut sequence, 1).unwrap();
        cv_seq_insert(&mut sequence, 4, 3).unwrap();
        cv_seq_insert(&mut sequence, -1, 8).unwrap();
        assert_eq!(sequence.values, [1, 2, 8, 3]);
        assert_eq!(cv_seq_remove(&mut sequence, -2), Ok(8));
        assert_eq!(cv_seq_pop_front(&mut sequence), Ok(1));
        cv_seq_push_multi(&mut sequence, &[4, 5], false).unwrap();
        assert_eq!(cv_seq_pop_multi(&mut sequence, 2, false), Ok(vec![4, 5]));
        assert_eq!(cv_seq_pop(&mut sequence), Ok(3));
        cv_clear_seq(&mut sequence);
        assert!(sequence.values.is_empty());
    }
    #[test]
    fn source_slice_operations_copy_wrapped_order_and_mutate_destination() {
        let mut sequence = CvSeq::new();
        sequence.push_multi(&[0, 1, 2, 3, 4], false);
        assert_eq!(
            cv_seq_slice(&sequence, CvSlice { start: 3, end: 1 }).values,
            [3, 4, 0]
        );
        cv_seq_remove_slice(&mut sequence, CvSlice { start: 3, end: 1 }).unwrap();
        assert_eq!(sequence.values, [1, 2]);
        cv_seq_insert_slice(&mut sequence, -1, &[7, 8]).unwrap();
        assert_eq!(sequence.values, [1, 7, 8, 2]);
    }
    #[test]
    fn source_sequence_algorithms_use_comparators_and_connected_classes() {
        let mut sequence = CvSeq::new();
        sequence.push_multi(&[4, 1, 3, 2], false);
        cv_seq_sort(&mut sequence, i32::cmp);
        assert_eq!(sequence.values, [1, 2, 3, 4]);
        assert_eq!(cv_seq_search(&sequence, &3, true, i32::cmp), Ok(2));
        assert_eq!(cv_seq_search(&sequence, &5, true, i32::cmp), Err(4));
        cv_seq_invert(&mut sequence);
        assert_eq!(sequence.values, [4, 3, 2, 1]);
        let (count, labels) = cv_seq_partition(&sequence, |a, b| a % 2 == b % 2);
        assert_eq!(count, 2);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[1], labels[3]);
        assert_ne!(labels[0], labels[1]);
    }
    #[test]
    fn storage_sequences_and_sets_keep_source_observable_operations() {
        let mut storage = CvMemStorage::new(0);
        storage.allocate(3).copy_from_slice(&[1, 2, 3]);
        let pos = storage.save_position();
        storage.allocate(2);
        storage.restore_position(pos);
        let mut sequence = CvSeq::new();
        sequence.push_multi(&[2, 3], false);
        sequence.push_front(1);
        sequence.insert(3, 4);
        assert_eq!(sequence.slice(CvSlice { start: -3, end: -1 }), vec![&2, &3]);
        assert_eq!(sequence.pop_front(), Some(1));
        let mut set = CvSet::new();
        let first = set.add(1);
        set.remove(first);
        assert_eq!(set.add(2), first);
    }
    #[test]
    fn graph_and_tree_replace_pointer_traversal_with_owned_indices() {
        let mut graph = CvGraph::new();
        let a = graph.add_vertex("a");
        let b = graph.add_vertex("b");
        assert_eq!(graph.add_edge(a, b, ()), Ok(true));
        assert_eq!(graph.degree(a), 1);
        assert_eq!(graph.breadth_first(a), vec![a, b]);
        let mut tree = CvTree {
            value: 1,
            children: vec![],
        };
        tree.insert_child(CvTree {
            value: 2,
            children: vec![],
        });
        assert_eq!(tree.preorder().values, vec![&1, &2]);
    }
    #[test]
    fn source_set_and_graph_entry_points_keep_stable_slot_identity() {
        let mut set = cv_create_set();
        let index = cv_set_add(&mut set, 4);
        assert_eq!(cv_set_remove(&mut set, index), Ok(4));
        assert_eq!(cv_set_add(&mut set, 7), index);
        cv_clear_set(&mut set);
        let mut graph = cv_create_graph();
        let left = cv_graph_add_vtx(&mut graph, 'a');
        let right = cv_graph_add_vtx(&mut graph, 'b');
        assert_eq!(cv_graph_add_edge(&mut graph, left, right, 9), Ok(true));
        assert!(cv_find_graph_edge(&graph, left, right).is_some());
        assert_eq!(cv_graph_remove_vtx(&mut graph, left), Ok('a'));
        assert!(cv_find_graph_edge(&graph, left, right).is_none());
    }
    #[test]
    fn graph_scanner_emits_owned_depth_first_events_and_can_release() {
        let mut graph = CvGraph::new();
        let a = graph.add_vertex('a');
        let b = graph.add_vertex('b');
        let c = graph.add_vertex('c');
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        let mut scanner = cv_create_graph_scanner(
            &graph,
            Some(a),
            CV_GRAPH_NEW_TREE | CV_GRAPH_VERTEX | CV_GRAPH_TREE_EDGE | CV_GRAPH_BACKTRACKING,
        )
        .unwrap();
        let mut codes = Vec::new();
        loop {
            let item = cv_next_graph_item(&mut scanner);
            codes.push(item.code);
            if item.code == CV_GRAPH_OVER {
                break;
            }
        }
        assert_eq!(
            codes,
            [
                CV_GRAPH_NEW_TREE,
                CV_GRAPH_VERTEX,
                CV_GRAPH_TREE_EDGE,
                CV_GRAPH_VERTEX,
                CV_GRAPH_TREE_EDGE,
                CV_GRAPH_VERTEX,
                CV_GRAPH_BACKTRACKING,
                CV_GRAPH_BACKTRACKING,
                CV_GRAPH_OVER
            ]
        );
        cv_release_graph_scanner(&mut scanner);
        assert_eq!(cv_next_graph_item(&mut scanner).code, CV_GRAPH_OVER);
    }
}
