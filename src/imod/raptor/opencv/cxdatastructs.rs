//! Owned Rust translation of `IMOD/raptor/opencv/cxdatastructs.cpp`.
//!
//! The original 4,000-line unit is an allocator plus pointer-based sequence,
//! set, graph, DFS scanner, and tree implementation.  Rust ownership removes
//! its block/link/free-list machinery: the public data structures below retain
//! their observable operations while `Vec`, `Option`, and stable slot indices
//! replace raw addresses and manual memory storage.

use std::collections::{HashSet, VecDeque};

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
        Self {
            block_size: block_size.max(64 * 1024),
            blocks: Vec::new(),
        }
    }

    /// C `cvCreateChildMemStorage`; Rust child storage owns its allocations.
    pub fn child(&self) -> Self {
        Self::new(self.block_size)
    }

    /// C `cvClearMemStorage` and `cvReleaseMemStorage`.
    pub fn clear(&mut self) {
        self.blocks.clear();
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
        self.values.push(value);
    }
    pub fn push_front(&mut self, value: T) {
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
        self.position = if relative {
            self.position + position
        } else {
            position
        };
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

#[cfg(test)]
mod tests {
    use super::{CvGraph, CvMemStorage, CvSeq, CvSet, CvSlice, CvTree};
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
}
