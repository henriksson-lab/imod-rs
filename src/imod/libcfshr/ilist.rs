//! A typed, owned replacement for IMOD's historic `Ilist` container.
//!
//! The C container addressed a `void *` allocation in element-sized byte
//! offsets. Rust callers instead choose the element type once, so list
//! operations never need casts, element sizes, or byte copies.

#![allow(dead_code)]

pub const LIST_QUANTUM: usize = 1;
pub const ILIST_SOURCE_FUNCTIONS: &[&str] = &[
    "ilistNew",
    "ilistTruncate",
    "ilistQuantum",
    "ilistDup",
    "ilistDelete",
    "ilistFirst",
    "ilistNext",
    "ilistLast",
    "ilistItem",
    "ilistSize",
    "ilistAppend",
    "ilistRemove",
    "ilistSwap",
    "ilistPush",
    "ilistPop",
    "ilistFloat",
    "ilistInsert",
    "ilistShift",
];

/// A growable list with the cursor operations provided by the original API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ilist<T> {
    pub data: Vec<T>,
    current: Option<usize>,
    quantum: usize,
}

/// Matches the useful part of C `ilistNew`: make an empty typed list with an
/// initial capacity.
pub fn ilist_new<T>(capacity: usize) -> Ilist<T> {
    Ilist {
        data: Vec::with_capacity(capacity),
        current: None,
        quantum: LIST_QUANTUM,
    }
}

/// Matches C `ilistTruncate`.
pub fn ilist_truncate<T>(list: &mut Ilist<T>, size: usize) {
    if size < list.data.len() {
        list.data.truncate(size);
        if list.current.is_some_and(|current| current >= size) {
            list.current = None;
        }
    }
}

/// Sets the minimum number of elements reserved when the list next grows.
pub fn ilist_quantum<T>(list: &mut Ilist<T>, size: usize) {
    if size != 0 {
        list.quantum = size;
    }
}

/// Matches C `ilistDup` without its nullable-pointer and allocation-failure
/// conventions.
pub fn ilist_dup<T: Clone>(list: &Ilist<T>) -> Ilist<T> {
    list.clone()
}

/// C `ilistDelete`: dropping the owned Rust list releases its storage.
pub fn ilist_delete<T>(_list: Ilist<T>) {}

/// Returns the first item and positions the cursor there.
pub fn ilist_first<T>(list: &mut Ilist<T>) -> Option<&mut T> {
    if list.data.is_empty() {
        list.current = None;
        return None;
    }
    list.current = Some(0);
    list.data.first_mut()
}

/// Returns the item after the cursor.
pub fn ilist_next<T>(list: &mut Ilist<T>) -> Option<&mut T> {
    let next = list.current.map_or(0, |current| current.saturating_add(1));
    if next >= list.data.len() {
        list.current = None;
        return None;
    }
    list.current = Some(next);
    list.data.get_mut(next)
}

/// Returns the last item and positions the cursor there.
pub fn ilist_last<T>(list: &mut Ilist<T>) -> Option<&mut T> {
    let last = list.data.len().checked_sub(1)?;
    list.current = Some(last);
    list.data.get_mut(last)
}

/// Returns an item by index and positions the cursor there.
pub fn ilist_item<T>(list: &mut Ilist<T>, element: usize) -> Option<&mut T> {
    if element >= list.data.len() {
        return None;
    }
    list.current = Some(element);
    list.data.get_mut(element)
}

/// Matches C `ilistSize`.
pub fn ilist_size<T>(list: &Ilist<T>) -> usize {
    list.data.len()
}

/// Appends an item.
pub fn ilist_append<T>(list: &mut Ilist<T>, item: T) {
    if list.data.len() == list.data.capacity() {
        list.data.reserve(list.quantum);
    }
    list.data.push(item);
}

/// Removes an item, if its index is valid.
pub fn ilist_remove<T>(list: &mut Ilist<T>, element: usize) -> Option<T> {
    if element >= list.data.len() {
        return None;
    }
    let item = list.data.remove(element);
    match list.current {
        Some(current) if current == element => list.current = None,
        Some(current) if current > element => list.current = Some(current - 1),
        _ => {}
    }
    Some(item)
}

/// Swaps two items, returning whether both indices were valid.
pub fn ilist_swap<T>(list: &mut Ilist<T>, first: usize, second: usize) -> bool {
    if first >= list.data.len() || second >= list.data.len() {
        return false;
    }
    list.data.swap(first, second);
    true
}

/// Pushes an item onto the front of the list.
pub fn ilist_push<T>(list: &mut Ilist<T>, item: T) {
    if list.data.len() == list.data.capacity() {
        list.data.reserve(list.quantum);
    }
    list.data.insert(0, item);
    if let Some(current) = list.current {
        list.current = Some(current + 1);
    }
}

/// Removes and returns the first item.
pub fn ilist_pop<T>(list: &mut Ilist<T>) -> Option<T> {
    ilist_remove(list, 0)
}

/// Moves an item to the front, returning whether the index was valid.
pub fn ilist_float<T>(list: &mut Ilist<T>, element: usize) -> bool {
    let Some(item) = ilist_remove(list, element) else {
        return false;
    };
    ilist_push(list, item);
    true
}

/// Inserts an item at an index. On failure the item is returned unchanged.
pub fn ilist_insert<T>(list: &mut Ilist<T>, item: T, element: usize) -> Result<(), T> {
    if element > list.data.len() {
        return Err(item);
    }
    if list.data.len() == list.data.capacity() {
        list.data.reserve(list.quantum);
    }
    list.data.insert(element, item);
    if let Some(current) = list.current.filter(|current| *current >= element) {
        list.current = Some(current + 1);
    }
    Ok(())
}

/// C `ilistShift`.  The C operation moves the initialized suffix in-place;
/// this safe form has the same result and requires `Clone` only because Rust
/// cannot duplicate an arbitrary owned element by byte-copying it.
pub fn ilist_shift<T: Clone>(list: &mut Ilist<T>, start: usize, amount: isize) {
    if start >= list.data.len() || amount == 0 {
        return;
    }
    if amount > 0 {
        let count = amount as usize;
        let end = list.data.len();
        for _ in 0..count {
            list.data.insert(start, list.data[start].clone());
        }
        list.data.truncate(end + count);
    } else {
        let count = (-amount) as usize;
        if start >= count {
            for index in start..list.data.len() {
                list.data[index - count] = list.data[index].clone();
            }
            list.data.truncate(list.data.len() - count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_list_orders_copies_and_removes_items() {
        let mut list = ilist_new::<i32>(1);
        ilist_append(&mut list, 4);
        ilist_append(&mut list, 9);
        assert_eq!(ilist_insert(&mut list, 2, 1), Ok(()));
        assert_eq!(ilist_size(&list), 3);
        assert_eq!(ilist_item(&mut list, 0), Some(&mut 4));
        assert_eq!(ilist_item(&mut list, 1), Some(&mut 2));
        assert_eq!(ilist_item(&mut list, 2), Some(&mut 9));
        assert!(ilist_swap(&mut list, 0, 2));
        assert_eq!(ilist_first(&mut list), Some(&mut 9));
        assert_eq!(ilist_next(&mut list), Some(&mut 2));
        assert_eq!(ilist_last(&mut list), Some(&mut 4));
        assert!(ilist_float(&mut list, 2));
        assert_eq!(ilist_first(&mut list), Some(&mut 4));
        assert_eq!(ilist_pop(&mut list), Some(4));
        assert_eq!(ilist_remove(&mut list, 0), Some(9));
        assert_eq!(list.data, [2]);
        assert_eq!(ilist_dup(&list).data, [2]);
    }

    #[test]
    fn invalid_indices_leave_typed_list_unchanged() {
        let mut list = ilist_new::<String>(0);
        assert_eq!(ilist_insert(&mut list, "one".into(), 1), Err("one".into()));
        assert_eq!(ilist_remove(&mut list, 0), None);
        assert!(!ilist_swap(&mut list, 0, 1));
        assert!(!ilist_float(&mut list, 0));
        assert_eq!(ilist_pop(&mut list), None);
    }
    #[test]
    fn source_shift_and_owned_delete_have_safe_forms() {
        let mut list = ilist_new(0);
        list.data = vec![1, 2, 3];
        ilist_shift(&mut list, 1, -1);
        assert_eq!(list.data, vec![2, 3]);
        ilist_delete(list);
    }
}
