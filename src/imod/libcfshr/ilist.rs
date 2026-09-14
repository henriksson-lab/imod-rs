//! Translation of `IMOD/include/ilist.h` and `IMOD/libcfshr/ilist.c`.
#![allow(dead_code)]

pub const LIST_QUANTUM: i32 = 1;

/// C `Ilist` (`ilist.h`), in declaration order.
///
/// Only the storage changes from the C struct: `void *data` becomes a `Vec<u8>`
/// of the same bytes.  `dsize`, `current`, `size`, `store` and `quantum` stay
/// exactly as they are, because the quantum growth policy and the byte-offset
/// element addressing are observable through `ilistNew`, `ilistAppend`,
/// `ilistFirst` and `ilistNext` — they are behaviour, not implementation.  The
/// list is deliberately *not* generic: the C is type-erased and its users store
/// several different element types through the same struct.
///
/// `#[repr(C)]` is dropped rather than kept as documentation: a `Vec<u8>` is not
/// a `void *`, so the layout no longer describes the C struct even though the
/// field order still does.
#[derive(Clone)]
pub struct Ilist {
    pub data: Vec<u8>,
    pub dsize: i32,
    pub current: i32,
    pub size: i32,
    pub store: i32,
    pub quantum: i32,
}

/// Matches C `ilistNew(int, int)` (`ilist.c:21`).
///
/// C returns NULL when either `malloc` fails; Rust's allocator aborts instead,
/// so the `None` arm exists to keep the caller's error path rather than because
/// it can be reached.
pub fn ilist_new(dsize: i32, asize: i32) -> Option<Box<Ilist>> {
    let mut list = Box::new(Ilist {
        data: Vec::new(),
        dsize,
        current: -1,
        size: 0,
        store: asize,
        quantum: LIST_QUANTUM,
    });
    if asize != 0 {
        list.data = vec![0_u8; (asize * dsize) as usize];
    }
    Some(list)
}

/// Matches C `ilistTruncate(Ilist *, int)` (`ilist.c:45`).
pub fn ilist_truncate(list: &mut Ilist, size: i32) {
    if size >= 0 && size < list.size {
        list.size = size;
    }
}

/// Matches C `ilistQuantum(Ilist *, int)` (`ilist.c:52`).
pub fn ilist_quantum(list: &mut Ilist, size: i32) {
    if size > 0 {
        list.quantum = size;
    }
}

/// Matches C `ilistDup(Ilist *)` (`ilist.c:59`).
pub fn ilist_dup(list: Option<&Ilist>) -> Option<Box<Ilist>> {
    let list = list?;

    /* First get a new list, then copy the structure, saving the data pointer */
    let mut new_list = ilist_new(list.dsize, list.store)?;
    let data_save = core::mem::take(&mut new_list.data);
    new_list.dsize = list.dsize;
    new_list.current = list.current;
    new_list.size = list.size;
    new_list.store = list.store;
    new_list.quantum = list.quantum;
    new_list.data = data_save;
    if list.size != 0 {
        let count = (list.dsize * list.size) as usize;
        new_list.data[..count].copy_from_slice(&list.data[..count]);
    }
    Some(new_list)
}

/// Matches C `ilistDelete(Ilist *)` (`ilist.c:79`).
pub fn ilist_delete(list: Option<Box<Ilist>>) {
    if list.is_none() {
        return;
    }
    drop(list);
}

/// Matches C `ilistFirst(Ilist *)` (`ilist.c:89`).
pub fn ilist_first(list: Option<&mut Ilist>) -> Option<&mut [u8]> {
    let list = list?;
    if list.size == 0 {
        return None;
    }
    list.current = 0;
    let dsize = list.dsize as usize;
    Some(&mut list.data[..dsize])
}

/// Matches C `ilistNext(Ilist *)` (`ilist.c:100`).
pub fn ilist_next(list: &mut Ilist) -> Option<&mut [u8]> {
    list.current += 1;
    if list.current >= list.size {
        return None;
    }
    let rptr = (list.current * list.dsize) as usize;
    Some(&mut list.data[rptr..rptr + list.dsize as usize])
}

/// Matches C `ilistLast(Ilist *)` (`ilist.c:113`).
pub fn ilist_last(list: Option<&mut Ilist>) -> Option<&mut [u8]> {
    let list = list?;
    if list.size == 0 {
        return None;
    }
    list.current = list.size - 1;
    let rptr = (list.current * list.dsize) as usize;
    Some(&mut list.data[rptr..rptr + list.dsize as usize])
}

/// Matches C `ilistItem(Ilist *, int)` (`ilist.c:128`).
pub fn ilist_item(list: Option<&mut Ilist>, element: i32) -> Option<&mut [u8]> {
    let list = list?;
    if list.size == 0 || element < 0 || element >= list.size {
        return None;
    }
    list.current = element;
    let rptr = (list.current * list.dsize) as usize;
    Some(&mut list.data[rptr..rptr + list.dsize as usize])
}

/// Matches C `ilistSize(Ilist *)` (`ilist.c:140`).
pub fn ilist_size(list: Option<&Ilist>) -> i32 {
    match list {
        None => 0,
        Some(list) => list.size,
    }
}

/// Matches C `ilistAppend(Ilist *, void *)` (`ilist.c:149`).
///
/// `data` is the item to copy in; the first `list.dsize` bytes are taken, which
/// is what the C `memcpy` reads through the `void *`.
pub fn ilist_append(list: &mut Ilist, data: &[u8]) -> i32 {
    if list.store <= list.size {
        // The C reallocs to `dsize * (size + quantum)` when something is already
        // stored and mallocs `dsize * quantum` when nothing is; `store` can only
        // be 0 here when `size` is too, so the two are the same length.
        list.data
            .resize((list.dsize * (list.size + list.quantum)) as usize, 0);
        list.store = list.size + list.quantum;
    }

    let to = (list.size * list.dsize) as usize;
    list.size += 1;
    list.data[to..to + list.dsize as usize].copy_from_slice(&data[..list.dsize as usize]);
    0
}

/// Matches C `ilistRemove(Ilist *, int)` (`ilist.c:171`).
pub fn ilist_remove(list: &mut Ilist, element: i32) {
    if list.size <= element {
        return;
    }

    ilist_shift(list, element + 1, -1);
    list.size -= 1;
}

/// Matches C `ilistSwap(Ilist *, int, int)` (`ilist.c:180`).
pub fn ilist_swap(list: &mut Ilist, e1: i32, e2: i32) -> i32 {
    if e1 >= list.size || e2 >= list.size {
        return 1;
    }

    let p1 = (e1 * list.dsize) as usize;
    let p2 = (e2 * list.dsize) as usize;
    let dsize = list.dsize as usize;
    let temporary = list.data[p1..p1 + dsize].to_vec();
    list.data.copy_within(p2..p2 + dsize, p1);
    list.data[p2..p2 + dsize].copy_from_slice(&temporary);
    0
}

/// Matches C `ilistPush(Ilist *, void *)` (`ilist.c:205`).
pub fn ilist_push(list: &mut Ilist, data: &[u8]) -> i32 {
    ilist_insert(list, data, 0)
}

/// Matches C `ilistPop(Ilist *)` (`ilist.c:212`).
///
/// The C hands back a `malloc`ed copy the caller must free; the `Vec` owns it.
pub fn ilist_pop(list: &mut Ilist) -> Option<Vec<u8>> {
    if list.size == 0 {
        return None;
    }
    let data = list.data[..list.dsize as usize].to_vec();
    ilist_remove(list, 0);
    Some(data)
}

/// Matches C `ilistFloat(Ilist *, int)` (`ilist.c:230`).
pub fn ilist_float(list: &mut Ilist, element: i32) -> i32 {
    if element < 1 || element >= list.size {
        return 1;
    }

    let p1 = (element * list.dsize) as usize;
    let data = list.data[p1..p1 + list.dsize as usize].to_vec();
    ilist_remove(list, element);
    ilist_push(list, &data)
}

/// Matches C `ilistInsert(Ilist *, void *, int)` (`ilist.c:252`).
pub fn ilist_insert(list: &mut Ilist, data: &[u8], element: i32) -> i32 {
    if element < 0 || element > list.size {
        return 1;
    }

    if ilist_append(list, data) != 0 {
        return 1;
    }
    if element == list.size - 1 {
        return 0;
    }

    ilist_shift(list, element, 1);
    let to = (element * list.dsize) as usize;
    list.data[to..to + list.dsize as usize].copy_from_slice(&data[..list.dsize as usize]);
    0
}

/// Matches C `ilistShift(Ilist *, int, int)` (`ilist.c:273`).
pub fn ilist_shift(list: &mut Ilist, start: i32, amount: i32) {
    let (lst, lnd, ldir) = if amount > 0 {
        (list.size - 1 - amount, start, -1)
    } else {
        (start, list.size - 1, 1)
    };

    let mut l = lst;
    while ldir * (l - lnd) <= 0 {
        let from = (l * list.dsize) as usize;
        let to = ((l + amount) * list.dsize) as usize;
        list.data.copy_within(from..from + list.dsize as usize, to);
        l += ldir;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_allocation_copy_ordering_and_removal_match_source() {
        let mut list = ilist_new(core::mem::size_of::<i32>() as i32, 1).unwrap();
        let first = 4_i32;
        let second = 9_i32;
        let third = 2_i32;
        assert_eq!(ilist_append(&mut list, &first.to_ne_bytes()), 0);
        assert_eq!(ilist_append(&mut list, &second.to_ne_bytes()), 0);
        assert_eq!(ilist_insert(&mut list, &third.to_ne_bytes(), 1), 0);
        assert_eq!(ilist_size(Some(&list)), 3);
        let item = |list: &mut Ilist, index| {
            i32::from_ne_bytes(ilist_item(Some(list), index).unwrap().try_into().unwrap())
        };
        assert_eq!(item(&mut list, 0), 4);
        assert_eq!(item(&mut list, 1), 2);
        assert_eq!(item(&mut list, 2), 9);
        assert_eq!(ilist_swap(&mut list, 0, 2), 0);
        assert_eq!(
            i32::from_ne_bytes(ilist_first(Some(&mut list)).unwrap().try_into().unwrap()),
            9
        );
        assert_eq!(
            i32::from_ne_bytes(ilist_next(&mut list).unwrap().try_into().unwrap()),
            2
        );
        assert_eq!(
            i32::from_ne_bytes(ilist_last(Some(&mut list)).unwrap().try_into().unwrap()),
            4
        );
        assert_eq!(ilist_float(&mut list, 2), 0);
        assert_eq!(
            i32::from_ne_bytes(ilist_first(Some(&mut list)).unwrap().try_into().unwrap()),
            4
        );
        let popped = ilist_pop(&mut list).unwrap();
        assert_eq!(i32::from_ne_bytes(popped.try_into().unwrap()), 4);
        ilist_remove(&mut list, 0);
        assert_eq!(ilist_size(Some(&list)), 1);
        assert_eq!(item(&mut list, 0), 2);
        let mut duplicate = ilist_dup(Some(&list)).unwrap();
        assert_eq!(item(&mut duplicate, 0), 2);
        ilist_delete(Some(duplicate));
        ilist_delete(Some(list));
    }
}
