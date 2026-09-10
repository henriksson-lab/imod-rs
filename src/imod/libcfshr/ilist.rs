//! Translation of `IMOD/include/ilist.h` and `IMOD/libcfshr/ilist.c`.
#![allow(dead_code)]

use core::ffi::c_void;

pub const LIST_QUANTUM: i32 = 1;

/// C `Ilist` (`ilist.h`), in declaration order.
#[repr(C)]
pub struct Ilist {
    pub data: *mut c_void,
    pub dsize: i32,
    pub current: i32,
    pub size: i32,
    pub store: i32,
    pub quantum: i32,
}

/// Matches C `ilistNew(int, int)` (`ilist.c:21`).
pub unsafe fn ilist_new(dsize: i32, asize: i32) -> *mut Ilist {
    let list = unsafe { libc::malloc(core::mem::size_of::<Ilist>()).cast::<Ilist>() };
    if list.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        (*list).dsize = dsize;
        (*list).current = -1;
        (*list).size = 0;
        (*list).store = asize;
        (*list).data = core::ptr::null_mut();
        (*list).quantum = LIST_QUANTUM;
        if asize != 0 {
            (*list).data = libc::malloc((asize * dsize) as usize);
            if (*list).data.is_null() {
                ilist_delete(list);
                return core::ptr::null_mut();
            }
        }
    }
    list
}

/// Matches C `ilistTruncate(Ilist *, int)` (`ilist.c:45`).
pub unsafe fn ilist_truncate(list: *mut Ilist, size: i32) {
    if size >= 0 && size < unsafe { (*list).size } {
        unsafe { (*list).size = size };
    }
}

/// Matches C `ilistQuantum(Ilist *, int)` (`ilist.c:52`).
pub unsafe fn ilist_quantum(list: *mut Ilist, size: i32) {
    if size > 0 {
        unsafe { (*list).quantum = size };
    }
}

/// Matches C `ilistDup(Ilist *)` (`ilist.c:59`).
pub unsafe fn ilist_dup(list: *mut Ilist) -> *mut Ilist {
    if list.is_null() {
        return core::ptr::null_mut();
    }
    let new_list = unsafe { ilist_new((*list).dsize, (*list).store) };
    if new_list.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        let data_save = (*new_list).data;
        core::ptr::copy_nonoverlapping(list, new_list, 1);
        (*new_list).data = data_save;
        if (*list).size != 0 {
            core::ptr::copy_nonoverlapping(
                (*list).data.cast::<u8>(),
                (*new_list).data.cast::<u8>(),
                ((*list).dsize * (*list).size) as usize,
            );
        }
    }
    new_list
}

/// Matches C `ilistDelete(Ilist *)` (`ilist.c:79`).
pub unsafe fn ilist_delete(list: *mut Ilist) {
    if list.is_null() {
        return;
    }
    unsafe {
        libc::free((*list).data);
        libc::free(list.cast::<c_void>());
    }
}

/// Matches C `ilistFirst(Ilist *)` (`ilist.c:89`).
pub unsafe fn ilist_first(list: *mut Ilist) -> *mut c_void {
    if list.is_null() || unsafe { (*list).size } == 0 {
        return core::ptr::null_mut();
    }
    unsafe {
        (*list).current = 0;
        (*list).data
    }
}

/// Matches C `ilistNext(Ilist *)` (`ilist.c:100`).
pub unsafe fn ilist_next(list: *mut Ilist) -> *mut c_void {
    unsafe {
        (*list).current += 1;
        if (*list).current >= (*list).size {
            return core::ptr::null_mut();
        }
        (*list)
            .data
            .cast::<u8>()
            .add(((*list).current * (*list).dsize) as usize)
            .cast::<c_void>()
    }
}

/// Matches C `ilistLast(Ilist *)` (`ilist.c:113`).
pub unsafe fn ilist_last(list: *mut Ilist) -> *mut c_void {
    if list.is_null() || unsafe { (*list).size } == 0 {
        return core::ptr::null_mut();
    }
    unsafe {
        (*list).current = (*list).size - 1;
        (*list)
            .data
            .cast::<u8>()
            .add(((*list).current * (*list).dsize) as usize)
            .cast::<c_void>()
    }
}

/// Matches C `ilistItem(Ilist *, int)` (`ilist.c:128`).
pub unsafe fn ilist_item(list: *mut Ilist, element: i32) -> *mut c_void {
    if list.is_null()
        || unsafe { (*list).size } == 0
        || element < 0
        || element >= unsafe { (*list).size }
    {
        return core::ptr::null_mut();
    }
    unsafe {
        (*list).current = element;
        (*list)
            .data
            .cast::<u8>()
            .add(((*list).current * (*list).dsize) as usize)
            .cast::<c_void>()
    }
}

/// Matches C `ilistSize(Ilist *)` (`ilist.c:140`).
pub unsafe fn ilist_size(list: *mut Ilist) -> i32 {
    if list.is_null() {
        return 0;
    }
    unsafe { (*list).size }
}

/// Matches C `ilistAppend(Ilist *, void *)` (`ilist.c:149`).
pub unsafe fn ilist_append(list: *mut Ilist, data: *mut c_void) -> i32 {
    unsafe {
        if (*list).store <= (*list).size {
            let new_data = if (*list).store != 0 {
                libc::realloc(
                    (*list).data,
                    ((*list).dsize * ((*list).size + (*list).quantum)) as usize,
                )
            } else {
                libc::malloc(((*list).dsize * (*list).quantum) as usize)
            };
            if new_data.is_null() {
                return 1;
            }
            (*list).data = new_data;
            (*list).store = (*list).size + (*list).quantum;
        }
        let to = (*list)
            .data
            .cast::<u8>()
            .add(((*list).size * (*list).dsize) as usize);
        (*list).size += 1;
        core::ptr::copy_nonoverlapping(data.cast::<u8>(), to, (*list).dsize as usize);
    }
    0
}

/// Matches C `ilistRemove(Ilist *, int)` (`ilist.c:171`).
pub unsafe fn ilist_remove(list: *mut Ilist, element: i32) {
    if unsafe { (*list).size } <= element {
        return;
    }
    unsafe {
        ilist_shift(list, element + 1, -1);
        (*list).size -= 1;
    }
}

/// Matches C `ilistSwap(Ilist *, int, int)` (`ilist.c:180`).
pub unsafe fn ilist_swap(list: *mut Ilist, e1: i32, e2: i32) -> i32 {
    if e1 >= unsafe { (*list).size } || e2 >= unsafe { (*list).size } {
        return 1;
    }
    unsafe {
        let p1 = (*list).data.cast::<u8>().add((e1 * (*list).dsize) as usize);
        let p2 = (*list).data.cast::<u8>().add((e2 * (*list).dsize) as usize);
        let temporary = libc::malloc((*list).dsize as usize).cast::<u8>();
        if temporary.is_null() {
            return 1;
        }
        core::ptr::copy_nonoverlapping(p1, temporary, (*list).dsize as usize);
        core::ptr::copy_nonoverlapping(p2, p1, (*list).dsize as usize);
        core::ptr::copy_nonoverlapping(temporary, p2, (*list).dsize as usize);
        libc::free(temporary.cast::<c_void>());
    }
    0
}

/// Matches C `ilistPush(Ilist *, void *)` (`ilist.c:205`).
pub unsafe fn ilist_push(list: *mut Ilist, data: *mut c_void) -> i32 {
    unsafe { ilist_insert(list, data, 0) }
}

/// Matches C `ilistPop(Ilist *)` (`ilist.c:212`).
pub unsafe fn ilist_pop(list: *mut Ilist) -> *mut c_void {
    if unsafe { (*list).size } == 0 {
        return core::ptr::null_mut();
    }
    unsafe {
        let data = libc::malloc((*list).dsize as usize);
        if data.is_null() {
            return core::ptr::null_mut();
        }
        core::ptr::copy_nonoverlapping(
            (*list).data.cast::<u8>(),
            data.cast::<u8>(),
            (*list).dsize as usize,
        );
        ilist_remove(list, 0);
        data
    }
}

/// Matches C `ilistFloat(Ilist *, int)` (`ilist.c:230`).
pub unsafe fn ilist_float(list: *mut Ilist, element: i32) -> i32 {
    if element < 1 || element >= unsafe { (*list).size } {
        return 1;
    }
    unsafe {
        let pointer = (*list)
            .data
            .cast::<u8>()
            .add((element * (*list).dsize) as usize);
        let data = libc::malloc((*list).dsize as usize);
        if data.is_null() {
            return 1;
        }
        core::ptr::copy_nonoverlapping(pointer, data.cast::<u8>(), (*list).dsize as usize);
        ilist_remove(list, element);
        let error = ilist_push(list, data);
        libc::free(data);
        error
    }
}

/// Matches C `ilistInsert(Ilist *, void *, int)` (`ilist.c:252`).
pub unsafe fn ilist_insert(list: *mut Ilist, data: *mut c_void, element: i32) -> i32 {
    if element < 0 || element > unsafe { (*list).size } {
        return 1;
    }
    unsafe {
        if ilist_append(list, data) != 0 {
            return 1;
        }
        if element == (*list).size - 1 {
            return 0;
        }
        ilist_shift(list, element, 1);
        let to = (*list)
            .data
            .cast::<u8>()
            .add((element * (*list).dsize) as usize);
        core::ptr::copy_nonoverlapping(data.cast::<u8>(), to, (*list).dsize as usize);
    }
    0
}

/// Matches C `ilistShift(Ilist *, int, int)` (`ilist.c:273`).
pub unsafe fn ilist_shift(list: *mut Ilist, start: i32, amount: i32) {
    let (mut index, last, direction) = if amount > 0 {
        (unsafe { (*list).size } - 1 - amount, start, -1)
    } else {
        (start, unsafe { (*list).size } - 1, 1)
    };
    while direction * (index - last) <= 0 {
        unsafe {
            let from = (*list)
                .data
                .cast::<u8>()
                .add((index * (*list).dsize) as usize);
            let to = from.offset((amount * (*list).dsize) as isize);
            core::ptr::copy_nonoverlapping(from, to, (*list).dsize as usize);
        }
        index += direction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_allocation_copy_ordering_and_removal_match_source() {
        unsafe {
            let list = ilist_new(core::mem::size_of::<i32>() as i32, 1);
            assert!(!list.is_null());
            let mut first = 4_i32;
            let mut second = 9_i32;
            let mut third = 2_i32;
            assert_eq!(ilist_append(list, (&mut first as *mut i32).cast()), 0);
            assert_eq!(ilist_append(list, (&mut second as *mut i32).cast()), 0);
            assert_eq!(ilist_insert(list, (&mut third as *mut i32).cast(), 1), 0);
            assert_eq!(ilist_size(list), 3);
            assert_eq!(*(ilist_item(list, 0).cast::<i32>()), 4);
            assert_eq!(*(ilist_item(list, 1).cast::<i32>()), 2);
            assert_eq!(*(ilist_item(list, 2).cast::<i32>()), 9);
            assert_eq!(ilist_swap(list, 0, 2), 0);
            assert_eq!(*(ilist_first(list).cast::<i32>()), 9);
            assert_eq!(*(ilist_next(list).cast::<i32>()), 2);
            assert_eq!(*(ilist_last(list).cast::<i32>()), 4);
            assert_eq!(ilist_float(list, 2), 0);
            assert_eq!(*(ilist_first(list).cast::<i32>()), 4);
            let popped = ilist_pop(list).cast::<i32>();
            assert_eq!(*popped, 4);
            libc::free(popped.cast());
            ilist_remove(list, 0);
            assert_eq!(ilist_size(list), 1);
            assert_eq!(*(ilist_item(list, 0).cast::<i32>()), 2);
            let duplicate = ilist_dup(list);
            assert_eq!(*(ilist_item(duplicate, 0).cast::<i32>()), 2);
            ilist_delete(duplicate);
            ilist_delete(list);
        }
    }
}
