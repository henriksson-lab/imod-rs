//! Translation of `IMOD/libwarp/istack.c` and `istack.h`.
#![allow(dead_code)]

/// C `struct istack` (`istack.h`).
#[repr(C)]
pub struct Istack {
    pub n: i32,
    pub nallocated: i32,
    pub v: *mut i32,
}

/// Original `istack_create` (`istack.c:28`).
pub unsafe fn istack_create() -> *mut Istack {
    unsafe {
        let stack = libc::malloc(core::mem::size_of::<Istack>()).cast::<Istack>();
        (*stack).n = 0;
        (*stack).nallocated = 50;
        (*stack).v = libc::malloc(50 * core::mem::size_of::<i32>()).cast();
        stack
    }
}

/// Original `istack_destroy` (`istack.c:38`).
pub unsafe fn istack_destroy(stack: *mut Istack) {
    unsafe {
        if !stack.is_null() {
            libc::free((*stack).v.cast());
            libc::free(stack.cast());
        }
    }
}

/// Original `istack_reset` (`istack.c:47`).
pub unsafe fn istack_reset(stack: *mut Istack) {
    unsafe { (*stack).n = 0 }
}

/// Original `istack_contains` (`istack.c:52`).
pub unsafe fn istack_contains(stack: *mut Istack, value: i32) -> i32 {
    unsafe {
        for index in 0..(*stack).n {
            if *(*stack).v.add(index as usize) == value {
                return 1;
            }
        }
        0
    }
}

/// Original `istack_push` (`istack.c:63`).
pub unsafe fn istack_push(stack: *mut Istack, value: i32) {
    unsafe {
        if (*stack).n == (*stack).nallocated {
            (*stack).nallocated *= 2;
            (*stack).v = libc::realloc(
                (*stack).v.cast(),
                (*stack).nallocated as usize * core::mem::size_of::<i32>(),
            )
            .cast();
        }
        *(*stack).v.add((*stack).n as usize) = value;
        (*stack).n += 1;
    }
}

/// Original `istack_pop` (`istack.c:74`).
pub unsafe fn istack_pop(stack: *mut Istack) -> i32 {
    unsafe {
        (*stack).n -= 1;
        *(*stack).v.add((*stack).n as usize)
    }
}

/// Original `istack_getnentries` (`istack.c:80`).
pub unsafe fn istack_getnentries(stack: *mut Istack) -> i32 {
    unsafe { (*stack).n }
}

/// Original `istack_getentries` (`istack.c:85`).
pub unsafe fn istack_getentries(stack: *mut Istack) -> *mut i32 {
    unsafe { (*stack).v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_in_source_sized_increments_and_lifo_order() {
        unsafe {
            let stack = istack_create();
            for value in 0..51 {
                istack_push(stack, value);
            }
            assert_eq!((*stack).nallocated, 100);
            assert_eq!(istack_getnentries(stack), 51);
            assert_eq!(istack_contains(stack, 17), 1);
            assert_eq!(istack_pop(stack), 50);
            istack_reset(stack);
            assert_eq!(istack_getnentries(stack), 0);
            istack_destroy(stack);
        }
    }
}
