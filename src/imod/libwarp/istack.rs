//! Translation of `IMOD/libwarp/istack.c` and `istack.h`.
//!
//! The source's `int* v` grown by `realloc` becomes a `Vec<i32>`, but the
//! `n`/`nallocated` pair stays: `istack_push` doubles `nallocated` only when
//! `n` reaches it, and keeping that policy is what lets the translation still
//! read like the source even though `Vec` would grow on its own.
#![allow(dead_code)]

/// C `STACK_NSTART` (`istack.c:17`).
const STACK_NSTART: i32 = 50;
/// C `STACK_NINC` (`istack.c:18`); declared by the source and unused by it.
const STACK_NINC: i32 = 50;

/// C `struct istack` (`istack.h`).
pub struct Istack {
    pub n: i32,
    pub nallocated: i32,
    pub v: Vec<i32>,
}

/// Original `istack_create` (`istack.c:28`).
pub fn istack_create() -> Istack {
    let mut s = Istack {
        n: 0,
        nallocated: 0,
        v: Vec::new(),
    };

    s.n = 0;
    s.nallocated = STACK_NSTART;
    // `malloc(STACK_NSTART * sizeof(int))`: the source leaves the elements
    // past `n` uninitialised and never reads them.
    s.v = vec![0; STACK_NSTART as usize];
    s
}

/// Original `istack_destroy` (`istack.c:38`).
pub fn istack_destroy(s: Option<Istack>) {
    if s.is_some() {
        // Dropping the owned vector is the source's `free(s->v)` then
        // `free(s)`.
        drop(s);
    }
}

/// Original `istack_reset` (`istack.c:47`).
pub fn istack_reset(s: &mut Istack) {
    s.n = 0;
}

/// Original `istack_contains` (`istack.c:52`).
pub fn istack_contains(s: &Istack, v: i32) -> i32 {
    for i in 0..s.n {
        if s.v[i as usize] == v {
            return 1;
        }
    }
    0
}

/// Original `istack_push` (`istack.c:63`).
pub fn istack_push(s: &mut Istack, v: i32) {
    if s.n == s.nallocated {
        s.nallocated *= 2;
        s.v.resize(s.nallocated as usize, 0);
    }

    s.v[s.n as usize] = v;
    s.n += 1;
}

/// Original `istack_pop` (`istack.c:74`).
pub fn istack_pop(s: &mut Istack) -> i32 {
    s.n -= 1;
    s.v[s.n as usize]
}

/// Original `istack_getnentries` (`istack.c:80`).
pub fn istack_getnentries(s: &Istack) -> i32 {
    s.n
}

/// Original `istack_getentries` (`istack.c:85`).
pub fn istack_getentries(s: &Istack) -> &[i32] {
    &s.v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_in_source_sized_increments_and_lifo_order() {
        let mut stack = istack_create();
        for value in 0..51 {
            istack_push(&mut stack, value);
        }
        assert_eq!(stack.nallocated, 100);
        assert_eq!(istack_getnentries(&stack), 51);
        assert_eq!(istack_contains(&stack, 17), 1);
        assert_eq!(istack_contains(&stack, 51), 0);
        assert_eq!(istack_getentries(&stack)[17], 17);
        assert_eq!(istack_pop(&mut stack), 50);
        istack_reset(&mut stack);
        assert_eq!(istack_getnentries(&stack), 0);
        istack_destroy(Some(stack));
        istack_destroy(None);
        assert_eq!(STACK_NINC, 50);
    }
}
