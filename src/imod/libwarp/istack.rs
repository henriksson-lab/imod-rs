//! Translation of `IMOD/libwarp/istack.c` and `istack.h`.
//!
//! The source's `int* v` grown by `realloc` becomes a `Vec<i32>` that holds
//! exactly the pushed entries: `n` is `v.len()`, and `nallocated` — which
//! `istack_push` doubled whenever `n` reached it — is the `Vec`'s own
//! capacity, seeded at `STACK_NSTART`.  Nothing reads either number apart
//! from the stack's own routines, so the growth policy is not observable.

/// C `STACK_NSTART` (`istack.c:17`).
const STACK_NSTART: i32 = 50;

/// C `struct istack` (`istack.h`); `n` is `v.len()`.
pub struct Istack {
    pub v: Vec<i32>,
}

/// Original `istack_create` (`istack.c:28`).
pub fn istack_create() -> Istack {
    let mut s = Istack { v: Vec::new() };

    // `s->n = 0; s->nallocated = STACK_NSTART;
    // s->v = malloc(STACK_NSTART * sizeof(int))`: the source leaves the
    // elements past `n` uninitialised and never reads them.
    s.v = Vec::with_capacity(STACK_NSTART as usize);
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
    /* s->n = 0 */
    s.v.clear();
}

/// Original `istack_contains` (`istack.c:52`).
pub fn istack_contains(s: &Istack, v: i32) -> i32 {
    for i in 0..s.v.len() {
        if s.v[i] == v {
            return 1;
        }
    }
    0
}

/// Original `istack_push` (`istack.c:63`).
pub fn istack_push(s: &mut Istack, v: i32) {
    /* The source doubles `nallocated` with `realloc` when `n` reaches it,
    then stores at `v[n++]`. */
    s.v.push(v);
}

/// Original `istack_pop` (`istack.c:74`).
pub fn istack_pop(s: &mut Istack) -> i32 {
    /* `s->n--; return s->v[s->n];` — on an empty stack the source reads
    `v[-1]`, which has no defined value; this panics instead. */
    s.v.pop().unwrap()
}

/// Original `istack_getnentries` (`istack.c:80`).
pub fn istack_getnentries(s: &Istack) -> i32 {
    s.v.len() as i32
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
        assert_eq!(istack_getnentries(&stack), 51);
        assert_eq!(istack_contains(&stack, 17), 1);
        assert_eq!(istack_contains(&stack, 51), 0);
        assert_eq!(istack_getentries(&stack)[17], 17);
        assert_eq!(istack_pop(&mut stack), 50);
        istack_reset(&mut stack);
        assert_eq!(istack_getnentries(&stack), 0);
        istack_destroy(Some(stack));
        istack_destroy(None);
    }
}
