//! Translation of `IMOD/raptor/suitesparse/cs_malloc.c`.  `cs_free` is
//! `Drop`; the allocators return owned `Vec`s of `CS_MAX(n,1)` elements.

/// `cs_malloc(n, size)`: `CS_MAX(n,1)` elements (zero here where C leaves
/// them uninitialised).
pub fn cs_malloc<T: Default + Clone>(n: i32) -> Vec<T> {
    vec![T::default(); if n > 1 { n as usize } else { 1 }]
}

/// `cs_calloc(n, size)`: `CS_MAX(n,1)` zeroed elements.
pub fn cs_calloc<T: Default + Clone>(n: i32) -> Vec<T> {
    vec![T::default(); if n > 1 { n as usize } else { 1 }]
}

/// `cs_realloc(p, n, size, &ok)`: resizes `p` to `CS_MAX(n,1)` elements,
/// keeping the leading ones; allocation cannot fail here, so `ok` is 1.
pub fn cs_realloc<T: Default + Clone>(p: &mut Vec<T>, n: i32, ok: &mut i32) {
    p.resize(if n > 1 { n as usize } else { 1 }, T::default());
    *ok = 1;
}
