//! Owned Rust translation of `IMOD/raptor/suitesparse/cs_malloc.c`.
//!
//! CSparse's C wrappers centralize raw allocation and use a null pointer for
//! failure.  The Rust representation owns `Vec` storage, so these source
//! counterparts use `Option` without introducing raw allocation or free.

/// C `cs_malloc`, reserving at least one slot as in `CS_MAX(n, 1)`.
pub fn cs_malloc<T>(count: usize) -> Option<Vec<T>> {
    let mut allocation = Vec::new();
    allocation
        .try_reserve_exact(count.max(1))
        .ok()
        .map(|()| allocation)
}

/// C `cs_calloc`, allocating initialized storage at least one element long.
pub fn cs_calloc<T: Default + Clone>(count: usize) -> Option<Vec<T>> {
    let count = count.max(1);
    let mut allocation = Vec::new();
    allocation.try_reserve_exact(count).ok()?;
    allocation.resize(count, T::default());
    Some(allocation)
}

/// C `cs_free`; consuming an owned allocation is its Rust equivalent.
pub fn cs_free<T>(allocation: Option<Vec<T>>) -> Option<Vec<T>> {
    drop(allocation);
    None
}

/// C `cs_realloc`, preserving the old allocation when growth cannot reserve.
pub fn cs_realloc<T>(mut allocation: Vec<T>, count: usize) -> (Vec<T>, bool) {
    let capacity = count.max(1);
    if capacity > allocation.capacity()
        && allocation
            .try_reserve_exact(capacity - allocation.capacity())
            .is_err()
    {
        return (allocation, false);
    }
    (allocation, true)
}

#[cfg(test)]
mod tests {
    use super::{cs_calloc, cs_free, cs_malloc, cs_realloc};

    #[test]
    fn allocation_wrappers_keep_csparse_minimum_capacity_and_owned_release() {
        assert!(cs_malloc::<u8>(0).unwrap().capacity() >= 1);
        assert_eq!(cs_calloc::<i32>(3), Some(vec![0, 0, 0]));
        assert_eq!(cs_free(Some(vec![1_u8, 2])), None);
    }

    #[test]
    fn reallocation_preserves_initialized_elements() {
        let (values, ok) = cs_realloc(vec![3_i32, 4], 10);
        assert!(ok);
        assert_eq!(values, [3, 4]);
        assert!(values.capacity() >= 10);
    }
}
