//! Translation of `IMOD/raptor/suitesparse/cs_randperm.c`.

/// C `cs_randperm`: returns an absent identity permutation, a reverse permutation,
/// or a seeded Fisher--Yates permutation.
///
/// As in CSparse, seed zero denotes the identity permutation and is represented by
/// `None`; seed minus one produces the reverse permutation.  The original calls
/// the platform C library's `srand`/`rand`, whose sequence is not portable.  This
/// safe translation instead specifies its seeded result with the traditional
/// 31-bit C linear-congruential recurrence, so a nonzero seed has the same result
/// on every Rust-supported platform.
pub fn cs_randperm(size: usize, seed: i32) -> Option<Vec<usize>> {
    if seed == 0 {
        return None;
    }

    let mut permutation: Vec<usize> = (0..size).rev().collect();
    if seed == -1 {
        return Some(permutation);
    }

    let mut state = seed as u32;
    for index in 0..size {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let random = (state >> 16) & 0x7fff;
        let destination = index + (random as usize % (size - index));
        permutation.swap(index, destination);
    }
    Some(permutation)
}

#[cfg(test)]
mod tests {
    use super::cs_randperm;

    #[test]
    fn randperm_keeps_csparse_identity_and_reverse_sentinels() {
        assert_eq!(cs_randperm(4, 0), None);
        assert_eq!(cs_randperm(4, -1), Some(vec![3, 2, 1, 0]));
        assert_eq!(cs_randperm(0, -1), Some(vec![]));
    }

    #[test]
    fn randperm_is_seeded_and_platform_independent() {
        let first = cs_randperm(8, 42).unwrap();
        assert_eq!(first, cs_randperm(8, 42).unwrap());
        assert_eq!(first, vec![6, 4, 0, 3, 7, 1, 5, 2]);
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }
}
