//! Translation of `IMOD/raptor/suitesparse/cs_pinv.c`.

/// C `cs_pinv`: returns the inverse of a permutation, or `None` for identity/invalid input.
pub fn cs_pinv(permutation: Option<&[usize]>) -> Option<Vec<usize>> {
    let permutation = permutation?;
    let mut inverse = vec![0; permutation.len()];
    for (index, &value) in permutation.iter().enumerate() {
        *inverse.get_mut(value)? = index;
    }
    Some(inverse)
}
