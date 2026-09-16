//! Translation of `IMOD/raptor/suitesparse/cs_pvec.c`.

/// C `cs_pvec`: writes `x[k] = b[p[k]]`, or copies `b` when `p` is absent.
pub fn cs_pvec(permutation: Option<&[usize]>, input: &[f64], output: &mut [f64]) -> bool {
    let count = output.len();
    if input.len() < count || permutation.is_some_and(|values| values.len() < count) {
        return false;
    }
    for k in 0..count {
        output[k] = input[permutation.map_or(k, |values| values[k])];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::cs_pvec;

    #[test]
    fn pvec_matches_c_permutation_direction() {
        let mut output = [0.0; 3];
        assert!(cs_pvec(Some(&[2, 0, 1]), &[3.0, 5.0, 7.0], &mut output));
        assert_eq!(output, [7.0, 3.0, 5.0]);
    }
}
