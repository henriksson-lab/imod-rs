//! Translation of `IMOD/raptor/suitesparse/cs_ipvec.c`.

/// C `cs_ipvec`: writes `x[p[k]] = b[k]`, or copies `b` when `p` is absent.
pub fn cs_ipvec(permutation: Option<&[usize]>, input: &[f64], output: &mut [f64]) -> bool {
    let count = output.len();
    if input.len() < count || permutation.is_some_and(|values| values.len() < count) {
        return false;
    }
    for k in 0..count {
        let destination = permutation.map_or(k, |values| values[k]);
        if destination >= count {
            return false;
        }
        output[destination] = input[k];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::cs_ipvec;

    #[test]
    fn ipvec_matches_c_inverse_permutation_direction() {
        let mut output = [0.0; 3];
        assert!(cs_ipvec(Some(&[2, 0, 1]), &[3.0, 5.0, 7.0], &mut output));
        assert_eq!(output, [5.0, 7.0, 3.0]);
    }
}
