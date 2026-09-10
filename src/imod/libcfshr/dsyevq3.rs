//! Translation of `IMOD/libcfshr/dsyevq3.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::dsytrd3::dsytrd3;

/// Original `dsyevq3` (`dsyevq3.c:32`).
pub fn dsyevq3(
    matrix: &[[f64; 3]; 3],
    orthogonal: &mut [[f64; 3]; 3],
    eigenvalues: &mut [f64; 3],
) -> i32 {
    let mut off_diagonal = [0.0; 3];
    let mut reduced_off_diagonal = [0.0; 2];
    dsytrd3(matrix, orthogonal, eigenvalues, &mut reduced_off_diagonal);
    off_diagonal[0] = reduced_off_diagonal[0];
    off_diagonal[1] = reduced_off_diagonal[1];
    for lower in 0..2 {
        let mut iterations = 0;
        loop {
            let mut middle = lower;
            while middle <= 1 {
                let g = eigenvalues[middle].abs() + eigenvalues[middle + 1].abs();
                if off_diagonal[middle].abs() + g == g {
                    break;
                }
                middle += 1;
            }
            if middle == lower {
                break;
            }
            if iterations >= 30 {
                return -1;
            }
            iterations += 1;
            let mut g = (eigenvalues[lower + 1] - eigenvalues[lower])
                / (off_diagonal[lower] + off_diagonal[lower]);
            let mut radius = (g * g + 1.0).sqrt();
            g = if g > 0.0 {
                eigenvalues[middle] - eigenvalues[lower] + off_diagonal[lower] / (g + radius)
            } else {
                eigenvalues[middle] - eigenvalues[lower] + off_diagonal[lower] / (g - radius)
            };
            let mut sine = 1.0;
            let mut cosine = 1.0;
            let mut p = 0.0;
            for index in (lower..middle).rev() {
                let f = sine * off_diagonal[index];
                let b = cosine * off_diagonal[index];
                if f.abs() > g.abs() {
                    cosine = g / f;
                    radius = (cosine * cosine + 1.0).sqrt();
                    off_diagonal[index + 1] = f * radius;
                    sine = 1.0 / radius;
                    cosine *= sine;
                } else {
                    sine = f / g;
                    radius = (sine * sine + 1.0).sqrt();
                    off_diagonal[index + 1] = g * radius;
                    cosine = 1.0 / radius;
                    sine *= cosine;
                }
                g = eigenvalues[index + 1] - p;
                radius = (eigenvalues[index] - g) * sine + 2.0 * cosine * b;
                p = sine * radius;
                eigenvalues[index + 1] = g + p;
                g = cosine * radius - b;
                for component in 0..3 {
                    let temporary = orthogonal[component][index + 1];
                    orthogonal[component][index + 1] =
                        sine * orthogonal[component][index] + cosine * temporary;
                    orthogonal[component][index] =
                        cosine * orthogonal[component][index] - sine * temporary;
                }
            }
            eigenvalues[lower] -= p;
            off_diagonal[lower] = g;
            off_diagonal[middle] = 0.0;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ql_stage_preserves_diagonal_eigenvectors_and_converges_general_input() {
        let diagonal = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let mut q = [[0.0; 3]; 3];
        let mut values = [0.0; 3];
        assert_eq!(dsyevq3(&diagonal, &mut q, &mut values), 0);
        assert_eq!(values, [2.0, 3.0, 5.0]);
        assert_eq!(q, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let general = [[4.0, 1.0, 2.0], [1.0, 3.0, 5.0], [2.0, 5.0, 6.0]];
        assert_eq!(dsyevq3(&general, &mut q, &mut values), 0);
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
