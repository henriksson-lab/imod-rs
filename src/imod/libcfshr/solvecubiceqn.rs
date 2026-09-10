//! Translation of `IMOD/libcfshr/solvecubiceqn.c`.
#![allow(dead_code)]

/// Original `solveCubicEqn` (`solvecubiceqn.c:63`).
pub fn solve_cubic_eqn(
    aa: f64,
    bb: f64,
    cc: f64,
    dd: f64,
    roots: &mut [[f64; 2]; 3],
    magnitude_ratio: &mut f64,
) {
    let qq = (3.0 * aa * cc - bb * bb) / (9.0 * aa * aa);
    let rr = [
        (9.0 * aa * bb * cc - 27.0 * aa * aa * dd - 2.0 * bb * bb * bb) / (54.0 * aa * aa * aa),
        0.0,
    ];
    let q3r2 = [qq * qq * qq + rr[0] * rr[0], 0.0];
    let mut square_roots = [[0.0; 2]; 2];
    let (first_square_root, second_square_root) = square_roots.split_at_mut(1);
    cmplx_sqrt(&q3r2, &mut first_square_root[0], &mut second_square_root[0]);
    let mut positive = [[0.0; 2]; 2];
    cmplx_sum(&rr, &square_roots[0], &mut positive[0]);
    let mut ss = [[0.0; 2]; 3];
    let (first_ss, remaining_ss) = ss.split_at_mut(1);
    let (second_ss, third_ss) = remaining_ss.split_at_mut(1);
    cmplx_cube_root(
        &positive[0],
        &mut first_ss[0],
        &mut second_ss[0],
        &mut third_ss[0],
    );
    let mut negative = [[0.0; 2]; 2];
    cmplx_diff(&rr, &square_roots[0], &mut negative[0]);
    let mut tt = [[0.0; 2]; 3];
    let (first_tt, remaining_tt) = tt.split_at_mut(1);
    let (second_tt, third_tt) = remaining_tt.split_at_mut(1);
    cmplx_cube_root(
        &negative[0],
        &mut first_tt[0],
        &mut second_tt[0],
        &mut third_tt[0],
    );
    let offset = [-bb / (3.0 * aa), 0.0];
    let mut candidate_magnitudes = [0.0_f32; 9];
    let mut candidates = [[0.0; 2]; 9];
    let mut solution_indexes = [0_i32; 9];
    let mut solution_count = 0_usize;
    let mut maximum_magnitude = 0.0;
    for first in 0..3 {
        for second in 0..3 {
            let mut sum = [0.0; 2];
            let mut root = [0.0; 2];
            cmplx_sum(&ss[first], &tt[second], &mut sum);
            cmplx_sum(&sum, &offset, &mut root);
            let mut root_square = [0.0; 2];
            let mut root_cube = [0.0; 2];
            cmplx_square(&root, &mut root_square);
            cmplx_product(&root, &root_square, &mut root_cube);
            let residual = [
                aa * root_cube[0] + bb * root_square[0] + cc * root[0] + dd,
                aa * root_cube[1] + bb * root_square[1] + cc * root[1],
            ];
            candidate_magnitudes[solution_count] =
                (residual[0] * residual[0] + residual[1] * residual[1]) as f32;
            let root_magnitude = root[0] * root[0] + root[1] * root[1];
            if root_magnitude > maximum_magnitude {
                maximum_magnitude = root_magnitude;
            }
            solution_indexes[solution_count] = solution_count as i32;
            candidates[solution_count] = root;
            solution_count += 1;
        }
    }
    let epsilon = maximum_magnitude.sqrt() * 2.0e-6;
    for first in 0..solution_count - 1 {
        for second in first + 1..solution_count {
            if candidate_magnitudes[solution_indexes[second] as usize]
                < candidate_magnitudes[solution_indexes[first] as usize]
            {
                solution_indexes.swap(first, second);
            }
        }
    }
    let mut roots_found = 0_usize;
    let mut last_index = 0;
    for index in 0..9 {
        if roots_found == 3 {
            break;
        }
        if index == 0
            || (candidates[solution_indexes[index] as usize][0]
                - candidates[solution_indexes[index - 1] as usize][0])
                .abs()
                > epsilon
            || (candidates[solution_indexes[index] as usize][1]
                - candidates[solution_indexes[index - 1] as usize][1])
                .abs()
                > epsilon
        {
            roots[roots_found] = candidates[solution_indexes[index] as usize];
            roots_found += 1;
            last_index = index;
        }
    }
    if roots_found == 3 && last_index < 8 {
        *magnitude_ratio = (candidate_magnitudes[solution_indexes[last_index + 1] as usize]
            / candidate_magnitudes[solution_indexes[last_index] as usize])
            .sqrt() as f64;
    } else {
        *magnitude_ratio = -(roots_found as f64);
    }
}

/// Original static `cmplxSum` (`solvecubiceqn.c:191`).
pub fn cmplx_sum(first: &[f64; 2], second: &[f64; 2], sum: &mut [f64; 2]) {
    sum[0] = first[0] + second[0];
    sum[1] = first[1] + second[1];
}
/// Original static `cmplxDiff` (`solvecubiceqn.c:198`).
pub fn cmplx_diff(first: &[f64; 2], second: &[f64; 2], difference: &mut [f64; 2]) {
    difference[0] = first[0] - second[0];
    difference[1] = first[1] - second[1];
}
/// Original static `cmplxProduct` (`solvecubiceqn.c:205`).
pub fn cmplx_product(first: &[f64; 2], second: &[f64; 2], product: &mut [f64; 2]) {
    product[0] = first[0] * second[0] - first[1] * second[1];
    product[1] = first[0] * second[1] + first[1] * second[0];
}
/// Original static `cmplxSquare` (`solvecubiceqn.c:212`).
pub fn cmplx_square(value: &[f64; 2], square: &mut [f64; 2]) {
    cmplx_product(value, value, square);
}
/// Original static `cmplxCube` (`solvecubiceqn.c:218`).
pub fn cmplx_cube(value: &[f64; 2], cube: &mut [f64; 2]) {
    let mut square = [0.0; 2];
    cmplx_square(value, &mut square);
    cmplx_product(value, &square, cube);
}
/// Original static `cmplxToPolar` (`solvecubiceqn.c:226`).
pub fn cmplx_to_polar(value: &[f64; 2], radius: &mut f64, angle: &mut f64) {
    *radius = (value[0] * value[0] + value[1] * value[1]).sqrt();
    *angle = value[1].atan2(value[0]);
}
/// Original static `cmplxFromPolar` (`solvecubiceqn.c:233`).
pub fn cmplx_from_polar(radius: f64, angle: f64, value: &mut [f64; 2]) {
    value[0] = radius * angle.cos();
    value[1] = radius * angle.sin();
}
/// Original static `cmplxSqrt` (`solvecubiceqn.c:240`).
pub fn cmplx_sqrt(value: &[f64; 2], first_root: &mut [f64; 2], second_root: &mut [f64; 2]) {
    let mut radius = 0.0;
    let mut angle = 0.0;
    cmplx_to_polar(value, &mut radius, &mut angle);
    radius = radius.sqrt();
    angle /= 2.0;
    cmplx_from_polar(radius, angle, first_root);
    cmplx_from_polar(radius, angle + 3.1415926535, second_root);
}
/// Original static `cmplxCubeRoot` (`solvecubiceqn.c:252`).
pub fn cmplx_cube_root(
    value: &[f64; 2],
    first_root: &mut [f64; 2],
    second_root: &mut [f64; 2],
    third_root: &mut [f64; 2],
) {
    let mut radius = 0.0;
    let mut angle = 0.0;
    cmplx_to_polar(value, &mut radius, &mut angle);
    radius = radius.powf(1.0 / 3.0);
    angle /= 3.0;
    cmplx_from_polar(radius, angle, first_root);
    cmplx_from_polar(radius, angle + 2.0 * 3.1415926535 / 3.0, second_root);
    cmplx_from_polar(radius, angle + 4.0 * 3.1415926535 / 3.0, third_root);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn returns_three_distinct_real_roots_with_source_candidate_selection() {
        let mut roots = [[0.0; 2]; 3];
        let mut ratio = 0.0;
        solve_cubic_eqn(1.0, -6.0, 11.0, -6.0, &mut roots, &mut ratio);
        let mut real = roots.map(|root| root[0]);
        real.sort_by(f64::total_cmp);
        assert!(
            (real[0] - 1.0).abs() < 1.0e-6
                && (real[1] - 2.0).abs() < 1.0e-6
                && (real[2] - 3.0).abs() < 1.0e-6
        );
        assert!(roots.iter().all(|root| root[1].abs() < 1.0e-6));
    }
}
