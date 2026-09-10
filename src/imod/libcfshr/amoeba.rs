//! Translation of `IMOD/libcfshr/amoeba.c`.
#![allow(dead_code)]

const NMAX: usize = 20;
const MAX_DUAL_AMOEBA_VAR: usize = 20;

/// Matches static `simpleSort` (`IMOD/libcfshr/amoeba.c:25`).
fn simple_sort(values: &[f32], index: &mut [usize], points: usize) {
    for point in 0..points.saturating_sub(1) {
        for next in point + 1..points {
            if values[index[point]] > values[index[next]] {
                index.swap(point, next);
            }
        }
    }
}

/// Matches static `acceptPoint` (`IMOD/libcfshr/amoeba.c:39`).
fn accept_point(
    points: &mut [f32],
    fastest_dimension: usize,
    dimensions: usize,
    index: &mut [usize],
    values: &mut [f32],
    new_point: &[f32],
    new_value: f32,
) {
    let mut point = 0;
    while point < dimensions && new_value >= values[index[point]] {
        point += 1;
    }
    let displaced = index[dimensions];
    for dimension in 0..dimensions {
        points[displaced + dimension * fastest_dimension] = new_point[dimension];
    }
    values[displaced] = new_value;
    for current in (point + 1..=dimensions).rev() {
        index[current] = index[current - 1];
    }
    index[point] = displaced;
}

/// Matches `amoeba` (`IMOD/libcfshr/amoeba.c:75`).
pub fn amoeba<Function: FnMut(&[f32]) -> f32>(
    points: &mut [f32],
    values: &mut [f32],
    fastest_dimension: usize,
    dimensions: usize,
    function_tolerance: f32,
    function: &mut Function,
    iterations: &mut i32,
    point_tolerance: &[f32],
    lowest_index: &mut usize,
) {
    assert!(
        dimensions <= NMAX
            && fastest_dimension >= dimensions + 1
            && points.len() >= fastest_dimension * dimensions
            && values.len() >= dimensions + 1
            && point_tolerance.len() >= dimensions
    );
    let points_count = dimensions + 1;
    let mut index = (0..points_count).collect::<Vec<_>>();
    simple_sort(values, &mut index, points_count);
    let mut center = vec![0.; dimensions];
    let mut reflection = vec![0.; dimensions];
    let mut expansion = vec![0.; dimensions];
    for iteration in 0..1000 {
        let low = index[0];
        let high = index[points_count - 1];
        let second = index[points_count - 2];
        let near = index[1..].iter().all(|point| {
            (0..dimensions).all(|dimension| {
                (points[*point + dimension * fastest_dimension]
                    - points[low + dimension * fastest_dimension])
                    .abs()
                    < point_tolerance[dimension]
            })
        });
        if near
            || values[high] - values[low]
                <= 0.5 * (values[high].abs() + values[low].abs()) * function_tolerance
        {
            *lowest_index = low;
            *iterations = iteration as i32;
            return;
        }
        for dimension in 0..dimensions {
            center[dimension] = index[..points_count - 1]
                .iter()
                .map(|point| points[*point + dimension * fastest_dimension])
                .sum::<f32>()
                / dimensions as f32;
            reflection[dimension] =
                2. * center[dimension] - points[high + dimension * fastest_dimension];
        }
        let reflected_value = function(&reflection);
        if reflected_value >= values[low] && reflected_value < values[second] {
            accept_point(
                points,
                fastest_dimension,
                dimensions,
                &mut index,
                values,
                &reflection,
                reflected_value,
            );
        } else if reflected_value < values[low] {
            for dimension in 0..dimensions {
                expansion[dimension] = -center[dimension] + 2. * reflection[dimension];
            }
            let expansion_value = function(&expansion);
            if expansion_value < reflected_value {
                accept_point(
                    points,
                    fastest_dimension,
                    dimensions,
                    &mut index,
                    values,
                    &expansion,
                    expansion_value,
                );
            } else {
                accept_point(
                    points,
                    fastest_dimension,
                    dimensions,
                    &mut index,
                    values,
                    &reflection,
                    reflected_value,
                );
            }
        } else {
            let mut shrink = false;
            if reflected_value <= values[high] {
                for dimension in 0..dimensions {
                    expansion[dimension] = 1.5 * center[dimension]
                        - 0.5 * points[high + dimension * fastest_dimension];
                }
                let contraction = function(&expansion);
                if contraction <= reflected_value {
                    accept_point(
                        points,
                        fastest_dimension,
                        dimensions,
                        &mut index,
                        values,
                        &expansion,
                        contraction,
                    );
                } else {
                    shrink = true;
                }
            } else {
                for dimension in 0..dimensions {
                    expansion[dimension] = 0.5 * center[dimension]
                        + 0.5 * points[high + dimension * fastest_dimension];
                }
                let contraction = function(&expansion);
                if contraction < values[high] {
                    accept_point(
                        points,
                        fastest_dimension,
                        dimensions,
                        &mut index,
                        values,
                        &expansion,
                        contraction,
                    );
                } else {
                    shrink = true;
                }
            }
            if shrink {
                let mut sort_from = 1;
                for position in 1..points_count {
                    let point = index[position];
                    for dimension in 0..dimensions {
                        expansion[dimension] = 0.5 * points[low + dimension * fastest_dimension]
                            + 0.5 * points[point + dimension * fastest_dimension];
                        points[point + dimension * fastest_dimension] = expansion[dimension];
                    }
                    values[point] = function(&expansion);
                    if values[point] < values[low] {
                        sort_from = 0;
                    }
                }
                simple_sort(values, &mut index[sort_from..], points_count - sort_from);
            }
        }
    }
    *lowest_index = index[0];
    *iterations = 1000;
}

/// Matches `amoebaInit` (`IMOD/libcfshr/amoeba.c:192`).
pub fn amoeba_init<Function: FnMut(&[f32]) -> f32>(
    points: &mut [f32],
    values: &mut [f32],
    fastest_dimension: usize,
    dimensions: usize,
    delta_factor: f32,
    tolerance_factor: f32,
    initial: &[f32],
    delta: &[f32],
    function: &mut Function,
    point_tolerance: &mut [f32],
) {
    for simplex in 0..=dimensions {
        let mut temporary = vec![0.; dimensions];
        for dimension in 0..dimensions {
            points[simplex + dimension * fastest_dimension] = initial[dimension]
                + if simplex != 0 && dimension == simplex - 1 {
                    delta_factor * delta[dimension]
                } else {
                    0.
                };
            temporary[dimension] = points[simplex + dimension * fastest_dimension];
            point_tolerance[dimension] = delta[dimension] * tolerance_factor;
        }
        values[simplex] = function(&temporary);
    }
}

/// Matches `dualAmoeba` (`IMOD/libcfshr/amoeba.c:221`).
pub fn dual_amoeba<Function: FnMut(&[f32]) -> f32>(
    values: &mut [f32],
    dimensions: usize,
    delta_factor: f32,
    tolerance_factors: &[f32; 2],
    function_tolerances: &[f32; 2],
    initial: &mut [f32],
    delta: &[f32],
    function: &mut Function,
    iterations: &mut i32,
) {
    assert!(dimensions <= MAX_DUAL_AMOEBA_VAR);
    let stride = MAX_DUAL_AMOEBA_VAR + 1;
    let mut points = vec![0.; stride * stride];
    let mut tolerance = vec![0.; MAX_DUAL_AMOEBA_VAR];
    let mut low = 0;
    amoeba_init(
        &mut points,
        values,
        stride,
        dimensions,
        delta_factor,
        tolerance_factors[0],
        initial,
        delta,
        function,
        &mut tolerance,
    );
    amoeba(
        &mut points,
        values,
        stride,
        dimensions,
        function_tolerances[0],
        function,
        iterations,
        &tolerance,
        &mut low,
    );
    for dimension in 0..dimensions {
        initial[dimension] = points[low + dimension * stride];
    }
    let first = *iterations;
    amoeba_init(
        &mut points,
        values,
        stride,
        dimensions,
        delta_factor,
        tolerance_factors[1],
        initial,
        delta,
        function,
        &mut tolerance,
    );
    amoeba(
        &mut points,
        values,
        stride,
        dimensions,
        function_tolerances[1],
        function,
        iterations,
        &tolerance,
        &mut low,
    );
    for dimension in 0..dimensions {
        initial[dimension] = points[low + dimension * stride];
    }
    *iterations += first;
}

/// Matches Fortran wrapper `amoebafwrap` (`IMOD/libcfshr/amoeba.c:251`).
pub fn amoebafwrap<Function: FnMut(&[f32]) -> f32>(
    points: &mut [f32],
    values: &mut [f32],
    fastest_dimension: usize,
    dimensions: usize,
    function_tolerance: f32,
    function: &mut Function,
    iterations: &mut i32,
    point_tolerance: &[f32],
    lowest_index_one_based: &mut usize,
) {
    let mut lowest = 0;
    amoeba(
        points,
        values,
        fastest_dimension,
        dimensions,
        function_tolerance,
        function,
        iterations,
        point_tolerance,
        &mut lowest,
    );
    *lowest_index_one_based = lowest + 1;
}
/// Matches Fortran wrapper `amoebainitfwrap` (`IMOD/libcfshr/amoeba.c:261`).
pub fn amoebainitfwrap<Function: FnMut(&[f32]) -> f32>(
    points: &mut [f32],
    values: &mut [f32],
    fastest_dimension: usize,
    dimensions: usize,
    delta_factor: f32,
    tolerance_factor: f32,
    initial: &[f32],
    delta: &[f32],
    function: &mut Function,
    point_tolerance: &mut [f32],
) {
    amoeba_init(
        points,
        values,
        fastest_dimension,
        dimensions,
        delta_factor,
        tolerance_factor,
        initial,
        delta,
        function,
        point_tolerance,
    )
}
/// Matches Fortran wrapper `dualamoe ba` (`IMOD/libcfshr/amoeba.c:269`).
pub fn dualamoeba<Function: FnMut(&[f32]) -> f32>(
    values: &mut [f32],
    dimensions: usize,
    delta_factor: f32,
    tolerance_factors: &[f32; 2],
    function_tolerances: &[f32; 2],
    initial: &mut [f32],
    delta: &[f32],
    function: &mut Function,
    iterations: &mut i32,
) {
    dual_amoeba(
        values,
        dimensions,
        delta_factor,
        tolerance_factors,
        function_tolerances,
        initial,
        delta,
        function,
        iterations,
    )
}
