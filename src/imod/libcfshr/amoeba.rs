//! Translation of `IMOD/libcfshr/amoeba.c`.

const NMAX: usize = 20;
/// `IMOD/include/cfsemshare.h:35` — **16**, not `NMAX`.  It sizes
/// `dualAmoeba`'s `pp`/`ptol` and, through `stride`, the row pitch it hands to
/// `amoebaInit`/`amoeba`, so the two constants are not interchangeable.
const MAX_DUAL_AMOEBA_VAR: usize = 16;

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
    // `amoeba.c:60-62`: `if (ynew < y[index[ipt]]) break;` -- the loop goes on
    // while that test is *false*, which includes a NaN on either side.  Writing
    // it as `ynew >= y` stops at a NaN instead.
    while point < dimensions && !(new_value < values[index[point]]) {
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
/// `amoeba.c:103`: `float rho = 1.0, gamma = 0.5, chi = 2.0, sigma = 0.5;`
/// — the reflection, contraction, expansion and shrinkage coefficients.
///
/// They are `float`, and that is load-bearing rather than incidental. Every
/// combination formula below writes them as `(1. + rho)`, `(1. - chi)`,
/// `(1. + rho * gamma)` and so on, where the `1.` is a **double** literal: so
/// the coefficient is a double, the term it multiplies evaluates in double,
/// and the term written as a bare `rho * p[...]` is a *single-precision*
/// product that only widens for the addition. Folding them to `f32`
/// constants — which the translation had done, as `2. * center - points`,
/// `1.5 * center - 0.5 * points` and the rest — collapses that to a
/// single-precision expression. This is CLAUDE.md's "a C `float` variable
/// inside a double expression rounds mid-expression", five times over.
const RHO: f32 = 1.0;
const GAMMA: f32 = 0.5;
const CHI: f32 = 2.0;
const SIGMA: f32 = 0.5;

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
    // `amoeba.c:104-105` declares `int index[NMAX]; float pcen[NMAX],
    // pref[NMAX], pexp[NMAX];` — fixed-size automatic arrays, not allocations.
    // `index` is written through `npts - 1 == dimensions`, so it needs one more
    // slot than the source's own declaration provides.
    let mut index = [0usize; NMAX + 1];
    // `amoeba.c:111-112`.
    for (point, slot) in index[..points_count].iter_mut().enumerate() {
        *slot = point;
    }
    simple_sort(values, &mut index[..points_count], points_count);
    let mut center = [0.0f32; NMAX];
    let mut reflection = [0.0f32; NMAX];
    let mut expansion = [0.0f32; NMAX];
    // `amoeba.c:216`: after the loop runs out, `*iloP = ilow` -- the lowest
    // point as of the start of the last iteration, not the current `index[0]`.
    let mut last_low = index[0];
    for iteration in 0..1000 {
        let low = index[0];
        last_low = low;
        let high = index[points_count - 1];
        let second = index[points_count - 2];
        // `amoeba.c:124`: `for (ipt = 1; ipt < npts && near; ipt++)` — the
        // fixed array is longer than `npts`, so the bound must be explicit.
        // `amoeba.c:127`: `near = 0` only when `fabs(...) >= ptol[idim]`, so a
        // NaN difference leaves the point counted as near.
        let near = index[1..points_count].iter().all(|point| {
            (0..dimensions).all(|dimension| {
                !((points[*point + dimension * fastest_dimension]
                    - points[low + dimension * fastest_dimension])
                    .abs()
                    >= point_tolerance[dimension])
            })
        });
        // `amoeba.c:136-137`: `y[ihigh] - y[ilow] <= 0.5 * (fabs((double)y[ihigh])
        // + fabs((double)y[ilow])) * ftol` -- the right side is a double
        // expression; only the difference on the left is a float.
        if near
            || ((values[high] - values[low]) as f64)
                <= 0.5
                    * ((values[high] as f64).abs() + (values[low] as f64).abs())
                    * function_tolerance as f64
        {
            *lowest_index = low;
            *iterations = iteration as i32;
            return;
        }
        for dimension in 0..dimensions {
            // `amoeba.c:143-145`: a float accumulator starting at `0.`, not
            // `Iterator::sum`, which starts at -0.0.
            center[dimension] = 0.;
            for point in &index[..points_count - 1] {
                center[dimension] += points[*point + dimension * fastest_dimension];
            }
            center[dimension] /= dimensions as f32;
            // `amoeba.c:147`: `pref[idim] = (1. + rho) * pcen[idim] - rho *
            // p[ihigh + idim*mp];`  `rho` is a `float` (`:103`) and the `1.`
            // is a double, so `(1. + rho)` is a *double* coefficient and the
            // whole left term evaluates in double, while `rho * p[...]` is a
            // single-precision product that only then widens.  The result
            // rounds to `float` once, at the assignment.
            reflection[dimension] = ((1. + RHO as f64) * center[dimension] as f64
                - (RHO * points[high + dimension * fastest_dimension]) as f64)
                as f32;
        }
        let reflected_value = function(&reflection[..dimensions]);
        if reflected_value >= values[low] && reflected_value < values[second] {
            accept_point(
                points,
                fastest_dimension,
                dimensions,
                &mut index,
                values,
                &reflection[..dimensions],
                reflected_value,
            );
        } else if reflected_value < values[low] {
            for dimension in 0..dimensions {
                // `amoeba.c:158`: `(1. - chi) * pcen[idim] + chi * pref[idim]`
                // — `chi` is a `float`, so the coefficient is a double and the
                // `chi * pref` product is single precision.
                expansion[dimension] = ((1. - CHI as f64) * center[dimension] as f64
                    + (CHI * reflection[dimension]) as f64)
                    as f32;
            }
            let expansion_value = function(&expansion[..dimensions]);
            if expansion_value < reflected_value {
                accept_point(
                    points,
                    fastest_dimension,
                    dimensions,
                    &mut index,
                    values,
                    &expansion[..dimensions],
                    expansion_value,
                );
            } else {
                accept_point(
                    points,
                    fastest_dimension,
                    dimensions,
                    &mut index,
                    values,
                    &reflection[..dimensions],
                    reflected_value,
                );
            }
        } else {
            let mut shrink = false;
            if reflected_value <= values[high] {
                for dimension in 0..dimensions {
                    // `amoeba.c:174-175`: `(1. + rho * gamma) * pcen[idim] -
                    // rho * gamma * p[ihigh + idim*mp]`.  `rho * gamma` is a
                    // single-precision product; `1. +` makes the coefficient a
                    // double; the subtracted term stays single until it widens.
                    expansion[dimension] = ((1. + (RHO * GAMMA) as f64) * center[dimension] as f64
                        - (RHO * GAMMA * points[high + dimension * fastest_dimension]) as f64)
                        as f32;
                }
                let contraction = function(&expansion[..dimensions]);
                if contraction <= reflected_value {
                    accept_point(
                        points,
                        fastest_dimension,
                        dimensions,
                        &mut index,
                        values,
                        &expansion[..dimensions],
                        contraction,
                    );
                } else {
                    shrink = true;
                }
            } else {
                for dimension in 0..dimensions {
                    // `amoeba.c:185`: `(1. - gamma) * pcen[idim] + gamma *
                    // p[ihigh + idim*mp]`.
                    expansion[dimension] = ((1. - GAMMA as f64) * center[dimension] as f64
                        + (GAMMA * points[high + dimension * fastest_dimension]) as f64)
                        as f32;
                }
                let contraction = function(&expansion[..dimensions]);
                if contraction < values[high] {
                    accept_point(
                        points,
                        fastest_dimension,
                        dimensions,
                        &mut index,
                        values,
                        &expansion[..dimensions],
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
                        // `amoeba.c:199-200`: `(1. - sigma) * p[ilow + idim*mp]
                        // + sigma * p[ind + idim*mp]`.
                        expansion[dimension] = ((1. - SIGMA as f64)
                            * points[low + dimension * fastest_dimension] as f64
                            + (SIGMA * points[point + dimension * fastest_dimension]) as f64)
                            as f32;
                        points[point + dimension * fastest_dimension] = expansion[dimension];
                    }
                    values[point] = function(&expansion[..dimensions]);
                    if values[point] < values[low] {
                        sort_from = 0;
                    }
                }
                simple_sort(values, &mut index[sort_from..], points_count - sort_from);
            }
        }
    }
    *lowest_index = last_low;
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
    // `amoeba.c:231`: `float ptmp[NMAX];` — one automatic array declared
    // outside the `j` loop, not an allocation per simplex point.
    let mut temporary = [0.0f32; NMAX];
    for simplex in 0..=dimensions {
        for dimension in 0..dimensions {
            // `amoeba.c:235-237`: the plain copy, then the displaced value
            // for the one perturbed dimension.  Adding `0.` instead would
            // turn a `-0.` start value into `+0.`.
            points[simplex + dimension * fastest_dimension] = initial[dimension];
            if simplex != 0 && dimension == simplex - 1 {
                points[simplex + dimension * fastest_dimension] =
                    initial[dimension] + delta_factor * delta[dimension];
            }
            temporary[dimension] = points[simplex + dimension * fastest_dimension];
            point_tolerance[dimension] = delta[dimension] * tolerance_factor;
        }
        values[simplex] = function(&temporary[..dimensions]);
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
    // `amoeba.c:236`: `float pp[MAX_DUAL_AMOEBA_VAR + 1][MAX_DUAL_AMOEBA_VAR
    // + 1], ptol[MAX_DUAL_AMOEBA_VAR];` — automatic arrays, not allocations.
    let mut points = [0.0f32; (MAX_DUAL_AMOEBA_VAR + 1) * (MAX_DUAL_AMOEBA_VAR + 1)];
    let mut tolerance = [0.0f32; MAX_DUAL_AMOEBA_VAR];
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
