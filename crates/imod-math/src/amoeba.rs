//! Nelder-Mead simplex minimizer.
//!
//! Translated from IMOD's `amoeba.c`, which follows the Nelder-Mead algorithm
//! as described by Lagarias et al., SIAM J. Optim. Vol. 9, No. 1, pp. 112-147.

/// Maximum number of variables supported by [`dual_amoeba`].
pub const MAX_DUAL_AMOEBA_VAR: usize = 16;

/// Result returned by [`amoeba`].
#[derive(Debug, Clone)]
pub struct AmoebaResult {
    /// Index of the best vertex in the simplex (row index in `p`).
    pub best_index: usize,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Result returned by [`dual_amoeba`].
#[derive(Debug, Clone)]
pub struct DualAmoebaResult {
    /// Best variable vector after both runs.
    pub best: Vec<f32>,
    /// Total iterations across both runs.
    pub iterations: usize,
}

// ---- internal helpers --------------------------------------------------------

/// Selection-sort `index[0..npts]` so that `y[index[i]]` is non-decreasing.
fn simple_sort(y: &[f32], index: &mut [usize], npts: usize) {
    for i in 0..npts.saturating_sub(1) {
        for j in (i + 1)..npts {
            if y[index[i]] > y[index[j]] {
                index.swap(i, j);
            }
        }
    }
}

/// Accept a new point into the simplex, replacing the worst vertex and
/// maintaining sorted order of `index`.
fn accept_point(
    p: &mut [f32],
    mp: usize,
    ndim: usize,
    index: &mut [usize],
    y: &mut [f32],
    pnew: &[f32],
    ynew: f32,
) {
    // Find insertion position: first index whose y value is greater than ynew
    let insert_pos = (0..ndim)
        .find(|&i| ynew < y[index[i]])
        .unwrap_or(ndim);

    // Overwrite the worst point's slot with the new data
    let ind = index[ndim];
    for idim in 0..ndim {
        p[ind + idim * mp] = pnew[idim];
    }
    y[ind] = ynew;

    // Shift index entries up and insert
    for j in (insert_pos + 1..=ndim).rev() {
        index[j] = index[j - 1];
    }
    index[insert_pos] = ind;
}

// ---- public API --------------------------------------------------------------

/// Initialise the simplex arrays `p`, `y`, and `ptol` before calling [`amoeba`].
///
/// * `p` -- simplex vertices stored column-major with leading dimension `mp`.
///   Must have room for at least `(ndim + 1)` rows and `ndim` columns,
///   i.e. `p.len() >= mp * ndim` with `mp >= ndim + 1`.
/// * `y` -- function values at each vertex; length >= `ndim + 1`.
/// * `mp` -- leading (row) dimension of `p`.
/// * `ndim` -- number of variables.
/// * `delfac` -- initial step size factor (step = `delfac * da[i]`).
/// * `ptol_fac` -- tolerance factor (tolerance = `ptol_fac * da[i]`).
/// * `a` -- initial variable values; length `ndim`.
/// * `da` -- scale factors for each variable; length `ndim`.
/// * `funk` -- objective function `f(x) -> value`.
///
/// Returns the `ptol` vector (length `ndim`).
pub fn amoeba_init<F>(
    p: &mut [f32],
    y: &mut [f32],
    mp: usize,
    ndim: usize,
    delfac: f32,
    ptol_fac: f32,
    a: &[f32],
    da: &[f32],
    funk: &F,
) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut ptol = vec![0.0f32; ndim];
    let mut ptmp = vec![0.0f32; ndim];

    for j in 0..=ndim {
        for i in 0..ndim {
            let mut val = a[i];
            if j > 0 && i == j - 1 {
                val = a[i] + delfac * da[i];
            }
            p[j + i * mp] = val;
            ptmp[i] = val;
            ptol[i] = da[i] * ptol_fac;
        }
        y[j] = funk(&ptmp);
    }
    ptol
}

/// Perform Nelder-Mead simplex minimization.
///
/// * `p`, `y`, `mp`, `ndim` -- simplex data as prepared by [`amoeba_init`].
/// * `ftol` -- fractional tolerance on function value range.
/// * `ptol` -- per-variable absolute tolerance on vertex spread.
/// * `funk` -- objective function `f(x) -> value`.
///
/// Returns an [`AmoebaResult`] with the index of the best vertex and
/// the number of iterations performed.
///
/// # Panics
///
/// Panics if `ndim` exceeds the internal stack limit (20).
pub fn amoeba<F>(
    p: &mut [f32],
    y: &mut [f32],
    mp: usize,
    ndim: usize,
    ftol: f32,
    ptol: &[f32],
    funk: &F,
) -> AmoebaResult
where
    F: Fn(&[f32]) -> f32,
{
    const NMAX: usize = 20;
    assert!(ndim <= NMAX, "ndim ({}) exceeds NMAX ({})", ndim, NMAX);

    // Nelder-Mead coefficients
    let rho: f32 = 1.0;
    let gamma: f32 = 0.5;
    let chi: f32 = 2.0;
    let sigma: f32 = 0.5;

    let iter_max: usize = 1000;
    let npts = ndim + 1;

    // Build sorted index
    let mut index: Vec<usize> = (0..npts).collect();
    simple_sort(y, &mut index, npts);

    let mut pcen = vec![0.0f32; ndim];
    let mut pref = vec![0.0f32; ndim];
    let mut pexp = vec![0.0f32; ndim];

    let mut iter = 0;
    while iter < iter_max {
        let ilow = index[0];
        let ihigh = index[npts - 1];
        let isecond = index[npts - 2];

        // Check per-variable convergence
        let near = (1..npts).all(|ipt| {
            let ind = index[ipt];
            (0..ndim).all(|idim| {
                (p[ind + idim * mp] - p[ilow + idim * mp]).abs() < ptol[idim]
            })
        });
        if near {
            break;
        }

        // Check fractional function-value convergence
        let yhi = y[ihigh];
        let ylo = y[ilow];
        if yhi - ylo <= 0.5 * (yhi.abs() + ylo.abs()) * ftol {
            break;
        }

        // Centroid of all but highest point, and reflection
        for idim in 0..ndim {
            pcen[idim] = 0.0;
            for ipt in 0..(npts - 1) {
                pcen[idim] += p[index[ipt] + idim * mp];
            }
            pcen[idim] /= ndim as f32;
            pref[idim] = (1.0 + rho) * pcen[idim] - rho * p[ihigh + idim * mp];
        }
        let yref = funk(&pref);

        if yref >= y[ilow] && yref < y[isecond] {
            // Accept reflection
            accept_point(p, mp, ndim, &mut index, y, &pref, yref);
        } else if yref < y[ilow] {
            // Expansion
            for idim in 0..ndim {
                pexp[idim] = (1.0 - chi) * pcen[idim] + chi * pref[idim];
            }
            let yexp = funk(&pexp);
            if yexp < yref {
                accept_point(p, mp, ndim, &mut index, y, &pexp, yexp);
            } else {
                accept_point(p, mp, ndim, &mut index, y, &pref, yref);
            }
        } else {
            // Contraction
            let mut shrink = false;

            if yref <= y[ihigh] {
                // Outside contraction
                for idim in 0..ndim {
                    pexp[idim] = (1.0 + rho * gamma) * pcen[idim]
                        - rho * gamma * p[ihigh + idim * mp];
                }
                let yexp = funk(&pexp);
                if yexp <= yref {
                    accept_point(p, mp, ndim, &mut index, y, &pexp, yexp);
                } else {
                    shrink = true;
                }
            } else {
                // Inside contraction
                for idim in 0..ndim {
                    pexp[idim] =
                        (1.0 - gamma) * pcen[idim] + gamma * p[ihigh + idim * mp];
                }
                let yexp = funk(&pexp);
                if yexp < y[ihigh] {
                    accept_point(p, mp, ndim, &mut index, y, &pexp, yexp);
                } else {
                    shrink = true;
                }
            }

            if shrink {
                let mut indsort: usize = 1;
                for ipt in 1..npts {
                    let ind = index[ipt];
                    for idim in 0..ndim {
                        let val = (1.0 - sigma) * p[ilow + idim * mp]
                            + sigma * p[ind + idim * mp];
                        p[ind + idim * mp] = val;
                        pexp[idim] = val;
                    }
                    y[ind] = funk(&pexp);
                    if y[ind] < y[ilow] {
                        indsort = 0;
                    }
                }
                simple_sort(y, &mut index[indsort..], npts - indsort);
            }
        }

        iter += 1;
    }

    AmoebaResult {
        best_index: index[0],
        iterations: iter,
    }
}

/// Run [`amoeba_init`] then [`amoeba`] twice, restarting from the best point
/// found in the first run.
///
/// * `ndim` -- number of variables (must be <= [`MAX_DUAL_AMOEBA_VAR`]).
/// * `delfac` -- initial step size factor.
/// * `ptol_facs` -- `[ptolFac_run1, ptolFac_run2]`.
/// * `ftol_facs` -- `[ftol_run1, ftol_run2]`.
/// * `a` -- initial variable values (length `ndim`); updated in place with
///   the best result.
/// * `da` -- per-variable scale factors.
/// * `funk` -- objective function.
///
/// Returns a [`DualAmoebaResult`].
pub fn dual_amoeba<F>(
    ndim: usize,
    delfac: f32,
    ptol_facs: [f32; 2],
    ftol_facs: [f32; 2],
    a: &mut [f32],
    da: &[f32],
    funk: &F,
) -> DualAmoebaResult
where
    F: Fn(&[f32]) -> f32,
{
    assert!(
        ndim <= MAX_DUAL_AMOEBA_VAR,
        "ndim ({}) exceeds MAX_DUAL_AMOEBA_VAR ({})",
        ndim,
        MAX_DUAL_AMOEBA_VAR
    );

    let mp = MAX_DUAL_AMOEBA_VAR + 1;
    let mut p = vec![0.0f32; mp * ndim];
    let mut y = vec![0.0f32; ndim + 1];

    // First run
    let mut ptol = amoeba_init(&mut p, &mut y, mp, ndim, delfac, ptol_facs[0], a, da, funk);
    let res1 = amoeba(&mut p, &mut y, mp, ndim, ftol_facs[0], &ptol, funk);
    let jmin = res1.best_index;
    for i in 0..ndim {
        a[i] = p[jmin + i * mp];
    }
    let mut total_iter = res1.iterations;

    // Second run
    ptol = amoeba_init(&mut p, &mut y, mp, ndim, delfac, ptol_facs[1], a, da, funk);
    let res2 = amoeba(&mut p, &mut y, mp, ndim, ftol_facs[1], &ptol, funk);
    let jmin = res2.best_index;
    for i in 0..ndim {
        a[i] = p[jmin + i * mp];
    }
    total_iter += res2.iterations;

    DualAmoebaResult {
        best: a.to_vec(),
        iterations: total_iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimise the 2D Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    /// Minimum is at (1, 1) with f = 0.
    #[test]
    fn rosenbrock_2d() {
        let rosenbrock = |v: &[f32]| -> f32 {
            let x = v[0] as f64;
            let y = v[1] as f64;
            ((1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)) as f32
        };

        let ndim = 2;
        let mp = ndim + 1;
        let mut p = vec![0.0f32; mp * ndim];
        let mut y = vec![0.0f32; ndim + 1];
        let a = [0.0f32, 0.0];
        let da = [1.0f32, 1.0];

        let ptol = amoeba_init(&mut p, &mut y, mp, ndim, 0.5, 1e-6, &a, &da, &rosenbrock);
        let res = amoeba(&mut p, &mut y, mp, ndim, 1e-10, &ptol, &rosenbrock);

        let best_x = p[res.best_index];
        let best_y = p[res.best_index + mp];
        assert!(
            (best_x - 1.0).abs() < 0.01 && (best_y - 1.0).abs() < 0.01,
            "Expected near (1,1), got ({}, {})",
            best_x,
            best_y
        );
    }

    /// Minimise a simple quadratic: f(x) = (x - 3)^2. Minimum at x = 3.
    #[test]
    fn simple_quadratic() {
        let f = |v: &[f32]| -> f32 { (v[0] - 3.0) * (v[0] - 3.0) };

        let ndim = 1;
        let mp = ndim + 1;
        let mut p = vec![0.0f32; mp * ndim];
        let mut y = vec![0.0f32; ndim + 1];
        let a = [0.0f32];
        let da = [1.0f32];

        let ptol = amoeba_init(&mut p, &mut y, mp, ndim, 1.0, 1e-7, &a, &da, &f);
        let res = amoeba(&mut p, &mut y, mp, ndim, 1e-10, &ptol, &f);

        let best = p[res.best_index];
        assert!(
            (best - 3.0).abs() < 1e-4,
            "Expected near 3.0, got {}",
            best
        );
    }

    /// Test dual_amoeba on a 2D quadratic.
    #[test]
    fn dual_amoeba_quadratic() {
        let f = |v: &[f32]| -> f32 { (v[0] - 2.0).powi(2) + (v[1] + 1.0).powi(2) };

        let mut a = [0.0f32, 0.0];
        let da = [1.0f32, 1.0];

        let res = dual_amoeba(2, 0.5, [1e-4, 1e-6], [1e-6, 1e-8], &mut a, &da, &f);

        assert!(
            (a[0] - 2.0).abs() < 1e-3 && (a[1] + 1.0).abs() < 1e-3,
            "Expected near (2, -1), got ({}, {}), iter={}",
            a[0],
            a[1],
            res.iterations
        );
    }
}
