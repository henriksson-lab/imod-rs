// Translated from IMOD libcfshr/regression.c and libcfshr/gaussj.c
// Original author: David Mastronarde, University of Colorado
//
// Multiple linear regression, robust regression with Tukey bisquare weighting,
// polynomial fitting, and statistical matrix computation.

use crate::gaussj::{gaussj as gaussj_solve, GaussjError};
use crate::{madn, median, sort_floats};

/// Error type for regression operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    /// Invalid weight column specification.
    InvalidWeightColumn,
    /// Invalid specification of initial zero-weight rows.
    InvalidZeroWeightSpec,
    /// Singular matrix encountered during Gauss-Jordan elimination.
    SingularMatrix,
    /// Matrix dimension exceeds maximum supported size (2000).
    MatrixTooLarge,
    /// Zero polynomial order is not allowed.
    ZeroOrder,
    /// Robust regression did not converge within the allowed iterations.
    NotConverged,
    /// Too few data points.
    TooFewData,
}

impl std::fmt::Display for RegressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionError::InvalidWeightColumn => write!(f, "Invalid weight column"),
            RegressionError::InvalidZeroWeightSpec => {
                write!(f, "Invalid zero-weight row specification")
            }
            RegressionError::SingularMatrix => write!(f, "Singular matrix"),
            RegressionError::MatrixTooLarge => {
                write!(f, "Matrix dimension exceeds maximum (2000)")
            }
            RegressionError::ZeroOrder => write!(f, "Zero polynomial order"),
            RegressionError::NotConverged => write!(f, "Robust regression did not converge"),
            RegressionError::TooFewData => write!(f, "Too few data points"),
        }
    }
}

impl std::error::Error for RegressionError {}

impl From<GaussjError> for RegressionError {
    fn from(e: GaussjError) -> Self {
        match e {
            GaussjError::DimensionTooLarge => RegressionError::MatrixTooLarge,
            GaussjError::SingularMatrix => RegressionError::SingularMatrix,
        }
    }
}

// ---------------------------------------------------------------------------
// Data matrix access helper
// ---------------------------------------------------------------------------

/// Access helper for a data matrix that may be stored in either row-major or
/// column-major order.
struct DataMatrix<'a> {
    data: &'a [f32],
    row_stride: usize,
    col_stride: usize,
}

impl<'a> DataMatrix<'a> {
    fn new(data: &'a [f32], x_size: usize, col_fast: bool) -> Self {
        let col_stride = if col_fast { 1 } else { x_size };
        let row_stride = if col_fast { x_size } else { 1 };
        Self {
            data,
            row_stride,
            col_stride,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.row_stride + col * self.col_stride]
    }
}

// ---------------------------------------------------------------------------
// statMatrices
// ---------------------------------------------------------------------------

/// Output of [`stat_matrices`]: basic statistical values and matrices.
#[derive(Debug, Clone)]
pub struct StatMatricesOutput {
    /// Weighted (or unweighted) sums of each column.
    pub sx: Vec<f32>,
    /// Raw sums of squares and cross-products (m x m, row-major with leading dim `msize`).
    pub ss: Vec<f32>,
    /// Sums of deviation squares and cross-products (m x m, leading dim `msize`).
    pub ssd: Vec<f32>,
    /// Dispersion (covariance) matrix (m x m, leading dim `msize`). Empty if `ifdisp == 0`.
    pub d: Vec<f32>,
    /// Correlation coefficient matrix (m x m, leading dim `msize`). Empty if `ifdisp == 0`.
    pub r: Vec<f32>,
    /// Means of each column.
    pub xm: Vec<f32>,
    /// Standard deviations of each column.
    pub sd: Vec<f32>,
}

/// Computes basic statistical values and matrices from a data matrix representing
/// a series of measurements of multiple variables.
///
/// # Parameters
///
/// - `x`: Input data matrix, stored flat.
/// - `x_size`: Size of the fastest-progressing dimension of `x`.
/// - `col_fast`: If `true`, the column dimension is the fastest progressing one.
/// - `m`: Number of columns of data (number of parameters).
/// - `msize`: Leading dimension for the output square matrices.
/// - `ndata`: Number of rows (measurements).
/// - `ifdisp`: Controls computation:
///   - `0`: skip dispersion and correlation matrices.
///   - `> 0`: compute all matrices without weighting.
///   - `< 0`: treat column `m` as weighting values.
pub fn stat_matrices(
    x: &[f32],
    x_size: usize,
    col_fast: bool,
    m: usize,
    msize: usize,
    ndata: usize,
    ifdisp: i32,
) -> StatMatricesOutput {
    let mat = DataMatrix::new(x, x_size, col_fast);
    let fndata = ndata as f32;

    let mut sx = vec![0.0f32; m];
    let mat_size = msize * m;
    let mut ssd = vec![0.0f32; mat_size];
    let mut r = vec![0.0f32; mat_size];
    let mut xm = vec![0.0f32; m];

    // Compute sums and means
    if ifdisp >= 0 {
        for i in 0..m {
            for k in 0..ndata {
                sx[i] += mat.get(k, i);
            }
            xm[i] = sx[i] / fndata;
        }
    } else {
        let mut wsum: f32 = 0.0;
        for k in 0..ndata {
            wsum += mat.get(k, m);
        }
        for i in 0..m {
            for k in 0..ndata {
                sx[i] += mat.get(k, i) * mat.get(k, m);
            }
            xm[i] = sx[i] / wsum;
        }
    }

    // Sums of deviation squares and cross-products
    for k in 0..ndata {
        let weight = if ifdisp < 0 { mat.get(k, m) } else { 1.0 };
        for i in 0..m {
            for j in 0..m {
                ssd[i * msize + j] +=
                    (mat.get(k, i) - xm[i]) * (mat.get(k, j) - xm[j]) * weight;
            }
        }
    }

    // Standard deviations and raw sums of squares
    let mut sd = vec![0.0f32; m];
    let mut ss = vec![0.0f32; mat_size];

    for i in 0..m {
        sd[i] = (ssd[i * msize + i] / (fndata - 1.0)).sqrt();
        for j in 0..m {
            ss[i * msize + j] = ssd[i * msize + j] + sx[i] * sx[j] / fndata;
            ss[j * msize + i] = ss[i * msize + j];
            ssd[j * msize + i] = ssd[i * msize + j];
        }
    }

    let mut d = Vec::new();
    if ifdisp != 0 {
        d = vec![0.0f32; mat_size];
        for i in 0..m {
            for j in 0..m {
                d[i * msize + j] = ssd[i * msize + j] / (fndata - 1.0);
                d[j * msize + i] = d[i * msize + j];
                let den = sd[i] * sd[j];
                r[i * msize + j] = if den > 1.0e-30 {
                    d[i * msize + j] / (sd[i] * sd[j])
                } else {
                    1.0
                };
                r[j * msize + i] = r[i * msize + j];
            }
        }
    }

    StatMatricesOutput {
        sx,
        ss,
        ssd,
        d,
        r,
        xm,
        sd,
    }
}

// ---------------------------------------------------------------------------
// multRegress
// ---------------------------------------------------------------------------

/// Result of [`mult_regress`].
#[derive(Debug, Clone)]
pub struct MultRegressResult {
    /// Coefficient matrix: `sol[i + sol_size * j]` is the coefficient of input
    /// variable `i` for output variable `j`.
    pub sol: Vec<f32>,
    /// Constant terms for each output variable (one per output column).
    /// `None` when fitting with no constant term.
    pub cons: Option<Vec<f32>>,
    /// Means of all input and output data columns.
    pub x_mean: Vec<f32>,
    /// Standard deviations of all input and output data columns.
    pub x_sd: Vec<f32>,
}

/// Computes a multiple linear regression (least-squares fit) for the
/// relationships between one or more dependent (output) variables and a set
/// of independent (input) variables.
///
/// # Parameters
///
/// - `x`: Flat data matrix.
/// - `x_size`: Leading dimension of `x`.
/// - `col_fast`: Whether the column dimension is the fastest progressing one.
/// - `num_inp_col`: Number of input (independent) variable columns. May be 0.
/// - `num_data`: Number of data rows (measurements).
/// - `num_out_col`: Number of output (dependent) variable columns.
/// - `wgt_col`: Column index holding weighting factors, or `None` for
///   unweighted regression.
/// - `sol_size`: Leading dimension of the solution matrix (must be >= `num_inp_col`).
/// - `with_constant`: Whether to include a constant (intercept) term.
///
/// # Returns
///
/// A [`MultRegressResult`] containing the fitted coefficients, optional
/// constant terms, means, and standard deviations.
pub fn mult_regress(
    x: &[f32],
    x_size: usize,
    col_fast: bool,
    num_inp_col: usize,
    num_data: usize,
    num_out_col: usize,
    wgt_col: Option<usize>,
    sol_size: usize,
    with_constant: bool,
) -> Result<MultRegressResult, RegressionError> {
    let mat = DataMatrix::new(x, x_size, col_fast);
    let mp = num_inp_col + num_out_col;
    let fndata = num_data as f32;

    // Validate weight column
    if let Some(wc) = wgt_col {
        if wc < mp || (!col_fast && wc >= x_size) {
            return Err(RegressionError::InvalidWeightColumn);
        }
    }

    let mut x_mean = vec![0.0f32; mp];
    let mut x_sd = vec![0.0f32; mp];

    // Compute means
    if wgt_col.is_none() {
        for i in 0..mp {
            let mut dsum: f64 = 0.0;
            for k in 0..num_data {
                dsum += mat.get(k, i) as f64;
            }
            x_mean[i] = (dsum / fndata as f64) as f32;
        }
    } else {
        let wc = wgt_col.unwrap();
        let mut wsum: f64 = 0.0;
        for k in 0..num_data {
            wsum += mat.get(k, wc) as f64;
        }
        for i in 0..mp {
            let mut dsum: f64 = 0.0;
            for k in 0..num_data {
                dsum += mat.get(k, i) as f64 * mat.get(k, wc) as f64;
            }
            x_mean[i] = (dsum / wsum) as f32;
        }
    }

    // Sums of squares and cross-products of deviations
    let mut work = vec![0.0f32; mp * mp];
    for i in 0..mp {
        for j in i..mp {
            if i >= num_inp_col && i != j {
                continue;
            }
            let mut dsum: f64 = 0.0;
            if with_constant {
                if wgt_col.is_none() {
                    for k in 0..num_data {
                        dsum += (mat.get(k, i) - x_mean[i]) as f64
                            * (mat.get(k, j) - x_mean[j]) as f64;
                    }
                } else {
                    let wc = wgt_col.unwrap();
                    for k in 0..num_data {
                        dsum += (mat.get(k, i) - x_mean[i]) as f64
                            * (mat.get(k, j) - x_mean[j]) as f64
                            * mat.get(k, wc) as f64;
                    }
                }
            } else if wgt_col.is_none() {
                for k in 0..num_data {
                    dsum += mat.get(k, i) as f64 * mat.get(k, j) as f64;
                }
            } else {
                let wc = wgt_col.unwrap();
                for k in 0..num_data {
                    dsum += mat.get(k, i) as f64 * mat.get(k, j) as f64 * mat.get(k, wc) as f64;
                }
            }
            work[j * mp + i] = dsum as f32;
        }
    }

    // Standard deviations
    for i in 0..mp {
        x_sd[i] = (work[i * mp + i] / (fndata - 1.0)).sqrt();
    }

    // If num_inp_col == 0, return the means as constant values
    if num_inp_col == 0 {
        let cons = if with_constant {
            Some(x_mean.clone())
        } else {
            None
        };
        return Ok(MultRegressResult {
            sol: Vec::new(),
            cons,
            x_mean,
            x_sd,
        });
    }

    // Scale by (n-1) and SDs to get correlation matrix
    for i in 0..num_inp_col {
        for j in i..mp {
            let den = x_sd[i] * x_sd[j];
            if den < 1.0e-30 {
                work[j * mp + i] = 1.0;
            } else {
                work[j * mp + i] /= den * (fndata - 1.0);
            }
            if j < num_inp_col {
                work[i * mp + j] = work[j * mp + i];
            }
        }
    }

    // Transpose final columns into sol for gaussj
    let mut sol = vec![0.0f32; num_inp_col * num_out_col];
    for j in 0..num_out_col {
        for i in 0..num_inp_col {
            sol[j + i * num_out_col] = work[(j + num_inp_col) * mp + i];
        }
    }

    gaussj_solve(&mut work, num_inp_col, mp, &mut sol, num_out_col, num_out_col)?;

    // Scale coefficients and transpose back; get constant terms
    let work_copy = sol.clone();
    let mut sol_out = vec![0.0f32; sol_size * num_out_col];
    let mut cons = if with_constant {
        Some(vec![0.0f32; num_out_col])
    } else {
        None
    };

    for j in 0..num_out_col {
        if let Some(ref mut c) = cons {
            c[j] = x_mean[num_inp_col + j];
        }
        for i in 0..num_inp_col {
            if x_sd[i] < 1.0e-30 {
                sol_out[i + sol_size * j] = 0.0;
            } else {
                sol_out[i + sol_size * j] =
                    work_copy[j + i * num_out_col] * x_sd[num_inp_col + j] / x_sd[i];
            }
            if let Some(ref mut c) = cons {
                c[j] -= sol_out[i + sol_size * j] * x_mean[i];
            }
        }
    }

    Ok(MultRegressResult {
        sol: sol_out,
        cons,
        x_mean,
        x_sd,
    })
}

// ---------------------------------------------------------------------------
// robustRegress
// ---------------------------------------------------------------------------

/// Parameters controlling the robust regression iteration.
#[derive(Debug, Clone)]
pub struct RobustParams {
    /// Factor dividing the standardized residual in the Tukey bisquare weight.
    /// 4.685 gives 95% efficiency for normally distributed errors.
    pub kfactor: f32,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Maximum number of points allowed to have zero weight.
    pub max_zero_wgt: usize,
    /// Maximum change in weights between iterations for convergence.
    /// Should not be less than 0.01.
    pub max_change: f32,
    /// Maximum change in weights when oscillating. Iterations stop when
    /// the biggest single-iteration change is below this AND the biggest
    /// two-iteration change is below `max_change`.
    pub max_oscill: f32,
}

impl Default for RobustParams {
    fn default() -> Self {
        Self {
            kfactor: 4.685,
            max_iter: 30,
            max_zero_wgt: 0,
            max_change: 0.02,
            max_oscill: 0.05,
        }
    }
}

/// Result of [`robust_regress`].
#[derive(Debug, Clone)]
pub struct RobustRegressResult {
    /// Coefficient matrix (same layout as [`MultRegressResult::sol`]).
    pub sol: Vec<f32>,
    /// Constant terms (one per output column).
    /// `None` when fitting with no constant term.
    pub cons: Option<Vec<f32>>,
    /// Means of all data columns.
    pub x_mean: Vec<f32>,
    /// Standard deviations of all data columns.
    pub x_sd: Vec<f32>,
    /// Number of iterations performed.
    pub num_iter: usize,
    /// Unweighted root-mean-square residual error.
    pub rms_err: f32,
    /// Weighted root-mean-square residual error.
    pub wgt_rms_err: f32,
}

/// Computes a robust least-squares fit by iteratively computing a Tukey
/// bisquare weight from the residual for each data point and then doing
/// a weighted regression.
///
/// The weight for each point is derived from the median and normalised median
/// absolute deviation (MADN) of the residuals, using the Tukey bisquare
/// equation with a configurable k-factor.
///
/// # Parameters
///
/// - `x`: Flat data matrix. Must have at least `num_inp_col + num_out_col + 2`
///   columns because the final two columns are used to store current and
///   previous iteration weights. **Modified in place.**
/// - `x_size`: Leading dimension of `x`.
/// - `col_fast`: Whether the column dimension is fastest progressing.
/// - `num_inp_col`: Number of input variable columns (may be 0).
/// - `num_data`: Number of data rows.
/// - `num_out_col`: Number of output variable columns.
/// - `sol_size`: Leading dimension of the solution matrix.
/// - `with_constant`: Whether to include a constant (intercept) term.
/// - `params`: Robust iteration control parameters.
/// - `initial_zero_rows`: Optional list of row indices to give zero initial weight.
///
/// # Returns
///
/// A [`RobustRegressResult`] on success.
pub fn robust_regress(
    x: &mut [f32],
    x_size: usize,
    col_fast: bool,
    num_inp_col: usize,
    num_data: usize,
    num_out_col: usize,
    sol_size: usize,
    with_constant: bool,
    params: &RobustParams,
    initial_zero_rows: Option<&[usize]>,
) -> Result<RobustRegressResult, RegressionError> {
    let wgt_col = num_inp_col + num_out_col;
    let prev_col = wgt_col + 1;
    let col_stride = if col_fast { 1 } else { x_size };
    let row_stride = if col_fast { x_size } else { 1 };

    let kfactor = params.kfactor.abs();
    let max_iter = params.max_iter;
    let max_zero_wgt = params.max_zero_wgt;
    let max_change = params.max_change;
    let max_oscill = params.max_oscill;
    let min_non_zero_wgt: f32 = 0.02;

    if col_fast && prev_col >= x_size {
        return Err(RegressionError::InvalidWeightColumn);
    }

    // Macro-like closure for indexed access
    let idx = |r: usize, c: usize| -> usize { r * row_stride + c * col_stride };

    // Initialize weights to 1
    for j in 0..num_data {
        x[idx(j, wgt_col)] = 1.0;
        x[idx(j, prev_col)] = 1.0;
    }

    // Set specified rows to zero weight
    if let Some(zero_rows) = initial_zero_rows {
        if zero_rows.is_empty() || zero_rows.len() > num_data / 2 {
            return Err(RegressionError::InvalidZeroWeightSpec);
        }
        for &j in zero_rows {
            if j >= num_data {
                return Err(RegressionError::InvalidZeroWeightSpec);
            }
            x[idx(j, wgt_col)] = 0.0;
            x[idx(j, prev_col)] = 0.0;
        }
    }

    let mut rms_err: f32 = 0.0;
    let mut wgt_rms_err: f32 = 0.0;
    let mut sol = vec![0.0f32; sol_size * num_out_col];
    let mut cons = if with_constant {
        Some(vec![0.0f32; num_out_col])
    } else {
        None
    };
    let mut x_mean = vec![0.0f32; num_inp_col + num_out_col];
    let mut x_sd = vec![0.0f32; num_inp_col + num_out_col];

    let mut split_last_time = 0i32;
    let mut keep_criterion = false;
    let mut criterion: f32 = 0.0;
    let mut iter = 0usize;

    for it in 0..max_iter {
        iter = it;

        // Get regression solution using the weight column
        let result = mult_regress(
            x,
            x_size,
            col_fast,
            num_inp_col,
            num_data,
            num_out_col,
            Some(wgt_col),
            sol_size,
            with_constant,
        )?;
        sol = result.sol;
        cons = result.cons;
        x_mean = result.x_mean;
        x_sd = result.x_sd;

        // Compute residuals
        let mut residuals = vec![0.0f32; num_data];
        rms_err = 0.0;
        wgt_rms_err = 0.0;

        for j in 0..num_data {
            if num_out_col == 1 {
                let mut colres = if let Some(ref c) = cons { c[0] } else { 0.0 };
                colres -= x[idx(j, num_inp_col)];
                for i in 0..num_inp_col {
                    colres += x[idx(j, i)] * sol[i];
                }
                residuals[j] = colres;
            } else {
                let mut ressum: f32 = 0.0;
                for k in 0..num_out_col {
                    let mut colres = if let Some(ref c) = cons { c[k] } else { 0.0 };
                    colres -= x[idx(j, num_inp_col + k)];
                    for i in 0..num_inp_col {
                        colres += x[idx(j, i)] * sol[i + k * sol_size];
                    }
                    ressum += colres * colres;
                }
                residuals[j] = ressum.sqrt();
            }
            rms_err += residuals[j] * residuals[j];
            wgt_rms_err += residuals[j] * residuals[j] * x[idx(j, wgt_col)];
        }
        rms_err = (rms_err / num_data as f32).sqrt();
        wgt_rms_err = (wgt_rms_err / num_data as f32).sqrt();

        // Get median and MADN
        let (med, _) = median(&residuals);
        let (madn_val, _) = madn(&residuals, med);
        if !keep_criterion {
            criterion = kfactor * madn_val;
        }

        // Convert residuals to deviations from median
        let mut deviations = vec![0.0f32; num_data];
        let mut num_out = 0usize;
        for j in 0..num_data {
            let mut dev = residuals[j] - med;
            if num_out_col > 1 {
                dev = dev.max(0.0);
            } else if dev < 0.0 {
                dev = -dev;
            }
            deviations[j] = dev;
            if dev > criterion {
                num_out += 1;
            }
        }

        // Adjust criterion if too many points have zero weight
        if num_out > max_zero_wgt {
            let mut sorted = deviations.clone();
            sort_floats(&mut sorted);
            if max_zero_wgt > 0 {
                criterion = (sorted[num_data - max_zero_wgt]
                    + sorted[num_data - max_zero_wgt - 1])
                    / 2.0;
            } else {
                criterion =
                    sorted[num_data - 1] / (1.0 - min_non_zero_wgt.sqrt()).sqrt();
            }
            keep_criterion = true;
        }

        // Compute new weights and evaluate changes
        let mut diffmax: f32 = 0.0;
        let mut prevmax: f32 = 0.0;

        for j in 0..num_data {
            let weight;
            if deviations[j] > criterion {
                weight = 0.0;
            } else if deviations[j] <= 1.0e-6 * criterion {
                weight = 1.0;
            } else {
                let dev = deviations[j] / criterion;
                let t = 1.0 - dev * dev;
                weight = t * t;
            }

            let diff = (weight - x[idx(j, wgt_col)]).abs();
            if diff > diffmax {
                diffmax = diff;
            }
            let prev = (weight - x[idx(j, prev_col)]).abs();
            if prev > prevmax {
                prevmax = prev;
            }
            x[idx(j, prev_col)] = x[idx(j, wgt_col)];
            x[idx(j, wgt_col)] = weight;
        }

        // Check convergence
        if split_last_time == 0
            && (diffmax < max_change || (diffmax < max_oscill && prevmax < max_change))
        {
            break;
        }

        // Handle oscillation: try half-way between
        if split_last_time == 0 && prevmax < max_change / 2.0 {
            split_last_time = 1;
            for j in 0..num_data {
                let avg = (x[idx(j, wgt_col)] + x[idx(j, prev_col)]) / 2.0;
                x[idx(j, wgt_col)] = avg;
            }
        } else if split_last_time == 1 {
            split_last_time = 2;
        } else {
            split_last_time = 0;
        }
    }

    if iter >= max_iter {
        return Err(RegressionError::NotConverged);
    }

    Ok(RobustRegressResult {
        sol,
        cons,
        x_mean,
        x_sd,
        num_iter: iter,
        rms_err,
        wgt_rms_err,
    })
}

// ---------------------------------------------------------------------------
// Polynomial fitting
// ---------------------------------------------------------------------------

/// Result of a polynomial fit.
#[derive(Debug, Clone)]
pub struct PolyFitResult {
    /// Coefficients: `slopes[i]` is the coefficient of x^(i+1).
    pub slopes: Vec<f32>,
    /// Constant (intercept) term.
    pub intercept: f32,
}

/// Fits a polynomial of the given `order` to `ndata` points.
///
/// The fitted equation is:
///
/// ```text
/// Y = intercept + slopes[0]*X + slopes[1]*X^2 + ... + slopes[order-1]*X^order
/// ```
///
/// Uses [`mult_regress`] internally.
///
/// # Errors
///
/// Returns [`RegressionError::ZeroOrder`] if `order` is 0, or propagates
/// errors from [`mult_regress`].
pub fn polynomial_fit(x: &[f32], y: &[f32], ndata: usize, order: usize) -> Result<PolyFitResult, RegressionError> {
    if order == 0 {
        return Err(RegressionError::ZeroOrder);
    }
    let wdim = order + 1;

    // Build the data matrix: column-major with ndata rows
    // Columns 0..order-1 are x^1 .. x^order, column order is y
    let mut work = vec![0.0f32; wdim * ndata];
    for i in 0..ndata {
        for j in 0..order {
            work[i + j * ndata] = (x[i] as f64).powi(j as i32 + 1) as f32;
        }
        work[i + order * ndata] = y[i];
    }

    let result = mult_regress(&work, ndata, false, order, ndata, 1, None, ndata, true)?;

    Ok(PolyFitResult {
        slopes: result.sol[..order].to_vec(),
        intercept: result.cons.unwrap()[0],
    })
}

/// Fits a weighted polynomial of the given `order` to `ndata` points.
///
/// Same equation as [`polynomial_fit`], but each data point has an associated
/// weight in the `weight` slice.
///
/// # Errors
///
/// Returns [`RegressionError::ZeroOrder`] if `order` is 0, or propagates
/// errors from [`mult_regress`].
pub fn weighted_poly_fit(
    x: &[f32],
    y: &[f32],
    weight: &[f32],
    ndata: usize,
    order: usize,
) -> Result<PolyFitResult, RegressionError> {
    if order == 0 {
        return Err(RegressionError::ZeroOrder);
    }

    // Build data matrix: columns 0..order-1 = x^1..x^order, column order = y,
    // column order+1 = weight
    let ncols = order + 2;
    let mut work = vec![0.0f32; ncols * ndata];
    for i in 0..ndata {
        for j in 0..order {
            work[i + j * ndata] = (x[i] as f64).powi(j as i32 + 1) as f32;
        }
        work[i + order * ndata] = y[i];
        work[i + (order + 1) * ndata] = weight[i];
    }

    let result = mult_regress(
        &work,
        ndata,
        false,
        order,
        ndata,
        1,
        Some(order + 1),
        ndata,
        true,
    )?;

    Ok(PolyFitResult {
        slopes: result.sol[..order].to_vec(),
        intercept: result.cons.unwrap()[0],
    })
}

/// Result of a robust polynomial fit.
#[derive(Debug, Clone)]
pub struct RobustPolyFitResult {
    /// Polynomial coefficients (same meaning as [`PolyFitResult::slopes`]).
    pub slopes: Vec<f32>,
    /// Constant (intercept) term.
    pub intercept: f32,
    /// Number of robust-regression iterations performed.
    pub num_iter: usize,
    /// Final weights for each data point.
    pub weights: Vec<f32>,
}

/// Fits a polynomial using robust regression with Tukey bisquare weighting.
///
/// Same equation as [`polynomial_fit`]. Uses fixed convergence parameters:
/// `max_change = 0.02`, `max_oscill = 0.05`.
///
/// # Errors
///
/// Returns [`RegressionError::ZeroOrder`] if `order` is 0, or propagates
/// errors from [`robust_regress`].
pub fn robust_poly_fit(
    x: &[f32],
    y: &[f32],
    ndata: usize,
    order: usize,
    kfactor: f32,
    max_iter: usize,
    max_zero_wgt: usize,
) -> Result<RobustPolyFitResult, RegressionError> {
    if order == 0 {
        return Err(RegressionError::ZeroOrder);
    }

    // Build data matrix with extra 2 columns for weights
    // Columns: 0..order-1 = x^1..x^order, order = y, order+1 = wgt, order+2 = prev_wgt
    let ncols = order + 3;
    let mut work = vec![0.0f32; ncols * ndata];
    for i in 0..ndata {
        for j in 0..order {
            work[i + j * ndata] = (x[i] as f64).powi(j as i32 + 1) as f32;
        }
        work[i + order * ndata] = y[i];
    }

    let params = RobustParams {
        kfactor,
        max_iter,
        max_zero_wgt,
        max_change: 0.02,
        max_oscill: 0.05,
    };

    let result = robust_regress(
        &mut work,
        ndata,
        false,
        order,
        ndata,
        1,
        order,
        true,
        &params,
        None,
    )?;

    // Extract final weights from the weight column (column order+1)
    let mut weights = vec![0.0f32; ndata];
    for i in 0..ndata {
        weights[i] = work[i + (order + 1) * ndata];
    }

    Ok(RobustPolyFitResult {
        slopes: result.sol[..order].to_vec(),
        intercept: result.cons.unwrap()[0],
        num_iter: result.num_iter,
        weights,
    })
}

/// Result of [`robust_poly_smooth`].
#[derive(Debug, Clone)]
pub struct RobustPolySmoothResult {
    /// Smoothed Y values.
    pub y_out: Vec<f32>,
    /// Optional weight for each data point from the fit that smoothed it.
    pub weights: Option<Vec<f32>>,
}

/// Smooths data by fitting a polynomial of the given `order` to `num_fit`
/// points around each data point, using robust polynomial fitting where
/// possible and falling back to ordinary polynomial fitting.
///
/// Near the ends of the data, at least `min_fit` points are used.
///
/// # Parameters
///
/// - `x`, `y_in`: The data coordinates.
/// - `order`: Polynomial order.
/// - `num_fit`: Number of points in the fitting window.
/// - `min_fit`: Minimum number of points used near the ends.
/// - `kfactor`, `max_iter`, `max_zero_wgt`: Robust regression parameters.
/// - `compute_weights`: Whether to return per-point weights.
///
/// # Errors
///
/// Returns [`RegressionError::TooFewData`] if `min_fit > ndata`, or propagates
/// errors from [`polynomial_fit`].
pub fn robust_poly_smooth(
    x: &[f32],
    y_in: &[f32],
    order: usize,
    num_fit: usize,
    min_fit: usize,
    kfactor: f32,
    max_iter: usize,
    max_zero_wgt: usize,
    compute_weights: bool,
) -> Result<RobustPolySmoothResult, RegressionError> {
    let ndata = x.len();
    if min_fit > ndata {
        return Err(RegressionError::TooFewData);
    }

    let mut y_out = vec![0.0f32; ndata];
    let mut weights = if compute_weights {
        Some(vec![1.0f32; ndata])
    } else {
        None
    };

    for ind in 0..ndata {
        // Determine fitting range
        let mut fit_start = ind as i64 - num_fit as i64 / 2;
        let mut fit_end = fit_start + num_fit as i64 - 1;
        if fit_end >= ndata as i64 {
            fit_end = ndata as i64 - 1;
            fit_start = fit_start.min(ndata as i64 - min_fit as i64);
        }
        if fit_start < 0 {
            fit_start = 0;
            fit_end = fit_end.max(min_fit as i64 - 1);
        }
        let fit_start = fit_start as usize;
        let fit_end = fit_end as usize;
        let this_fit = fit_end + 1 - fit_start;

        // Compute local means and subtract
        let mut xfit: Vec<f32> = x[fit_start..=fit_end].to_vec();
        let mut yfit: Vec<f32> = y_in[fit_start..=fit_end].to_vec();
        let xmean: f32 = xfit.iter().sum::<f32>() / this_fit as f32;
        let ymean: f32 = yfit.iter().sum::<f32>() / this_fit as f32;
        for j in 0..this_fit {
            xfit[j] -= xmean;
            yfit[j] -= ymean;
        }

        // Try robust fit, fall back to regular
        let fit_result = robust_poly_fit(&xfit, &yfit, this_fit, order, kfactor, max_iter, max_zero_wgt);
        let (slopes, intercept) = match fit_result {
            Ok(ref r) => {
                if let Some(ref mut w) = weights {
                    w[ind] = r.weights[ind - fit_start];
                }
                (r.slopes.clone(), r.intercept)
            }
            Err(_) => {
                if let Some(ref mut w) = weights {
                    w[ind] = 1.0;
                }
                let pf = polynomial_fit(&xfit, &yfit, this_fit, order)?;
                (pf.slopes, pf.intercept)
            }
        };

        y_out[ind] = intercept + ymean;
        for j in 0..order {
            y_out[ind] += slopes[j] * (x[ind] - xmean).powi(j as i32 + 1);
        }
    }

    Ok(RobustPolySmoothResult { y_out, weights })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_fit_linear() {
        // Fit y = 2x + 1
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let result = polynomial_fit(&x, &y, 10, 1).unwrap();
        assert!((result.intercept - 1.0).abs() < 1e-3, "intercept={}", result.intercept);
        assert!((result.slopes[0] - 2.0).abs() < 1e-3, "slope={}", result.slopes[0]);
    }

    #[test]
    fn test_polynomial_fit_quadratic() {
        // Fit y = 0.5*x^2 - x + 3
        let x: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let y: Vec<f32> = x.iter().map(|&xi| 0.5 * xi * xi - xi + 3.0).collect();
        let result = polynomial_fit(&x, &y, 20, 2).unwrap();
        assert!((result.intercept - 3.0).abs() < 0.1, "intercept={}", result.intercept);
        assert!((result.slopes[0] - (-1.0)).abs() < 0.1, "slope0={}", result.slopes[0]);
        assert!((result.slopes[1] - 0.5).abs() < 0.1, "slope1={}", result.slopes[1]);
    }

    #[test]
    fn test_zero_order_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(polynomial_fit(&x, &y, 3, 0).is_err());
    }

    #[test]
    fn test_stat_matrices_basic() {
        // 3 variables, 4 measurements, col_fast=true means row i col j = x[j + i*xsize]
        // xsize = 3
        let x = vec![
            1.0, 2.0, 3.0,   // row 0
            2.0, 4.0, 5.0,   // row 1
            3.0, 6.0, 7.0,   // row 2
            4.0, 8.0, 9.0,   // row 3
        ];
        let result = stat_matrices(&x, 3, true, 3, 3, 4, 1);
        // Means should be 2.5, 5.0, 6.0
        assert!((result.xm[0] - 2.5).abs() < 1e-5);
        assert!((result.xm[1] - 5.0).abs() < 1e-5);
        assert!((result.xm[2] - 6.0).abs() < 1e-5);
    }
}
