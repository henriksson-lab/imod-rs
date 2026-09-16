//! Discrete transform functions from `IMOD/raptor/opencv/cxdxt.cpp`.
//!
//! `rustfft` replaces the source unit's 32- and 64-bit butterfly kernels and
//! IPP dispatch.  [`IcvDftPlan`] retains the source factor/twiddle planning
//! state; the explicit CCS routines cover the source real packing/copy paths.

use num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;

pub const CV_DXT_FORWARD: i32 = 0;
pub const CV_DXT_INVERSE: i32 = 1;
pub const CV_DXT_SCALE: i32 = 2;
pub const CV_DXT_ROWS: i32 = 4;
pub const CV_DXT_MUL_CONJ: i32 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvDxtError {
    BadArgument,
    UnmatchedSizes,
}

/// Source `icvlog2`, for positive power-of-two planning sizes.
pub fn icv_log2(value: usize) -> Result<usize, CvDxtError> {
    if value == 0 {
        return Err(CvDxtError::BadArgument);
    }
    Ok(value.ilog2() as usize)
}

/// Source `icvDFTFactorize`, retained as owned factor metadata even though the
/// transform engine below delegates mixed-radix execution to `rustfft`.
pub fn icv_dft_factorize(mut value: usize) -> Result<Vec<usize>, CvDxtError> {
    if value == 0 {
        return Err(CvDxtError::BadArgument);
    }
    let mut factors = Vec::new();
    let mut factor = 2;
    while factor * factor <= value {
        while value % factor == 0 {
            factors.push(factor);
            value /= factor;
        }
        factor += if factor == 2 { 1 } else { 2 };
    }
    if value > 1 {
        factors.push(value);
    }
    if factors.is_empty() {
        factors.push(1);
    }
    Ok(factors)
}

/// `icvRealDFT_*`: forward real DFT in OpenCV's one-dimensional packed CCS
/// layout.  Even lengths are `[R0, R1, I1, ..., Rn/2]`; odd lengths are
/// `[R0, R1, I1, ...]`.
pub fn icv_real_dft_ccs(source: &[f64], destination: &mut [f64]) -> Result<(), CvDxtError> {
    if source.is_empty() || destination.len() != source.len() {
        return Err(CvDxtError::BadArgument);
    }
    let mut values: Vec<Complex64> = source
        .iter()
        .copied()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(values.len()).process(&mut values);
    destination[0] = values[0].re;
    if values.len() % 2 == 0 {
        for index in 1..values.len() / 2 {
            destination[2 * index - 1] = values[index].re;
            destination[2 * index] = values[index].im;
        }
        destination[values.len() - 1] = values[values.len() / 2].re;
    } else {
        for index in 1..=(values.len() / 2) {
            destination[2 * index - 1] = values[index].re;
            destination[2 * index] = values[index].im;
        }
    }
    Ok(())
}

/// Source `icvRealDFT_64f` macro instance.
pub fn icv_real_dft_64f(source: &[f64], destination: &mut [f64]) -> Result<(), CvDxtError> {
    icv_real_dft_ccs(source, destination)
}

/// Source `icvRealDFT_32f` macro instance, preserving single-precision output.
pub fn icv_real_dft_32f(source: &[f32], destination: &mut [f32]) -> Result<(), CvDxtError> {
    if source.is_empty() || destination.len() != source.len() {
        return Err(CvDxtError::BadArgument);
    }
    let mut values: Vec<Complex32> = source
        .iter()
        .copied()
        .map(|value| Complex32::new(value, 0.0))
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_forward(values.len()).process(&mut values);
    destination[0] = values[0].re;
    if values.len() % 2 == 0 {
        for index in 1..values.len() / 2 {
            destination[2 * index - 1] = values[index].re;
            destination[2 * index] = values[index].im;
        }
        destination[values.len() - 1] = values[values.len() / 2].re;
    } else {
        for index in 1..=values.len() / 2 {
            destination[2 * index - 1] = values[index].re;
            destination[2 * index] = values[index].im;
        }
    }
    Ok(())
}

/// Source `icvExpandCCS`, expressed as an owned full Hermitian spectrum.
///
/// The source expands in-place to make room for complex values. An owned
/// vector is the equivalent safe representation: index `k` is the coefficient
/// at frequency `k`, including the conjugate half that CCS omits.
pub fn icv_expand_ccs(source: &[f64]) -> Result<Vec<Complex64>, CvDxtError> {
    if source.is_empty() {
        return Err(CvDxtError::BadArgument);
    }
    let n = source.len();
    let mut values = vec![Complex64::default(); n];
    values[0] = Complex64::new(source[0], 0.0);
    if n % 2 == 0 {
        values[n / 2] = Complex64::new(source[n - 1], 0.0);
        for index in 1..n / 2 {
            values[index] = Complex64::new(source[2 * index - 1], source[2 * index]);
            values[n - index] = values[index].conj();
        }
    } else {
        for index in 1..=n / 2 {
            values[index] = Complex64::new(source[2 * index - 1], source[2 * index]);
            values[n - index] = values[index].conj();
        }
    }
    Ok(values)
}

/// Source `icvCopyColumn`, with strides measured in Rust elements rather than
/// bytes. `T` is one complete source element (scalar or complex), so its copy
/// is independent of the source's C `elem_size` branches.
pub fn icv_copy_column<T: Copy>(
    source: &[T],
    source_step: usize,
    destination: &mut [T],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0 {
        return Ok(());
    }
    if source_step == 0
        || destination_step == 0
        || (length - 1)
            .checked_mul(source_step)
            .map_or(true, |last| last >= source.len())
        || (length - 1)
            .checked_mul(destination_step)
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for index in 0..length {
        destination[index * destination_step] = source[index * source_step];
    }
    Ok(())
}

/// Source `icvCopyFrom2Columns`: deinterleave contiguous pairs from a strided
/// matrix column pair into two owned contiguous columns.
pub fn icv_copy_from_2_columns<T: Copy>(
    source: &[T],
    source_step: usize,
    destination_first: &mut [T],
    destination_second: &mut [T],
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0 {
        return Ok(());
    }
    if source_step < 2
        || destination_first.len() < length
        || destination_second.len() < length
        || (length - 1)
            .checked_mul(source_step)
            .and_then(|last| last.checked_add(1))
            .map_or(true, |last| last >= source.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for index in 0..length {
        destination_first[index] = source[index * source_step];
        destination_second[index] = source[index * source_step + 1];
    }
    Ok(())
}

/// Source `icvCopyTo2Columns`: interleave two contiguous columns into a
/// strided matrix column pair.
pub fn icv_copy_to_2_columns<T: Copy>(
    source_first: &[T],
    source_second: &[T],
    destination: &mut [T],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0 {
        return Ok(());
    }
    if destination_step < 2
        || source_first.len() < length
        || source_second.len() < length
        || (length - 1)
            .checked_mul(destination_step)
            .and_then(|last| last.checked_add(1))
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for index in 0..length {
        destination[index * destination_step] = source_first[index];
        destination[index * destination_step + 1] = source_second[index];
    }
    Ok(())
}

/// `icvCCSIDFT_*`: inverse one-dimensional packed CCS transform.
pub fn icv_ccs_idft(source: &[f64], destination: &mut [f64], flags: i32) -> Result<(), CvDxtError> {
    if source.is_empty() || destination.len() != source.len() {
        return Err(CvDxtError::BadArgument);
    }
    let n = source.len();
    let mut values = icv_expand_ccs(source)?;
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_inverse(n).process(&mut values);
    let scale = if flags & CV_DXT_SCALE != 0 {
        n as f64
    } else {
        1.0
    };
    for index in 0..n {
        destination[index] = values[index].re / scale;
    }
    Ok(())
}

/// Source `icvCCSIDFT_64f` macro instance.
pub fn icv_ccs_idft_64f(
    source: &[f64],
    destination: &mut [f64],
    flags: i32,
) -> Result<(), CvDxtError> {
    icv_ccs_idft(source, destination, flags)
}

/// Source `icvCCSIDFT_32f` macro instance.
pub fn icv_ccs_idft_32f(
    source: &[f32],
    destination: &mut [f32],
    flags: i32,
) -> Result<(), CvDxtError> {
    if source.is_empty() || destination.len() != source.len() {
        return Err(CvDxtError::BadArgument);
    }
    let n = source.len();
    let mut values = vec![Complex32::default(); n];
    values[0] = Complex32::new(source[0], 0.0);
    if n % 2 == 0 {
        values[n / 2] = Complex32::new(source[n - 1], 0.0);
        for index in 1..n / 2 {
            values[index] = Complex32::new(source[2 * index - 1], source[2 * index]);
            values[n - index] = values[index].conj();
        }
    } else {
        for index in 1..=n / 2 {
            values[index] = Complex32::new(source[2 * index - 1], source[2 * index]);
            values[n - index] = values[index].conj();
        }
    }
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(n).process(&mut values);
    let scale = if flags & CV_DXT_SCALE != 0 {
        n as f32
    } else {
        1.0
    };
    for index in 0..n {
        destination[index] = values[index].re / scale;
    }
    Ok(())
}

/// Forward two-dimensional real `cvDFT` in the source's packed CCS matrix layout.
pub fn cv_dft_real_ccs_2d(
    source: &[f64],
    destination: &mut [f64],
    rows: usize,
    columns: usize,
    flags: i32,
    nonzero_rows: usize,
) -> Result<(), CvDxtError> {
    if rows == 0
        || columns == 0
        || source.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
        || destination.len() != source.len()
        || nonzero_rows > rows
    {
        return Err(CvDxtError::BadArgument);
    }
    if flags & CV_DXT_ROWS != 0 {
        for row in 0..rows {
            icv_real_dft_ccs(
                &source[row * columns..(row + 1) * columns],
                &mut destination[row * columns..(row + 1) * columns],
            )?;
        }
        return Ok(());
    }
    let mut spectrum: Vec<Complex64> = source
        .iter()
        .copied()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    if nonzero_rows != 0 {
        for row in nonzero_rows..rows {
            spectrum[row * columns..(row + 1) * columns].fill(Complex64::default());
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let row_fft = planner.plan_fft_forward(columns);
    for row in 0..rows {
        row_fft.process(&mut spectrum[row * columns..(row + 1) * columns]);
    }
    let column_fft = planner.plan_fft_forward(rows);
    let mut vertical = vec![Complex64::default(); rows];
    for column in 0..columns {
        for row in 0..rows {
            vertical[row] = spectrum[row * columns + column];
        }
        column_fft.process(&mut vertical);
        for row in 0..rows {
            spectrum[row * columns + column] = vertical[row];
        }
    }
    // The first (and, for even widths, final) horizontal frequencies are
    // Hermitian vertical spectra.  Their CCS columns are the source's
    // interleaved R/I representation directly, not another DFT.
    destination[0] = spectrum[0].re;
    for frequency in 1..=(rows - 1) / 2 {
        destination[(2 * frequency - 1) * columns] = spectrum[frequency * columns].re;
        destination[2 * frequency * columns] = spectrum[frequency * columns].im;
    }
    if rows % 2 == 0 {
        destination[(rows - 1) * columns] = spectrum[rows / 2 * columns].re;
    }
    if columns % 2 == 0 {
        destination[columns - 1] = spectrum[columns / 2].re;
        for frequency in 1..=(rows - 1) / 2 {
            destination[(2 * frequency - 1) * columns + columns - 1] =
                spectrum[frequency * columns + columns / 2].re;
            destination[2 * frequency * columns + columns - 1] =
                spectrum[frequency * columns + columns / 2].im;
        }
        if rows % 2 == 0 {
            destination[(rows - 1) * columns + columns - 1] =
                spectrum[rows / 2 * columns + columns / 2].re;
        }
    }
    for frequency in 1..=(columns - 1) / 2 {
        for row in 0..rows {
            destination[row * columns + 2 * frequency - 1] = spectrum[row * columns + frequency].re;
            destination[row * columns + 2 * frequency] = spectrum[row * columns + frequency].im;
        }
    }
    Ok(())
}

/// Inverse two-dimensional packed CCS `cvDFT` to real samples.
pub fn cv_dft_ccs_2d(
    source: &[f64],
    destination: &mut [f64],
    rows: usize,
    columns: usize,
    flags: i32,
) -> Result<(), CvDxtError> {
    if rows == 0
        || columns == 0
        || source.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
        || destination.len() != source.len()
    {
        return Err(CvDxtError::BadArgument);
    }
    if flags & CV_DXT_ROWS != 0 {
        for row in 0..rows {
            icv_ccs_idft(
                &source[row * columns..(row + 1) * columns],
                &mut destination[row * columns..(row + 1) * columns],
                flags,
            )?;
        }
        return Ok(());
    }
    let mut spectrum = vec![Complex64::default(); rows * columns];
    let mut packed = vec![0.0; rows];
    for row in 0..rows {
        packed[row] = source[row * columns];
    }
    let vertical_spectrum = icv_expand_ccs(&packed)?;
    for row in 0..rows {
        spectrum[row * columns] = vertical_spectrum[row];
    }
    if columns % 2 == 0 {
        for row in 0..rows {
            packed[row] = source[row * columns + columns - 1];
        }
        let vertical_spectrum = icv_expand_ccs(&packed)?;
        for row in 0..rows {
            spectrum[row * columns + columns / 2] = vertical_spectrum[row];
        }
    }
    for frequency in 1..=(columns - 1) / 2 {
        for row in 0..rows {
            spectrum[row * columns + frequency] = Complex64::new(
                source[row * columns + 2 * frequency - 1],
                source[row * columns + 2 * frequency],
            );
            let mirror_row = (rows - row) % rows;
            spectrum[mirror_row * columns + columns - frequency] =
                spectrum[row * columns + frequency].conj();
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let column_fft = planner.plan_fft_inverse(rows);
    let mut vertical = vec![Complex64::default(); rows];
    for column in 0..columns {
        for row in 0..rows {
            vertical[row] = spectrum[row * columns + column];
        }
        column_fft.process(&mut vertical);
        for row in 0..rows {
            spectrum[row * columns + column] = vertical[row];
        }
    }
    let row_fft = planner.plan_fft_inverse(columns);
    let scale = if flags & CV_DXT_SCALE != 0 {
        (rows * columns) as f64
    } else {
        1.0
    };
    for row in 0..rows {
        row_fft.process(&mut spectrum[row * columns..(row + 1) * columns]);
        for column in 0..columns {
            destination[row * columns + column] = spectrum[row * columns + column].re / scale;
        }
    }
    Ok(())
}

/// Forward real `cvDFT` to the source-supported compact complex shape
/// `rows × (columns / 2 + 1)`.
pub fn cv_dft_real_to_complex_2d(
    source: &[f64],
    destination: &mut [Complex64],
    rows: usize,
    columns: usize,
    flags: i32,
    nonzero_rows: usize,
) -> Result<(), CvDxtError> {
    let compact_columns = columns / 2 + 1;
    if rows == 0
        || columns == 0
        || source.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
        || destination.len()
            != rows
                .checked_mul(compact_columns)
                .ok_or(CvDxtError::BadArgument)?
        || nonzero_rows > rows
    {
        return Err(CvDxtError::BadArgument);
    }
    let mut full: Vec<Complex64> = source
        .iter()
        .copied()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    if nonzero_rows != 0 {
        for row in nonzero_rows..rows {
            full[row * columns..(row + 1) * columns].fill(Complex64::default());
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let row_fft = planner.plan_fft_forward(columns);
    for row in 0..rows {
        row_fft.process(&mut full[row * columns..(row + 1) * columns]);
    }
    if flags & CV_DXT_ROWS == 0 {
        let column_fft = planner.plan_fft_forward(rows);
        let mut column = vec![Complex64::default(); rows];
        for x in 0..columns {
            for y in 0..rows {
                column[y] = full[y * columns + x];
            }
            column_fft.process(&mut column);
            for y in 0..rows {
                full[y * columns + x] = column[y];
            }
        }
    }
    for row in 0..rows {
        destination[row * compact_columns..(row + 1) * compact_columns]
            .copy_from_slice(&full[row * columns..row * columns + compact_columns]);
    }
    Ok(())
}

/// Inverse compact complex `cvDFT` to real output.  The omitted half-plane is
/// reconstructed by conjugate symmetry before the inverse transform.
pub fn cv_dft_complex_to_real_2d(
    source: &[Complex64],
    destination: &mut [f64],
    rows: usize,
    columns: usize,
    flags: i32,
) -> Result<(), CvDxtError> {
    let compact_columns = columns / 2 + 1;
    if rows == 0
        || columns == 0
        || source.len()
            != rows
                .checked_mul(compact_columns)
                .ok_or(CvDxtError::BadArgument)?
        || destination.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
    {
        return Err(CvDxtError::BadArgument);
    }
    let mut full = vec![Complex64::default(); rows * columns];
    for row in 0..rows {
        full[row * columns..row * columns + compact_columns]
            .copy_from_slice(&source[row * compact_columns..(row + 1) * compact_columns]);
    }
    for row in 0..rows {
        for column in compact_columns..columns {
            let mirror_row = (rows - row) % rows;
            full[row * columns + column] = full[mirror_row * columns + columns - column].conj();
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    if flags & CV_DXT_ROWS == 0 {
        let column_fft = planner.plan_fft_inverse(rows);
        let mut column = vec![Complex64::default(); rows];
        for x in 0..columns {
            for y in 0..rows {
                column[y] = full[y * columns + x];
            }
            column_fft.process(&mut column);
            for y in 0..rows {
                full[y * columns + x] = column[y];
            }
        }
    }
    let row_fft = planner.plan_fft_inverse(columns);
    let scale = if flags & CV_DXT_SCALE != 0 {
        (if flags & CV_DXT_ROWS == 0 {
            rows * columns
        } else {
            columns
        }) as f64
    } else {
        1.0
    };
    for row in 0..rows {
        row_fft.process(&mut full[row * columns..(row + 1) * columns]);
        for column in 0..columns {
            destination[row * columns + column] = full[row * columns + column].re / scale;
        }
    }
    Ok(())
}

/// Source `icvDFTInit` metadata, retained for auditability even when
/// `rustfft` executes the corresponding mixed-radix butterflies.
#[derive(Clone, Debug, PartialEq)]
pub struct IcvDftPlan {
    pub length: usize,
    pub factors: Vec<usize>,
    pub wave: Vec<Complex64>,
    pub inverse: bool,
}

/// Safe owned equivalent of `icvDFTInit`'s factor and twiddle construction.
pub fn icv_dft_init(length: usize, inverse: bool) -> Result<IcvDftPlan, CvDxtError> {
    let factors = icv_dft_factorize(length)?;
    let sign = if inverse { 1.0 } else { -1.0 };
    let wave = (0..length)
        .map(|index| {
            Complex64::from_polar(
                1.0,
                sign * 2.0 * core::f64::consts::PI * index as f64 / length as f64,
            )
        })
        .collect();
    Ok(IcvDftPlan {
        length,
        factors,
        wave,
        inverse,
    })
}

/// Source `icvDFT_64fc` kernel, using RustFFT for the source's mixed-radix
/// butterfly execution. `scale` is applied after the transform.
pub fn icv_dft_64fc(values: &mut [Complex64], inverse: bool, scale: f64) -> Result<(), CvDxtError> {
    if values.is_empty() || !scale.is_finite() {
        return Err(CvDxtError::BadArgument);
    }
    let mut planner = FftPlanner::<f64>::new();
    if inverse {
        planner.plan_fft_inverse(values.len()).process(values);
    } else {
        planner.plan_fft_forward(values.len()).process(values);
    }
    if scale != 1.0 {
        for value in values {
            *value *= scale;
        }
    }
    Ok(())
}

/// Source `icvDFT_32fc` kernel, retaining its single-precision arithmetic.
pub fn icv_dft_32fc(values: &mut [Complex32], inverse: bool, scale: f32) -> Result<(), CvDxtError> {
    if values.is_empty() || !scale.is_finite() {
        return Err(CvDxtError::BadArgument);
    }
    let mut planner = FftPlanner::<f32>::new();
    if inverse {
        planner.plan_fft_inverse(values.len()).process(values);
    } else {
        planner.plan_fft_forward(values.len()).process(values);
    }
    if scale != 1.0 {
        for value in values {
            *value *= scale;
        }
    }
    Ok(())
}

/// Source `icvDCTInit`'s reusable cosine table, without C allocation state.
#[derive(Clone, Debug, PartialEq)]
pub struct IcvDctPlan {
    pub length: usize,
    pub cosine: Vec<f64>,
    pub inverse: bool,
}

/// Safe owned equivalent of `icvDCTInit`.
pub fn icv_dct_init(length: usize, inverse: bool) -> Result<IcvDctPlan, CvDxtError> {
    if length == 0 {
        return Err(CvDxtError::BadArgument);
    }
    let cosine = (0..length)
        .flat_map(|frequency| {
            (0..length).map(move |sample| {
                (core::f64::consts::PI * (sample as f64 + 0.5) * frequency as f64 / length as f64)
                    .cos()
            })
        })
        .collect();
    Ok(IcvDctPlan {
        length,
        cosine,
        inverse,
    })
}

/// Owned complex-to-complex `cvDFT`.  `rows`/`columns` describe contiguous
/// row-major data.  `CV_DXT_ROWS` performs independent row transforms.
pub fn cv_dft(
    data: &mut [Complex64],
    rows: usize,
    columns: usize,
    flags: i32,
    nonzero_rows: usize,
) -> Result<(), CvDxtError> {
    if rows == 0
        || columns == 0
        || data.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
        || nonzero_rows > rows
    {
        return Err(CvDxtError::BadArgument);
    }
    let inverse = flags & CV_DXT_INVERSE != 0;
    let active_rows = if nonzero_rows == 0 {
        rows
    } else {
        nonzero_rows
    };
    for row in 0..active_rows {
        icv_dft_64fc(&mut data[row * columns..(row + 1) * columns], inverse, 1.0)?;
    }
    if !inverse && nonzero_rows != 0 {
        for row in active_rows..rows {
            data[row * columns..(row + 1) * columns].fill(Complex64::default());
        }
    }
    if flags & CV_DXT_ROWS == 0 {
        let mut column = vec![Complex64::default(); rows];
        for x in 0..columns {
            for y in 0..rows {
                column[y] = data[y * columns + x];
            }
            icv_dft_64fc(&mut column, inverse, 1.0)?;
            for y in 0..rows {
                data[y * columns + x] = column[y];
            }
        }
    }
    if flags & CV_DXT_SCALE != 0 {
        let scale = (if flags & CV_DXT_ROWS != 0 {
            columns
        } else {
            rows * columns
        }) as f64;
        for value in data {
            *value /= scale;
        }
    }
    Ok(())
}

/// Owned `cvMulSpectrums` for complex spectra.
pub fn cv_mul_spectrums(
    first: &[Complex64],
    second: &[Complex64],
    destination: &mut [Complex64],
    flags: i32,
) -> Result<(), CvDxtError> {
    if first.len() != second.len() || destination.len() != first.len() {
        return Err(CvDxtError::UnmatchedSizes);
    }
    for index in 0..first.len() {
        destination[index] = first[index]
            * if flags & CV_DXT_MUL_CONJ != 0 {
                second[index].conj()
            } else {
                second[index]
            };
    }
    Ok(())
}

/// Packed one-dimensional CCS form of `cvMulSpectrums`.
///
/// OpenCV uses its one-channel real matrix representation to select this path;
/// the separate Rust signature makes that layout explicit. `CV_DXT_MUL_CONJ`
/// conjugates the second spectrum before multiplication.
pub fn cv_mul_spectrums_ccs(
    first: &[f64],
    second: &[f64],
    destination: &mut [f64],
    flags: i32,
) -> Result<(), CvDxtError> {
    if first.is_empty() || first.len() != second.len() || destination.len() != first.len() {
        return Err(CvDxtError::UnmatchedSizes);
    }
    let first_full = icv_expand_ccs(first)?;
    let second_full = icv_expand_ccs(second)?;
    let n = first.len();
    let mut product = vec![Complex64::default(); n];
    for index in 0..n {
        product[index] = first_full[index]
            * if flags & CV_DXT_MUL_CONJ != 0 {
                second_full[index].conj()
            } else {
                second_full[index]
            };
    }
    destination[0] = product[0].re;
    if n % 2 == 0 {
        for index in 1..n / 2 {
            destination[2 * index - 1] = product[index].re;
            destination[2 * index] = product[index].im;
        }
        destination[n - 1] = product[n / 2].re;
    } else {
        for index in 1..=n / 2 {
            destination[2 * index - 1] = product[index].re;
            destination[2 * index] = product[index].im;
        }
    }
    Ok(())
}

/// Source `icvDCT_fwd_64f`, with element rather than byte strides.
pub fn icv_dct_fwd_64f(
    source: &[f64],
    source_step: usize,
    destination: &mut [f64],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0
        || source_step == 0
        || destination_step == 0
        || (length - 1)
            .checked_mul(source_step)
            .map_or(true, |last| last >= source.len())
        || (length - 1)
            .checked_mul(destination_step)
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for frequency in 0..length {
        let mut sum = 0.0;
        for sample in 0..length {
            sum += source[sample * source_step]
                * (core::f64::consts::PI * (sample as f64 + 0.5) * frequency as f64
                    / length as f64)
                    .cos();
        }
        destination[frequency * destination_step] = sum
            * if frequency == 0 {
                (1.0 / length as f64).sqrt()
            } else {
                (2.0 / length as f64).sqrt()
            };
    }
    Ok(())
}

/// Source `icvDCT_inv_64f`, with element rather than byte strides.
pub fn icv_dct_inv_64f(
    source: &[f64],
    source_step: usize,
    destination: &mut [f64],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0
        || source_step == 0
        || destination_step == 0
        || (length - 1)
            .checked_mul(source_step)
            .map_or(true, |last| last >= source.len())
        || (length - 1)
            .checked_mul(destination_step)
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for sample in 0..length {
        let mut sum = 0.0;
        for frequency in 0..length {
            let weight = if frequency == 0 {
                (1.0 / length as f64).sqrt()
            } else {
                (2.0 / length as f64).sqrt()
            };
            sum += source[frequency * source_step]
                * weight
                * (core::f64::consts::PI * (sample as f64 + 0.5) * frequency as f64
                    / length as f64)
                    .cos();
        }
        destination[sample * destination_step] = sum;
    }
    Ok(())
}

/// Source `icvDCT_fwd_32f`, retaining its single-precision destination.
pub fn icv_dct_fwd_32f(
    source: &[f32],
    source_step: usize,
    destination: &mut [f32],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0
        || source_step == 0
        || destination_step == 0
        || (length - 1)
            .checked_mul(source_step)
            .map_or(true, |last| last >= source.len())
        || (length - 1)
            .checked_mul(destination_step)
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for frequency in 0..length {
        let mut sum = 0.0f64;
        for sample in 0..length {
            sum += source[sample * source_step] as f64
                * (core::f64::consts::PI * (sample as f64 + 0.5) * frequency as f64
                    / length as f64)
                    .cos();
        }
        destination[frequency * destination_step] = (sum
            * if frequency == 0 {
                (1.0 / length as f64).sqrt()
            } else {
                (2.0 / length as f64).sqrt()
            }) as f32;
    }
    Ok(())
}

/// Source `icvDCT_inv_32f`, retaining its single-precision destination.
pub fn icv_dct_inv_32f(
    source: &[f32],
    source_step: usize,
    destination: &mut [f32],
    destination_step: usize,
    length: usize,
) -> Result<(), CvDxtError> {
    if length == 0
        || source_step == 0
        || destination_step == 0
        || (length - 1)
            .checked_mul(source_step)
            .map_or(true, |last| last >= source.len())
        || (length - 1)
            .checked_mul(destination_step)
            .map_or(true, |last| last >= destination.len())
    {
        return Err(CvDxtError::BadArgument);
    }
    for sample in 0..length {
        let mut sum = 0.0f64;
        for frequency in 0..length {
            let weight = if frequency == 0 {
                (1.0 / length as f64).sqrt()
            } else {
                (2.0 / length as f64).sqrt()
            };
            sum += source[frequency * source_step] as f64
                * weight
                * (core::f64::consts::PI * (sample as f64 + 0.5) * frequency as f64
                    / length as f64)
                    .cos();
        }
        destination[sample * destination_step] = sum as f32;
    }
    Ok(())
}

/// Owned orthonormal DCT-II/DCT-III implementation of `cvDCT` for a 2D real matrix.
pub fn cv_dct(data: &mut [f64], rows: usize, columns: usize, flags: i32) -> Result<(), CvDxtError> {
    if rows == 0
        || columns == 0
        || data.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
    {
        return Err(CvDxtError::BadArgument);
    }
    let inverse = flags & CV_DXT_INVERSE != 0;
    let mut temporary = vec![0.0; data.len()];
    for row in 0..rows {
        if inverse {
            icv_dct_inv_64f(
                &data[row * columns..(row + 1) * columns],
                1,
                &mut temporary[row * columns..(row + 1) * columns],
                1,
                columns,
            )?;
        } else {
            icv_dct_fwd_64f(
                &data[row * columns..(row + 1) * columns],
                1,
                &mut temporary[row * columns..(row + 1) * columns],
                1,
                columns,
            )?;
        }
    }
    if flags & CV_DXT_ROWS != 0 {
        data.copy_from_slice(&temporary);
        return Ok(());
    }
    let mut source_column = vec![0.0; rows];
    let mut destination_column = vec![0.0; rows];
    for column in 0..columns {
        for row in 0..rows {
            source_column[row] = temporary[row * columns + column];
        }
        if inverse {
            icv_dct_inv_64f(&source_column, 1, &mut destination_column, 1, rows)?;
        } else {
            icv_dct_fwd_64f(&source_column, 1, &mut destination_column, 1, rows)?;
        }
        for row in 0..rows {
            data[row * columns + column] = destination_column[row];
        }
    }
    Ok(())
}

/// Single-precision counterpart of `cvDCT`, backed by the source-named
/// `icvDCT_{fwd,inv}_32f` kernels.
pub fn cv_dct_32f(
    data: &mut [f32],
    rows: usize,
    columns: usize,
    flags: i32,
) -> Result<(), CvDxtError> {
    if rows == 0
        || columns == 0
        || data.len() != rows.checked_mul(columns).ok_or(CvDxtError::BadArgument)?
    {
        return Err(CvDxtError::BadArgument);
    }
    let inverse = flags & CV_DXT_INVERSE != 0;
    let mut temporary = vec![0.0f32; data.len()];
    for row in 0..rows {
        if inverse {
            icv_dct_inv_32f(
                &data[row * columns..(row + 1) * columns],
                1,
                &mut temporary[row * columns..(row + 1) * columns],
                1,
                columns,
            )?;
        } else {
            icv_dct_fwd_32f(
                &data[row * columns..(row + 1) * columns],
                1,
                &mut temporary[row * columns..(row + 1) * columns],
                1,
                columns,
            )?;
        }
    }
    if flags & CV_DXT_ROWS != 0 {
        data.copy_from_slice(&temporary);
        return Ok(());
    }
    let mut source_column = vec![0.0f32; rows];
    let mut destination_column = vec![0.0f32; rows];
    for column in 0..columns {
        for row in 0..rows {
            source_column[row] = temporary[row * columns + column];
        }
        if inverse {
            icv_dct_inv_32f(&source_column, 1, &mut destination_column, 1, rows)?;
        } else {
            icv_dct_fwd_32f(&source_column, 1, &mut destination_column, 1, rows)?;
        }
        for row in 0..rows {
            data[row * columns + column] = destination_column[row];
        }
    }
    Ok(())
}

/// `cvGetOptimalDFTSize`: the smallest `2^p 3^q 5^r >= size`.
pub fn cv_get_optimal_dft_size(size: usize) -> Option<usize> {
    if size == 0 {
        return Some(1);
    }
    let mut best = usize::MAX;
    let mut a = 1usize;
    while a < best {
        let mut b = a;
        while b < best {
            let mut c = b;
            while c < best {
                if c >= size {
                    best = c;
                }
                c = match c.checked_mul(5) {
                    Some(value) => value,
                    None => break,
                };
            }
            b = match b.checked_mul(3) {
                Some(value) => value,
                None => break,
            };
        }
        a = match a.checked_mul(2) {
            Some(value) => value,
            None => break,
        };
    }
    (best != usize::MAX).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dft_inverse_and_spectrum_product_preserve_complex_contract() {
        let original = vec![
            Complex64::new(1., 0.),
            Complex64::new(2., -1.),
            Complex64::new(0., 3.),
            Complex64::new(-2., 0.),
        ];
        let mut data = original.clone();
        cv_dft(&mut data, 1, 4, CV_DXT_FORWARD, 0).unwrap();
        let spectrum = data.clone();
        let mut product = vec![Complex64::default(); 4];
        cv_mul_spectrums(&spectrum, &spectrum, &mut product, CV_DXT_MUL_CONJ).unwrap();
        cv_dft(&mut data, 1, 4, CV_DXT_INVERSE | CV_DXT_SCALE, 0).unwrap();
        for (found, expected) in data.iter().zip(original) {
            assert!((*found - expected).norm() < 1e-10);
        }
        let mut single_precision = [Complex32::new(1.0, 0.0), Complex32::new(-1.0, 0.0)];
        icv_dft_32fc(&mut single_precision, false, 1.0).unwrap();
        assert!((single_precision[0] - Complex32::new(0.0, 0.0)).norm() < 1e-6);
        assert!((single_precision[1] - Complex32::new(2.0, 0.0)).norm() < 1e-6);
    }
    #[test]
    fn dct_roundtrip_and_optimal_sizes_work() {
        let mut data = vec![1., 2., 3., 4.];
        let original = data.clone();
        cv_dct(&mut data, 2, 2, 0).unwrap();
        cv_dct(&mut data, 2, 2, CV_DXT_INVERSE).unwrap();
        for (found, expected) in data.iter().zip(original) {
            assert!((found - expected).abs() < 1e-10);
        }
        assert_eq!(cv_get_optimal_dft_size(7), Some(8));
        assert_eq!(cv_get_optimal_dft_size(13), Some(15));
        let mut rows_only = vec![1., 2., 3., 4.];
        let original_rows = rows_only.clone();
        cv_dct(&mut rows_only, 2, 2, CV_DXT_ROWS).unwrap();
        cv_dct(&mut rows_only, 2, 2, CV_DXT_ROWS | CV_DXT_INVERSE).unwrap();
        for (found, expected) in rows_only.iter().zip(original_rows) {
            assert!((found - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn packed_ccs_matches_the_documented_even_and_odd_layouts() {
        let mut even = [0.0; 4];
        icv_real_dft_ccs(&[1., 0., -1., 0.], &mut even).unwrap();
        assert!((even[0], even[1], even[2], even[3]) == (0., 2., 0., 0.));
        let mut restored = [0.0; 4];
        icv_ccs_idft(&even, &mut restored, CV_DXT_INVERSE | CV_DXT_SCALE).unwrap();
        for (found, expected) in restored.iter().zip([1., 0., -1., 0.]) {
            assert!((found - expected).abs() < 1e-12);
        }
        let mut odd = [0.0; 5];
        icv_real_dft_ccs(&[1., 2., 3., 4., 5.], &mut odd).unwrap();
        assert_eq!(odd[0], 15.);
        assert_eq!(icv_dft_factorize(60).unwrap(), [2, 2, 3, 5]);
        assert_eq!(icv_log2(16), Ok(4));
    }

    #[test]
    fn single_precision_ccs_and_dct_variants_roundtrip() {
        let original = [1.0f32, -2.0, 3.0, 4.0];
        let mut packed = [0.0f32; 4];
        let mut restored = [0.0f32; 4];
        icv_real_dft_32f(&original, &mut packed).unwrap();
        icv_ccs_idft_32f(&packed, &mut restored, CV_DXT_INVERSE | CV_DXT_SCALE).unwrap();
        for (found, expected) in restored.iter().zip(original) {
            assert!((found - expected).abs() < 2e-5);
        }
        let mut dct = original;
        cv_dct_32f(&mut dct, 1, 4, 0).unwrap();
        cv_dct_32f(&mut dct, 1, 4, CV_DXT_INVERSE).unwrap();
        for (found, expected) in dct.iter().zip(original) {
            assert!((found - expected).abs() < 2e-5);
        }
    }

    #[test]
    fn two_dimensional_ccs_roundtrips_and_rows_mode_is_per_row() {
        let original = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let mut ccs = [0.0; 12];
        let mut restored = [0.0; 12];
        cv_dft_real_ccs_2d(&original, &mut ccs, 3, 4, 0, 0).unwrap();
        cv_dft_ccs_2d(&ccs, &mut restored, 3, 4, CV_DXT_INVERSE | CV_DXT_SCALE).unwrap();
        for (found, expected) in restored.iter().zip(original) {
            assert!((found - expected).abs() < 1e-10);
        }
        cv_dft_real_ccs_2d(&original, &mut ccs, 3, 4, CV_DXT_ROWS, 0).unwrap();
        let mut row = [0.0; 4];
        icv_real_dft_ccs(&original[..4], &mut row).unwrap();
        assert_eq!(&ccs[..4], &row);
    }

    #[test]
    fn compact_complex_shape_and_plan_metadata_roundtrip() {
        let original = [1., 2., 3., 4., 5., 6., 7., 8.];
        let mut compact = vec![Complex64::default(); 2 * 3];
        let mut restored = [0.0; 8];
        cv_dft_real_to_complex_2d(&original, &mut compact, 2, 4, 0, 0).unwrap();
        cv_dft_complex_to_real_2d(&compact, &mut restored, 2, 4, CV_DXT_INVERSE | CV_DXT_SCALE)
            .unwrap();
        for (found, expected) in restored.iter().zip(original) {
            assert!((found - expected).abs() < 1e-10);
        }
        let plan = icv_dft_init(6, false).unwrap();
        assert_eq!(plan.factors, [2, 3]);
        assert!(
            (plan.wave[1] - Complex64::from_polar(1.0, -core::f64::consts::PI / 3.0)).norm()
                < 1e-14
        );
        let dct_plan = icv_dct_init(3, false).unwrap();
        assert_eq!(dct_plan.cosine.len(), 9);
        assert!((dct_plan.cosine[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn packed_ccs_spectrum_product_preserves_the_real_layout() {
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        let mut product = [0.0; 4];
        icv_real_dft_ccs(&[1., 2., 0., 0.], &mut first).unwrap();
        icv_real_dft_ccs(&[2., 1., 0., 0.], &mut second).unwrap();
        cv_mul_spectrums_ccs(&first, &second, &mut product, 0).unwrap();
        let mut convolution = [0.0; 4];
        icv_ccs_idft(&product, &mut convolution, CV_DXT_INVERSE | CV_DXT_SCALE).unwrap();
        assert_eq!(convolution, [2.0, 5.0, 2.0, 0.0]);
    }

    #[test]
    fn strided_column_copy_primitives_preserve_elements() {
        let source = [10, 11, 12, 20, 21, 22, 30, 31, 32];
        let mut column = [0; 3];
        icv_copy_column(&source, 3, &mut column, 1, 3).unwrap();
        assert_eq!(column, [10, 20, 30]);
        let mut first = [0; 3];
        let mut second = [0; 3];
        icv_copy_from_2_columns(&source, 3, &mut first, &mut second, 3).unwrap();
        assert_eq!(first, [10, 20, 30]);
        assert_eq!(second, [11, 21, 31]);
        let mut interleaved = [0; 9];
        icv_copy_to_2_columns(&first, &second, &mut interleaved, 3, 3).unwrap();
        assert_eq!(&interleaved[..8], &[10, 11, 0, 20, 21, 0, 30, 31]);
    }
}
