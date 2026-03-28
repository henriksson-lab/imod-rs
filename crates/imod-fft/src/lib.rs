//! FFT (Fast Fourier Transform) operations for 1D and 2D image processing.
//!
//! Built on top of the [`rustfft`] crate, this module provides real-to-complex
//! and complex-to-real transforms in 1-D and 2-D, plus convenience functions
//! for computing power spectra and cross-correlations.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Compute 1D forward FFT (real to complex).
/// Input: real-valued slice of length n.
/// Output: complex values of length n/2 + 1.
pub fn fft_r2c_1d(input: &[f32]) -> Vec<Complex<f32>> {
    let n = input.len();
    let mut buffer: Vec<Complex<f32>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);
    buffer.truncate(n / 2 + 1);
    buffer
}

/// Compute 1D inverse FFT (complex to real).
/// Input: complex values of length n/2 + 1.
/// Output: real values of length n (caller specifies n).
pub fn fft_c2r_1d(input: &[Complex<f32>], n: usize) -> Vec<f32> {
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    // Copy the positive frequencies
    let copy_len = input.len().min(n);
    buffer[..copy_len].copy_from_slice(&input[..copy_len]);
    // Mirror the negative frequencies
    for i in 1..n / 2 {
        if i < input.len() {
            buffer[n - i] = input[i].conj();
        }
    }
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);
    let scale = 1.0 / n as f32;
    buffer.iter().map(|c| c.re * scale).collect()
}

/// Compute 2D forward FFT on a row-major float image (nx x ny).
/// Returns complex array in row-major order, each row has nx/2+1 complex values.
pub fn fft_r2c_2d(data: &[f32], nx: usize, ny: usize) -> Vec<Complex<f32>> {
    let nxc = nx / 2 + 1;
    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);

    // Step 1: FFT along rows
    let mut row_result = vec![Complex::new(0.0f32, 0.0); nxc * ny];
    for j in 0..ny {
        let mut row: Vec<Complex<f32>> = data[j * nx..(j + 1) * nx]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft_x.process(&mut row);
        row_result[j * nxc..(j + 1) * nxc].copy_from_slice(&row[..nxc]);
    }

    // Step 2: FFT along columns
    let mut output = row_result.clone();
    let mut col = vec![Complex::new(0.0f32, 0.0); ny];
    for i in 0..nxc {
        for j in 0..ny {
            col[j] = output[j * nxc + i];
        }
        fft_y.process(&mut col);
        for j in 0..ny {
            output[j * nxc + i] = col[j];
        }
    }

    output
}

/// Compute 2D inverse FFT. Input is nxc x ny complex (nxc = nx/2+1).
/// Returns real array nx x ny.
pub fn fft_c2r_2d(data: &[Complex<f32>], nx: usize, ny: usize) -> Vec<f32> {
    let nxc = nx / 2 + 1;
    let mut planner = FftPlanner::new();
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_x = planner.plan_fft_inverse(nx);

    // Step 1: IFFT along columns
    let mut work = data.to_vec();
    let mut col = vec![Complex::new(0.0f32, 0.0); ny];
    for i in 0..nxc {
        for j in 0..ny {
            col[j] = work[j * nxc + i];
        }
        ifft_y.process(&mut col);
        for j in 0..ny {
            work[j * nxc + i] = col[j];
        }
    }

    // Step 2: IFFT along rows (expand to full nx)
    let scale = 1.0 / (nx * ny) as f32;
    let mut output = vec![0.0f32; nx * ny];
    for j in 0..ny {
        let mut row = vec![Complex::new(0.0f32, 0.0); nx];
        for i in 0..nxc {
            row[i] = work[j * nxc + i];
        }
        // Mirror negative frequencies
        for i in 1..nx / 2 {
            row[nx - i] = row[i].conj();
        }
        ifft_x.process(&mut row);
        for i in 0..nx {
            output[j * nx + i] = row[i].re * scale;
        }
    }

    output
}

/// Compute the power spectrum (|F|^2) from complex FFT output.
pub fn power_spectrum(fft_data: &[Complex<f32>]) -> Vec<f32> {
    fft_data.iter().map(|c| c.norm_sqr()).collect()
}

/// Cross-correlation of two real images using FFT.
/// Both images must be the same size (nx x ny).
pub fn cross_correlate_2d(a: &[f32], b: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let fa = fft_r2c_2d(a, nx, ny);
    let fb = fft_r2c_2d(b, nx, ny);

    // Multiply FA * conj(FB)
    let product: Vec<Complex<f32>> = fa
        .iter()
        .zip(fb.iter())
        .map(|(a, b)| a * b.conj())
        .collect();

    fft_c2r_2d(&product, nx, ny)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_1d_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = input.len();
        let freq = fft_r2c_1d(&input);
        let output = fft_c2r_1d(&freq, n);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn fft_2d_roundtrip() {
        let nx = 8;
        let ny = 8;
        let input: Vec<f32> = (0..nx * ny).map(|i| (i as f32).sin()).collect();
        let freq = fft_r2c_2d(&input, nx, ny);
        let output = fft_c2r_2d(&freq, nx, ny);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn fft_dc_component() {
        let input = vec![3.0; 16];
        let freq = fft_r2c_1d(&input);
        // DC component should be sum of inputs = 48
        assert!((freq[0].re - 48.0).abs() < 1e-4);
        // All other components should be ~0
        for c in &freq[1..] {
            assert!(c.norm() < 1e-4);
        }
    }

    #[test]
    fn cross_correlation_peak() {
        let nx = 16;
        let ny = 16;
        // Create a simple image with a peak
        let mut a = vec![0.0f32; nx * ny];
        a[0] = 1.0; // delta at origin
        let b = a.clone();
        let cc = cross_correlate_2d(&a, &b, nx, ny);
        // Auto-correlation of delta should peak at origin
        let max_val = cc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((cc[0] - max_val).abs() < 1e-4);
    }

    #[test]
    fn power_spectrum_test() {
        let input = vec![1.0, 0.0, -1.0, 0.0];
        let freq = fft_r2c_1d(&input);
        let ps = power_spectrum(&freq);
        // DC should be 0 (sum = 0)
        assert!(ps[0] < 1e-6);
    }
}
