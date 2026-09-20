//! Translation of `IMOD/3dmod/xcorr.cpp` and `xcorr.h`.
//!
//! The routines operate on owned `Islice` storage and padded FFT vectors;
//! viewer calls remain outside this lower correlation unit.

#![allow(dead_code)]

use crate::imod::libcfshr::filtxcorr::{nice_frame, xcorr_filter_part, xcorr_set_ctf};
use crate::imod::libcfshr::islice::Islice;
use crate::imod::libcfshr::reduce_by_binning::{
    SLICE_MODE_BYTE, SLICE_MODE_FLOAT, SLICE_MODE_SHORT, SLICE_MODE_USHORT,
};
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad, slice_taper_out_pad};
use crate::imod::libfft::todfft;
use crate::imod::libiimod::mrcfiles::mrc_get_complex_scale;
use crate::imod::three_dmod::imodview::ivw_bin_by_n;

/// `sliceByteBinnedFFT` (`xcorr.cpp:26`).
pub fn slice_byte_binned_fft(
    sin: &mut Islice,
    binning: i32,
    mut ix0: i32,
    mut ix1: i32,
    mut iy0: i32,
    mut iy1: i32,
    xcen: &mut i32,
    ycen: &mut i32,
) -> f32 {
    let nxin = sin.xsize;
    let nyin = sin.ysize;
    let log_scale = mrc_get_complex_scale() as f64;

    if sin.mode != SLICE_MODE_BYTE {
        return -1.;
    }

    let mut nx_dim = sin.xsize;

    // If binning, reduce size, get array and bin into it
    let mut binned = Vec::new();
    if binning > 1 {
        ix0 /= binning;
        ix1 /= binning;
        iy0 /= binning;
        iy1 /= binning;
        nx_dim /= binning;
        binned = vec![0_u8; (nx_dim * (sin.ysize / binning)) as usize];
        ivw_bin_by_n(sin.data.b(), sin.xsize, sin.ysize, binning, &mut binned);
    }

    // Figure out size of square array and the padding and tapering
    let nx_proc = ix1 + 1 - ix0;
    let ny_proc = iy1 + 1 - iy0;

    let square_size = nx_proc.max(ny_proc);
    let n_taper = (0.02_f32 * square_size as f32) as i32;
    let n_pad = (0.02_f32 * square_size as f32) as i32;
    let n_pad_size = nice_frame(square_size + n_pad, 2, 19);
    let n_pad_pix = (n_pad_size + 2) * n_pad_size;

    // Get array for FFT
    let mut fft_array = vec![0_f32; n_pad_pix as usize];

    // Put into array, take FFT
    slice_taper_in_pad(
        PadIn::Byte(if binning > 1 { &binned } else { sin.data.b() }),
        SLICE_MODE_BYTE,
        nx_dim,
        ix0,
        ix1,
        iy0,
        iy1,
        &mut fft_array,
        n_pad_size + 2,
        n_pad_size,
        n_pad_size,
        n_taper,
        n_taper,
    );
    drop(binned);

    todfft(&mut fft_array, n_pad_size, n_pad_size, 0);

    // Find top two magnitudes while scaling real parts to magnitudes
    let mut max = 0_f64;
    let mut max2 = 0_f64;
    let mut sum = 0_f64;
    let mut iyin = 0_usize;
    for i in (0..n_pad_pix as usize).step_by(2) {
        let val =
            ((fft_array[i] * fft_array[i] + fft_array[i + 1] * fft_array[i + 1]) as f64).sqrt();
        if max < val {
            max2 = max;
            max = val as f32 as f64;
            iyin = i;
        }
        sum += val;
        fft_array[i] = val as f32;
    }

    // Make the maximum 10% of way from second highest to highest and replace max
    max = max2 + 0.1 * (max - max2);
    fft_array[iyin] = max as f32;
    let scale = ((sin.max - sin.min) as f64 / (log_scale * max + 1.).ln()) as f32;
    let sin_min = sin.min;
    let fill_val =
        (scale as f64 * (log_scale * sum / n_pad_pix as f64 + 1.).ln() + sin_min as f64) as u8;

    // Loop on output lines
    let ixcen = binning * (ix1 + 1 + ix0) / 2;
    *xcen = ixcen;
    let iycen = binning * (iy1 + 1 + iy0) / 2;
    *ycen = iycen;
    let data = sin.data.b_mut();
    for iyout in 0..nyin {
        let mut iyin = iyout - iycen;

        // If line is out of range, fill it
        if iyin < -n_pad_size / 2 || iyin >= n_pad_size / 2 {
            let mut indata = iyout * nxin;
            for _ in 0..nxin {
                data[indata as usize] = fill_val;
                indata += 1;
            }
        } else {
            // Wrap if negative
            if iyin < 0 {
                iyin += n_pad_size;
            }

            // Copy as many values as fit or exist
            let mut ixbase = iyin * (n_pad_size + 2);
            let mut indata = iyout * nxin + ixcen;
            let mut ixnd = (ixcen + n_pad_size / 2).min(nxin - 1);
            let mut ncopy = 2 * (ixnd + 1 - ixcen);
            for i in (ixbase..ixbase + ncopy).step_by(2) {
                data[indata as usize] = (scale as f64
                    * (log_scale * fft_array[i as usize] as f64 + 1.).ln()
                    + sin_min as f64) as u8;
                indata += 1;
            }

            // Fill the right side
            for _ in ixcen + n_pad_size / 2 + 1..nxin {
                data[indata as usize] = fill_val;
                indata += 1;
            }

            // Get Y line to mirror from, copy data in reverse
            if iyin != 0 {
                iyin = n_pad_size - iyin;
                if iyin == n_pad_size / 2 {
                    iyin -= 1;
                }
            }
            ixbase = iyin * (n_pad_size + 2) + 2;
            indata = iyout * nxin + ixcen - 1;
            ixnd = (ixcen - n_pad_size / 2).max(0);
            ncopy = 2 * (ixcen - ixnd);
            for i in (ixbase..ixbase + ncopy).step_by(2) {
                data[indata as usize] = (scale as f64
                    * (log_scale * fft_array[i as usize] as f64 + 1.).ln()
                    + sin_min as f64) as u8;
                indata -= 1;
            }

            // Fill left side
            for _ in 0..ixcen - n_pad_size / 2 {
                data[indata as usize] = fill_val;
                indata -= 1;
            }
        }
    }

    (1. / n_pad_size as f64) as f32
}

/// `sliceFourierFilter` (`xcorr.cpp:170`).
pub fn slice_fourier_filter(
    sin: &mut Islice,
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
) -> i32 {
    let nx = sin.xsize;
    let ny = sin.ysize;
    let mut ctf = [0_f32; 8193];
    let mut delta = 0.;

    let mut padx = (0.05 * nx as f64) as i32;
    if padx < 8 {
        padx = 8;
    }
    let mut pady = (0.05 * ny as f64) as i32;
    if pady < 8 {
        pady = 8;
    }
    let nxpad = nice_frame(nx + padx, 2, 19);
    let nypad = nice_frame(ny + pady, 2, 19);
    let mut brray = vec![0_f32; ((nxpad + 2) * nypad) as usize];

    xcorr_set_ctf(
        sigma1, sigma2, radius1, radius2, &mut ctf, nxpad, nypad, &mut delta,
    );
    // `(void *)sin->data.b`: the member `sliceTaperOutPad` reads is chosen by
    // `sin->mode`, so the typed view is picked here.
    let array = match sin.mode {
        SLICE_MODE_BYTE => PadIn::Byte(sin.data.b()),
        SLICE_MODE_SHORT => PadIn::Short(sin.data.s()),
        SLICE_MODE_USHORT => PadIn::UShort(sin.data.us()),
        SLICE_MODE_FLOAT => PadIn::Float(sin.data.f()),
        _ => return -1,
    };
    slice_taper_out_pad(
        array,
        sin.mode,
        nx,
        ny,
        &mut brray,
        nxpad + 2,
        nxpad,
        nypad,
        0,
        0.,
    );

    todfft(&mut brray, nxpad, nypad, 0);
    xcorr_filter_part(
        crate::imod::libcfshr::filtxcorr::FilterIn::InPlace,
        &mut brray,
        nxpad,
        nypad,
        &ctf,
        delta,
    );
    todfft(&mut brray, nxpad, nypad, 1);

    let ixlo = (nxpad - nx) / 2;
    let iylo = (nypad - ny) / 2;
    let mut out_min = sin.min;
    let mut out_max = sin.max;
    if !(out_min != 0. || out_max != 0.) {
        if sin.mode == SLICE_MODE_BYTE {
            out_min = 0.01;
            out_max = 255.;
        } else if sin.mode == SLICE_MODE_USHORT {
            out_min = 0.01;
            out_max = 30000.;
        }
    }
    let mode = sin.mode;
    xcorr_extract_convert(
        &brray,
        nxpad + 2,
        ixlo,
        iylo,
        sin,
        mode,
        nx,
        ny,
        out_min,
        out_max,
    );
    0
}

/// `XCorrExtractConvert` (`xcorr.cpp:235`).  The C takes `void *brray` and
/// reads it through the member `type` selects; here it takes the slice.
pub fn xcorr_extract_convert(
    array: &[f32],
    nxdim: i32,
    ixlo: i32,
    iylo: i32,
    brray: &mut Islice,
    typ: i32,
    nx: i32,
    ny: i32,
    out_min: f32,
    out_max: f32,
) {
    let mut scale = 0_f32;
    let mut base = 0_f32;

    /* Need to scale data?  */
    if out_min != 0. || out_max != 0. {
        let mut min = 1.0e30_f32;
        let mut max = -min;
        for iy in 0..ny {
            let mut inp = (ixlo + (iy + iylo) * nxdim) as usize;
            for _ in 0..nx {
                if min > array[inp] {
                    min = array[inp];
                }
                if max < array[inp] {
                    max = array[inp];
                }
                inp += 1;
            }
        }
        if max == min {
            max += 1.;
        }
        scale = (out_max - out_min) / (max - min);
        base = out_min - scale * min;
    }

    for iy in 0..ny {
        let mut inp = (ixlo + (iy + iylo) * nxdim) as usize;

        match typ {
            SLICE_MODE_BYTE => {
                let byteout = brray.data.b_mut();
                let mut out = (iy * nx) as usize;
                if scale != 0. {
                    for _ in 0..nx {
                        byteout[out] = (array[inp] * scale + base) as u8;
                        inp += 1;
                        out += 1;
                    }
                } else {
                    for _ in 0..nx {
                        byteout[out] = array[inp] as u8;
                        inp += 1;
                        out += 1;
                    }
                }
            }

            SLICE_MODE_SHORT => {
                let intout = brray.data.s_mut();
                let mut out = (iy * nx) as usize;
                if scale != 0. {
                    for _ in 0..nx {
                        intout[out] = (array[inp] * scale + base) as i16;
                        inp += 1;
                        out += 1;
                    }
                } else {
                    for _ in 0..nx {
                        intout[out] = array[inp] as i16;
                        inp += 1;
                        out += 1;
                    }
                }
            }

            SLICE_MODE_USHORT => {
                let uintout = brray.data.us_mut();
                let mut out = (iy * nx) as usize;
                if scale != 0. {
                    for _ in 0..nx {
                        uintout[out] = (array[inp] * scale + base) as u16;
                        inp += 1;
                        out += 1;
                    }
                } else {
                    for _ in 0..nx {
                        uintout[out] = array[inp] as u16;
                        inp += 1;
                        out += 1;
                    }
                }
            }

            SLICE_MODE_FLOAT => {
                let floatout = brray.data.f_mut();
                let mut out = (iy * nx) as usize;
                // This inverted branch is intentional: it is exactly the C source.
                if scale != 0. {
                    for _ in 0..nx {
                        floatout[out] = array[inp];
                        inp += 1;
                        out += 1;
                    }
                } else {
                    for _ in 0..nx {
                        floatout[out] = array[inp] * scale + base;
                        inp += 1;
                        out += 1;
                    }
                }
            }

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::islice::MrcData;

    #[test]
    fn extract_convert_scales_byte_subarea() {
        let source = [0., 1., 2., 3., 4., 5., 6., 7., 8.];
        let mut output = Islice {
            data: MrcData::B(vec![0; 4]),
            xsize: 2,
            ysize: 2,
            mode: SLICE_MODE_BYTE,
            csize: 1,
            dsize: 1,
            min: 0.,
            max: 0.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        xcorr_extract_convert(
            &source,
            3,
            1,
            1,
            &mut output,
            SLICE_MODE_BYTE,
            2,
            2,
            10.,
            250.,
        );
        assert_eq!(output.data.b(), [10, 70, 190, 250]);
    }

    #[test]
    fn extract_convert_keeps_source_float_branch() {
        let source = [2., 4.];
        let mut output = Islice {
            data: MrcData::F(vec![0.; 2]),
            xsize: 2,
            ysize: 1,
            mode: SLICE_MODE_FLOAT,
            csize: 1,
            dsize: 4,
            min: 0.,
            max: 0.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        xcorr_extract_convert(
            &source,
            2,
            0,
            0,
            &mut output,
            SLICE_MODE_FLOAT,
            2,
            1,
            5.,
            9.,
        );
        assert_eq!(output.data.f(), source);
    }

    #[test]
    fn fourier_filter_processes_a_byte_slice_in_place() {
        let data: Vec<u8> = (0..64).map(|value| (value * 4) as u8).collect();
        let mut slice = Islice {
            data: MrcData::B(data),
            xsize: 8,
            ysize: 8,
            mode: SLICE_MODE_BYTE,
            csize: 1,
            dsize: 1,
            min: 0.,
            max: 255.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        assert_eq!(slice_fourier_filter(&mut slice, 0.15, 0.1, 0.02, 0.4), 0);
        assert!(slice.data.b().iter().any(|&value| value != 0));
    }
}
