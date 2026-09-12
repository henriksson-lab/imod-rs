//! Translation of `IMOD/3dmod/xcorr.cpp` and `xcorr.h`.
//!
//! The routines retain the source raw `Islice` and padded FFT buffer contracts;
//! viewer calls remain outside this lower correlation unit.

#![allow(dead_code)]

use core::ffi::c_void;

use crate::imod::libcfshr::filtxcorr::{nice_frame, xcorr_filter_part, xcorr_set_ctf};
use crate::imod::libcfshr::islice::Islice;
use crate::imod::libcfshr::reduce_by_binning::{
    SLICE_MODE_BYTE, SLICE_MODE_FLOAT, SLICE_MODE_SHORT, SLICE_MODE_USHORT,
};
use crate::imod::libcfshr::taperpad::{slice_taper_in_pad, slice_taper_out_pad};
use crate::imod::libfft::todfft;
use crate::imod::libiimod::mrcfiles::mrc_get_complex_scale;

/// `sliceByteBinnedFFT` (`xcorr.cpp`).
pub unsafe fn slice_byte_binned_fft(
    sin: *mut Islice,
    binning: i32,
    mut ix0: i32,
    mut ix1: i32,
    mut iy0: i32,
    mut iy1: i32,
    xcen: *mut i32,
    ycen: *mut i32,
) -> f32 {
    unsafe {
        if (*sin).mode != SLICE_MODE_BYTE {
            return -1.;
        }
        let nxin = (*sin).xsize;
        let nyin = (*sin).ysize;
        let mut nx_dim = nxin;
        let mut input = (*sin).data.b;
        let mut binned = Vec::new();
        if binning > 1 {
            ix0 /= binning;
            ix1 /= binning;
            iy0 /= binning;
            iy1 /= binning;
            nx_dim /= binning;
            binned.resize((nx_dim * (nyin / binning)) as usize, 0);
            crate::imod::three_dmod::imodview::ivw_bin_by_n(
                core::slice::from_raw_parts((*sin).data.b, (nxin * nyin) as usize),
                nxin,
                nyin,
                binning,
                &mut binned,
            );
            input = binned.as_mut_ptr();
        }
        let nx_proc = ix1 + 1 - ix0;
        let ny_proc = iy1 + 1 - iy0;
        let square_size = nx_proc.max(ny_proc);
        let n_taper = (0.02 * square_size as f32) as i32;
        let n_pad = (0.02 * square_size as f32) as i32;
        let mut n_pad_size = nice_frame(square_size + n_pad, 2, 19);
        let n_pad_pix = (n_pad_size + 2) * n_pad_size;
        let mut fft_array = vec![0.; n_pad_pix as usize];
        slice_taper_in_pad(
            input.cast::<c_void>(),
            SLICE_MODE_BYTE,
            nx_dim,
            ix0,
            ix1,
            iy0,
            iy1,
            fft_array.as_mut_ptr(),
            n_pad_size + 2,
            n_pad_size,
            n_pad_size,
            n_taper,
            n_taper,
        );
        let mut idir = 0;
        todfft(
            fft_array.as_mut_ptr(),
            &mut n_pad_size,
            &mut n_pad_size,
            &mut idir,
        );
        let mut max = 0_f64;
        let mut max2 = 0_f64;
        let mut sum = 0_f64;
        let mut max_index = 0_usize;
        for index in (0..n_pad_pix as usize).step_by(2) {
            let value = ((fft_array[index] * fft_array[index]
                + fft_array[index + 1] * fft_array[index + 1]) as f64)
                .sqrt();
            if max < value {
                max2 = max;
                max = value;
                max_index = index;
            }
            sum += value;
            fft_array[index] = value as f32;
        }
        max = max2 + 0.1 * (max - max2);
        fft_array[max_index] = max as f32;
        let log_scale = mrc_get_complex_scale() as f64;
        let scale = ((*sin).max - (*sin).min) as f64 / (log_scale * max + 1.).ln();
        let fill_val =
            (scale * (log_scale * sum / n_pad_pix as f64 + 1.).ln() + (*sin).min as f64) as u8;
        let ixcen = binning * (ix1 + 1 + ix0) / 2;
        let iycen = binning * (iy1 + 1 + iy0) / 2;
        *xcen = ixcen;
        *ycen = iycen;
        for iyout in 0..nyin {
            let mut iyin = iyout - iycen;
            if iyin < -n_pad_size / 2 || iyin >= n_pad_size / 2 {
                for i in 0..nxin {
                    *(*sin).data.b.add((i + iyout * nxin) as usize) = fill_val;
                }
                continue;
            }
            if iyin < 0 {
                iyin += n_pad_size;
            }
            let mut ixbase = iyin * (n_pad_size + 2);
            let ixnd = (ixcen + n_pad_size / 2).min(nxin - 1);
            let ncopy = 2 * (ixnd + 1 - ixcen);
            let mut output = (*sin).data.b.add((iyout * nxin + ixcen) as usize);
            for i in (ixbase..ixbase + ncopy).step_by(2) {
                *output = (scale * (log_scale * fft_array[i as usize] as f64 + 1.).ln()
                    + (*sin).min as f64) as u8;
                output = output.add(1);
            }
            for _ in ixcen + n_pad_size / 2 + 1..nxin {
                *output = fill_val;
                output = output.add(1);
            }
            if iyin != 0 {
                iyin = n_pad_size - iyin;
                if iyin == n_pad_size / 2 {
                    iyin -= 1;
                }
            }
            ixbase = iyin * (n_pad_size + 2) + 2;
            output = (*sin).data.b.add((iyout * nxin + ixcen - 1) as usize);
            let ixnd = (ixcen - n_pad_size / 2).max(0);
            let ncopy = 2 * (ixcen - ixnd);
            for i in (ixbase..ixbase + ncopy).step_by(2) {
                *output = (scale * (log_scale * fft_array[i as usize] as f64 + 1.).ln()
                    + (*sin).min as f64) as u8;
                output = output.sub(1);
            }
            for _ in 0..ixcen - n_pad_size / 2 {
                *output = fill_val;
                output = output.sub(1);
            }
        }
        1. / n_pad_size as f32
    }
}

/// `sliceFourierFilter` (`xcorr.cpp`).
pub unsafe fn slice_fourier_filter(
    sin: *mut Islice,
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
) -> i32 {
    unsafe {
        let nx = (*sin).xsize;
        let ny = (*sin).ysize;
        let padx = ((0.05 * nx as f32) as i32).max(8);
        let pady = ((0.05 * ny as f32) as i32).max(8);
        let mut nxpad = nice_frame(nx + padx, 2, 19);
        let mut nypad = nice_frame(ny + pady, 2, 19);
        let mut brray = vec![0.; ((nxpad + 2) * nypad) as usize];
        let mut ctf = [0.; 8193];
        let mut delta = 0.;
        xcorr_set_ctf(
            sigma1,
            sigma2,
            radius1,
            radius2,
            ctf.as_mut_ptr(),
            nxpad,
            nypad,
            &mut delta,
        );
        slice_taper_out_pad(
            (*sin).data.b.cast::<c_void>(),
            (*sin).mode,
            nx,
            ny,
            brray.as_mut_ptr(),
            nxpad + 2,
            nxpad,
            nypad,
            0,
            0.,
        );
        let mut idir = 0;
        todfft(brray.as_mut_ptr(), &mut nxpad, &mut nypad, &mut idir);
        xcorr_filter_part(
            brray.as_ptr(),
            brray.as_mut_ptr(),
            nxpad,
            nypad,
            ctf.as_ptr(),
            delta,
        );
        idir = 1;
        todfft(brray.as_mut_ptr(), &mut nxpad, &mut nypad, &mut idir);
        let (out_min, out_max) = if (*sin).min == 0. && (*sin).max == 0. {
            match (*sin).mode {
                SLICE_MODE_BYTE => (0.01, 255.),
                SLICE_MODE_USHORT => (0.01, 30000.),
                _ => ((*sin).min, (*sin).max),
            }
        } else {
            ((*sin).min, (*sin).max)
        };
        xcorr_extract_convert(
            brray.as_mut_ptr(),
            nxpad + 2,
            (nxpad - nx) / 2,
            (nypad - ny) / 2,
            (*sin).data.b.cast::<c_void>(),
            (*sin).mode,
            nx,
            ny,
            out_min,
            out_max,
        );
        0
    }
}

/// `XCorrExtractConvert` (`xcorr.cpp`).
pub unsafe fn xcorr_extract_convert(
    array: *mut f32,
    nxdim: i32,
    ixlo: i32,
    iylo: i32,
    brray: *mut c_void,
    typ: i32,
    nx: i32,
    ny: i32,
    out_min: f32,
    out_max: f32,
) {
    unsafe {
        let mut scale = 0.;
        let mut base = 0.;
        if out_min != 0. || out_max != 0. {
            let mut min = 1.0e30_f32;
            let mut max = -min;
            for iy in 0..ny {
                for ix in 0..nx {
                    let value = *array.add((ixlo + ix + (iy + iylo) * nxdim) as usize);
                    min = min.min(value);
                    max = max.max(value);
                }
            }
            if max == min {
                max += 1.;
            }
            scale = (out_max - out_min) / (max - min);
            base = out_min - scale * min;
        }
        for iy in 0..ny {
            let input = array.add((ixlo + (iy + iylo) * nxdim) as usize);
            match typ {
                SLICE_MODE_BYTE => {
                    for ix in 0..nx {
                        *brray.cast::<u8>().add((iy * nx + ix) as usize) = if scale != 0. {
                            (*input.add(ix as usize) * scale + base) as u8
                        } else {
                            *input.add(ix as usize) as u8
                        };
                    }
                }
                SLICE_MODE_SHORT => {
                    for ix in 0..nx {
                        *brray.cast::<i16>().add((iy * nx + ix) as usize) = if scale != 0. {
                            (*input.add(ix as usize) * scale + base) as i16
                        } else {
                            *input.add(ix as usize) as i16
                        };
                    }
                }
                SLICE_MODE_USHORT => {
                    for ix in 0..nx {
                        *brray.cast::<u16>().add((iy * nx + ix) as usize) = if scale != 0. {
                            (*input.add(ix as usize) * scale + base) as u16
                        } else {
                            *input.add(ix as usize) as u16
                        };
                    }
                }
                SLICE_MODE_FLOAT => {
                    for ix in 0..nx {
                        // This inverted branch is intentional: it is exactly the C source.
                        *brray.cast::<f32>().add((iy * nx + ix) as usize) = if scale != 0. {
                            *input.add(ix as usize)
                        } else {
                            *input.add(ix as usize) * scale + base
                        };
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::islice::MrcData;

    #[test]
    fn extract_convert_scales_byte_subarea() {
        let mut source = vec![0., 1., 2., 3., 4., 5., 6., 7., 8.];
        let mut output = [0_u8; 4];
        unsafe {
            xcorr_extract_convert(
                source.as_mut_ptr(),
                3,
                1,
                1,
                output.as_mut_ptr().cast(),
                SLICE_MODE_BYTE,
                2,
                2,
                10.,
                250.,
            );
        }
        assert_eq!(output, [10, 70, 190, 250]);
    }

    #[test]
    fn extract_convert_keeps_source_float_branch() {
        let mut source = [2., 4.];
        let mut output = [0.; 2];
        unsafe {
            xcorr_extract_convert(
                source.as_mut_ptr(),
                2,
                0,
                0,
                output.as_mut_ptr().cast(),
                SLICE_MODE_FLOAT,
                2,
                1,
                5.,
                9.,
            );
        }
        assert_eq!(output, source);
    }

    #[test]
    fn fourier_filter_processes_a_byte_slice_in_place() {
        let mut data: Vec<u8> = (0..64).map(|value| (value * 4) as u8).collect();
        let mut slice = Islice {
            data: MrcData {
                b: data.as_mut_ptr(),
            },
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
        unsafe {
            assert_eq!(slice_fourier_filter(&mut slice, 0.15, 0.1, 0.02, 0.4), 0);
        }
        assert!(data.iter().any(|&value| value != 0));
    }
}
