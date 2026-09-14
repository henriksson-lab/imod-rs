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
use crate::imod::libcfshr::taperpad::{slice_taper_in_pad, slice_taper_out_pad};
use crate::imod::libfft::todfft;
use crate::imod::libiimod::mrcfiles::mrc_get_complex_scale;

/// `sliceByteBinnedFFT` (`xcorr.cpp`).
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
    if sin.mode != SLICE_MODE_BYTE || binning <= 0 {
        return -1.;
    }
    let nxin = sin.xsize;
    let nyin = sin.ysize;
    if nxin <= 0 || nyin <= 0 || sin.data.len() != (nxin * nyin) as usize {
        return -1.;
    }
    let mut nx_dim = nxin;
    let mut binned = Vec::new();
    if binning > 1 {
        ix0 /= binning;
        ix1 /= binning;
        iy0 /= binning;
        iy1 /= binning;
        nx_dim /= binning;
        binned.resize((nx_dim * (nyin / binning)) as usize, 0);
        let ix_offset = (nxin % binning) / 2;
        let iy_offset = (nyin % binning) / 2;
        for iy in 0..nyin / binning {
            for ix in 0..nxin / binning {
                let mut sum = 0_i32;
                for by in 0..binning {
                    for bx in 0..binning {
                        sum += sin.data[(ix * binning
                            + ix_offset
                            + bx
                            + (iy * binning + iy_offset + by) * nxin)
                            as usize] as i32;
                    }
                }
                binned[(ix + iy * nx_dim) as usize] = (sum / (binning * binning)) as u8;
            }
        }
    }
    let nx_proc = ix1 + 1 - ix0;
    let ny_proc = iy1 + 1 - iy0;
    let square_size = nx_proc.max(ny_proc);
    let n_taper = (0.02 * square_size as f32) as i32;
    let n_pad = (0.02 * square_size as f32) as i32;
    let n_pad_size = nice_frame(square_size + n_pad, 2, 19);
    let n_pad_pix = (n_pad_size + 2) * n_pad_size;
    let mut fft_array = vec![0.; n_pad_pix as usize];
    let input = if binning > 1 { &binned } else { &sin.data };
    if ix0 < 0 || iy0 < 0 || ix1 < ix0 || iy1 < iy0 || ix1 >= nx_dim || iy1 >= nyin / binning {
        return -1.;
    }
    slice_taper_in_pad(
        crate::imod::libcfshr::taperpad::PadIn::Byte(input),
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
    todfft(&mut fft_array, n_pad_size, n_pad_size, 0);
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
    let scale = (sin.max - sin.min) as f64 / (log_scale * max + 1.).ln();
    let fill_val = (scale * (log_scale * sum / n_pad_pix as f64 + 1.).ln() + sin.min as f64) as u8;
    let ixcen = binning * (ix1 + 1 + ix0) / 2;
    let iycen = binning * (iy1 + 1 + iy0) / 2;
    *xcen = ixcen;
    *ycen = iycen;
    for iyout in 0..nyin {
        let mut iyin = iyout - iycen;
        if iyin < -n_pad_size / 2 || iyin >= n_pad_size / 2 {
            for i in 0..nxin {
                sin.data[(i + iyout * nxin) as usize] = fill_val;
            }
            continue;
        }
        if iyin < 0 {
            iyin += n_pad_size;
        }
        let mut ixbase = iyin * (n_pad_size + 2);
        let ixnd = (ixcen + n_pad_size / 2).min(nxin - 1);
        let ncopy = 2 * (ixnd + 1 - ixcen);
        let mut output = iyout * nxin + ixcen;
        for i in (ixbase..ixbase + ncopy).step_by(2) {
            let Some(dest) = sin.data.get_mut(output as usize) else {
                return -1.;
            };
            *dest = (scale * (log_scale * fft_array[i as usize] as f64 + 1.).ln() + sin.min as f64)
                as u8;
            output += 1;
        }
        for _ in ixcen + n_pad_size / 2 + 1..nxin {
            let Some(dest) = sin.data.get_mut(output as usize) else {
                return -1.;
            };
            *dest = fill_val;
            output += 1;
        }
        if iyin != 0 {
            iyin = n_pad_size - iyin;
            if iyin == n_pad_size / 2 {
                iyin -= 1;
            }
        }
        ixbase = iyin * (n_pad_size + 2) + 2;
        output = iyout * nxin + ixcen - 1;
        let ixnd = (ixcen - n_pad_size / 2).max(0);
        let ncopy = 2 * (ixcen - ixnd);
        for i in (ixbase..ixbase + ncopy).step_by(2) {
            let Some(dest) = sin.data.get_mut(output as usize) else {
                return -1.;
            };
            *dest = (scale * (log_scale * fft_array[i as usize] as f64 + 1.).ln() + sin.min as f64)
                as u8;
            output -= 1;
        }
        for _ in 0..ixcen - n_pad_size / 2 {
            let Some(dest) = sin.data.get_mut(output as usize) else {
                return -1.;
            };
            *dest = fill_val;
            output -= 1;
        }
    }
    1. / n_pad_size as f32
}

/// `sliceFourierFilter` (`xcorr.cpp`).
pub fn slice_fourier_filter(
    sin: &mut Islice,
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
) -> i32 {
    let nx = sin.xsize;
    let ny = sin.ysize;
    if nx <= 0 || ny <= 0 {
        return -1;
    }
    let padx = ((0.05 * nx as f32) as i32).max(8);
    let pady = ((0.05 * ny as f32) as i32).max(8);
    let nxpad = nice_frame(nx + padx, 2, 19);
    let nypad = nice_frame(ny + pady, 2, 19);
    let mut brray = vec![0.; ((nxpad + 2) * nypad) as usize];
    let mut ctf = [0.; 8193];
    let mut delta = 0.;
    xcorr_set_ctf(
        sigma1, sigma2, radius1, radius2, &mut ctf, nxpad, nypad, &mut delta,
    );
    match sin.mode {
        SLICE_MODE_BYTE => {
            if sin.data.len() != (nx * ny) as usize {
                return -1;
            }
            slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Byte(&sin.data),
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
        }
        SLICE_MODE_SHORT => {
            if sin.data.len() != (nx * ny * 2) as usize {
                return -1;
            }
            let values: Vec<i16> = sin
                .data
                .chunks_exact(2)
                .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Short(&values),
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
        }
        SLICE_MODE_USHORT => {
            if sin.data.len() != (nx * ny * 2) as usize {
                return -1;
            }
            let values: Vec<u16> = sin
                .data
                .chunks_exact(2)
                .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::UShort(&values),
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
        }
        SLICE_MODE_FLOAT => {
            if sin.data.len() != (nx * ny * 4) as usize {
                return -1;
            }
            let values: Vec<f32> = sin
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Float(&values),
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
        }
        _ => return -1,
    }
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
    let (out_min, out_max) = if sin.min == 0. && sin.max == 0. {
        match sin.mode {
            SLICE_MODE_BYTE => (0.01, 255.),
            SLICE_MODE_USHORT => (0.01, 30000.),
            _ => (sin.min, sin.max),
        }
    } else {
        (sin.min, sin.max)
    };
    xcorr_extract_convert(
        &brray,
        nxpad + 2,
        (nxpad - nx) / 2,
        (nypad - ny) / 2,
        sin,
        sin.mode,
        nx,
        ny,
        out_min,
        out_max,
    )
}

/// `XCorrExtractConvert` (`xcorr.cpp`).
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
) -> i32 {
    if nx < 0 || ny < 0 || nxdim <= 0 || ixlo < 0 || iylo < 0 {
        return -1;
    }
    let Some(pixels) = usize::try_from(nx)
        .ok()
        .and_then(|nx| usize::try_from(ny).ok().and_then(|ny| nx.checked_mul(ny)))
    else {
        return -1;
    };
    let Some(last) = usize::try_from(ixlo + nx - 1).ok().and_then(|ix| {
        usize::try_from(iylo + ny - 1)
            .ok()
            .and_then(|iy| iy.checked_mul(nxdim as usize)?.checked_add(ix))
    }) else {
        return -1;
    };
    if last >= array.len() {
        return -1;
    }
    let mut scale = 0.;
    let mut base = 0.;
    if out_min != 0. || out_max != 0. {
        let mut min = 1.0e30_f32;
        let mut max = -min;
        for iy in 0..ny {
            for ix in 0..nx {
                let value = array[(ixlo + ix + (iy + iylo) * nxdim) as usize];
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
        match typ {
            SLICE_MODE_BYTE => {
                if brray.data.len() != pixels {
                    return -1;
                }
                for ix in 0..nx {
                    let value = array[(ixlo + ix + (iy + iylo) * nxdim) as usize];
                    brray.data[(iy * nx + ix) as usize] = if scale != 0. {
                        (value * scale + base) as u8
                    } else {
                        value as u8
                    };
                }
            }
            SLICE_MODE_SHORT => {
                if brray.data.len() != pixels * 2 {
                    return -1;
                }
                for ix in 0..nx {
                    let value = array[(ixlo + ix + (iy + iylo) * nxdim) as usize];
                    let output = if scale != 0. {
                        (value * scale + base) as i16
                    } else {
                        value as i16
                    };
                    brray.data[(iy * nx + ix) as usize * 2..][..2]
                        .copy_from_slice(&output.to_ne_bytes());
                }
            }
            SLICE_MODE_USHORT => {
                if brray.data.len() != pixels * 2 {
                    return -1;
                }
                for ix in 0..nx {
                    let value = array[(ixlo + ix + (iy + iylo) * nxdim) as usize];
                    let output = if scale != 0. {
                        (value * scale + base) as u16
                    } else {
                        value as u16
                    };
                    brray.data[(iy * nx + ix) as usize * 2..][..2]
                        .copy_from_slice(&output.to_ne_bytes());
                }
            }
            SLICE_MODE_FLOAT => {
                if brray.data.len() != pixels * 4 {
                    return -1;
                }
                for ix in 0..nx {
                    let value = array[(ixlo + ix + (iy + iylo) * nxdim) as usize];
                    // This inverted branch is intentional: it is exactly the C source.
                    let output = if scale != 0. {
                        value
                    } else {
                        value * scale + base
                    };
                    brray.data[(iy * nx + ix) as usize * 4..][..4]
                        .copy_from_slice(&output.to_ne_bytes());
                }
            }
            _ => {}
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_convert_scales_byte_subarea() {
        let source = [0., 1., 2., 3., 4., 5., 6., 7., 8.];
        let mut output = Islice {
            data: vec![0; 4],
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
        assert_eq!(
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
                250.
            ),
            0
        );
        assert_eq!(output.data, [10, 70, 190, 250]);
    }

    #[test]
    fn extract_convert_keeps_source_float_branch() {
        let source = [2., 4.];
        let mut output = Islice {
            data: vec![0; 8],
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
        assert_eq!(
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
                9.
            ),
            0
        );
        let values: Vec<f32> = output
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, source);
    }

    #[test]
    fn fourier_filter_processes_a_byte_slice_in_place() {
        let data: Vec<u8> = (0..64).map(|value| (value * 4) as u8).collect();
        let mut slice = Islice {
            data,
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
        assert!(slice.data.iter().any(|&value| value != 0));
    }
}
