//! Translation of `IMOD/libcfshr/spectrumscaled.c`.
#![allow(dead_code)]

/// C `makeAmplitudeSpectrum` (`spectrumscaled.c:276`).
pub unsafe fn make_amplitude_spectrum(
    fft_array: *mut f32,
    spectrum: *mut f32,
    mut pad_size: i32,
    out_xdim: i32,
) {
    unsafe {
        let mut amplitudes = false;
        let mut in_xdim = pad_size + 2;
        if pad_size < 0 {
            amplitudes = true;
            pad_size = -pad_size;
            in_xdim = (pad_size + 2) / 2;
        }
        let mut iyin = pad_size / 2;
        for iyout in 0..pad_size {
            let ixbase = iyin * in_xdim;
            if amplitudes {
                for i in 0..pad_size / 2 {
                    *spectrum.add((iyout * out_xdim + pad_size / 2 + i) as usize) =
                        *fft_array.add((ixbase + i) as usize);
                }
                *spectrum.add((((pad_size - iyout) % pad_size) * out_xdim) as usize) =
                    *fft_array.add((ixbase + pad_size / 2) as usize);
            } else {
                for i in (0..pad_size).step_by(2) {
                    let real = *fft_array.add((ixbase + i) as usize);
                    let imag = *fft_array.add((ixbase + i + 1) as usize);
                    *spectrum.add((iyout * out_xdim + pad_size / 2 + i / 2) as usize) =
                        (real * real + imag * imag).sqrt();
                }
                let real = *fft_array.add((ixbase + pad_size) as usize);
                let imag = *fft_array.add((ixbase + pad_size + 1) as usize);
                *spectrum.add((((pad_size - iyout) % pad_size) * out_xdim) as usize) =
                    (real * real + imag * imag).sqrt();
            }
            iyin = (iyin + 1) % pad_size;
        }
        for iyout in 0..pad_size {
            let iyin = if iyout != 0 { pad_size - iyout } else { 0 };
            for i in 0..pad_size / 2 - 1 {
                *spectrum.add((iyout * out_xdim + pad_size / 2 - 1 - i) as usize) =
                    *spectrum.add((iyin * out_xdim + pad_size / 2 + 1 + i) as usize);
            }
        }
    }
}

/// C static `FFTMagnitude`.
unsafe fn fft_magnitude(array: *mut f32, nx: i32, _ny: i32, ix: i32, iy: i32) -> f64 {
    unsafe {
        let i = (ix * 2 + iy * (nx + 2)) as usize;
        ((*array.add(i) * *array.add(i) + *array.add(i + 1) * *array.add(i + 1)) as f64).sqrt()
    }
}

/// C `spectrumScaled`.
pub unsafe fn spectrum_scaled(
    image: *mut core::ffi::c_void,
    typ: i32,
    nx: i32,
    ny: i32,
    spectrum: *mut core::ffi::c_void,
    mut pad_size: i32,
    final_size: i32,
    bkgd_gray: i32,
    trunc_diam: f32,
    filt_type: i32,
    two_d_fft: unsafe fn(*mut f32, *mut i32, *mut i32, *mut i32),
) -> i32 {
    unsafe {
        let mut taper = 0.05_f32;
        if pad_size < 0 {
            pad_size = -pad_size;
            taper = 0.;
        }
        let padx = pad_size + 2;
        let reducing = final_size < pad_size && filt_type >= 0;
        let cropping = final_size < pad_size && filt_type < 0;
        // The source selects the reduction filter before checking the general
        // argument constraints, so an invalid filter has precedence over -4.
        if reducing {
            let mut iyout = 0;
            let ret = crate::imod::libcfshr::zoomdown::select_zoom_filter(
                filt_type,
                final_size as f64 / pad_size as f64,
                &mut iyout,
            );
            if ret != 0 {
                return -ret;
            }
        }
        if final_size > pad_size
            || bkgd_gray > 192
            || trunc_diam < 0.
            || trunc_diam > 0.75
            || (bkgd_gray > 0 && filt_type < 0)
        {
            return -4;
        }
        let fft =
            libc::malloc((padx * pad_size) as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if fft.is_null() {
            return -3;
        }
        let mut temp: *mut i16 = core::ptr::null_mut();
        let mut crop: *mut f32 = core::ptr::null_mut();
        let mut lines: *mut *mut u8 = core::ptr::null_mut();
        if bkgd_gray > 0 || reducing {
            temp = libc::malloc((padx * pad_size) as usize * 2).cast();
            if temp.is_null() {
                libc::free(fft.cast());
                return -3;
            }
            if reducing {
                lines = crate::imod::libcfshr::b3dutil::make_line_pointers(
                    temp.cast(),
                    pad_size,
                    pad_size,
                    2,
                );
                if lines.is_null() {
                    libc::free(temp.cast());
                    libc::free(fft.cast());
                    return -3;
                }
            }
        }
        if cropping {
            crop = libc::malloc((padx * pad_size) as usize * 4).cast();
            if crop.is_null() {
                libc::free(fft.cast());
                return -3;
            }
        }
        crate::imod::libcfshr::taperpad::slice_taper_in_pad(
            match typ {
                0 => crate::imod::libcfshr::taperpad::PadIn::Byte(core::slice::from_raw_parts(
                    image.cast::<u8>(),
                    (nx * ny) as usize,
                )),
                1 => crate::imod::libcfshr::taperpad::PadIn::Short(core::slice::from_raw_parts(
                    image.cast::<i16>(),
                    (nx * ny) as usize,
                )),
                6 => crate::imod::libcfshr::taperpad::PadIn::UShort(core::slice::from_raw_parts(
                    image.cast::<u16>(),
                    (nx * ny) as usize,
                )),
                16 => crate::imod::libcfshr::taperpad::PadIn::Rgb(core::slice::from_raw_parts(
                    image.cast::<u8>(),
                    (3 * nx * ny) as usize,
                )),
                _ => crate::imod::libcfshr::taperpad::PadIn::Float(core::slice::from_raw_parts(
                    image.cast::<f32>(),
                    (nx * ny) as usize,
                )),
            },
            typ,
            nx,
            0,
            nx - 1,
            0,
            ny - 1,
            core::slice::from_raw_parts_mut(fft, (padx * pad_size) as usize),
            padx,
            pad_size,
            pad_size,
            (nx as f32 * taper) as i32,
            (ny as f32 * taper) as i32,
        );
        let (mut px, mut py, mut idir) = (pad_size, pad_size, 0);
        two_d_fft(fft, &mut px, &mut py, &mut idir);
        if filt_type < 0 {
            make_amplitude_spectrum(
                fft,
                if cropping { crop } else { spectrum.cast() },
                pad_size,
                padx,
            );
            let place = (padx * pad_size / 2 + pad_size / 2) as usize;
            *(if cropping {
                crop
            } else {
                spectrum.cast::<f32>()
            })
            .add(place) = 0.;
            if cropping {
                idir = 0;
                two_d_fft(crop, &mut px, &mut py, &mut idir);
                let shift = 0.5 * (pad_size as f32 / final_size as f32 - 1.);
                crate::imod::libcfshr::filtxcorr::fourier_reduce_image(
                    core::slice::from_raw_parts(crop, (padx * pad_size) as usize),
                    pad_size,
                    pad_size,
                    core::slice::from_raw_parts_mut(
                        spectrum.cast::<f32>(),
                        ((final_size + 2) * final_size) as usize,
                    ),
                    final_size,
                    final_size,
                    shift,
                    shift,
                    Some(core::slice::from_raw_parts_mut(
                        fft,
                        (padx * pad_size) as usize,
                    )),
                );
                let (mut fx, mut fy, mut inv) = (final_size, final_size, 1);
                two_d_fft(spectrum.cast(), &mut fx, &mut fy, &mut inv);
            }
            libc::free(crop.cast());
            libc::free(fft.cast());
            return 0;
        }
        let mut cen = 0_f64;
        let lim = (pad_size / 10).max(5).min(pad_size / 2 - 1);
        for x in 0..lim {
            for y in 0..lim {
                if x == 0 && y == 0 {
                    continue;
                }
                cen = cen.max(fft_magnitude(fft, pad_size, pad_size, x, y));
                cen = cen.max(fft_magnitude(fft, pad_size, pad_size, x, pad_size - 1 - y));
            }
        }
        let mut sum = 0.;
        for y in 0..pad_size {
            sum += fft_magnitude(fft, pad_size, pad_size, pad_size / 2, y);
        }
        let log_scale = 5. / (sum / pad_size as f64);
        let scale = 32000. / (log_scale * cen + 1.).ln();
        let stemp = if temp.is_null() {
            spectrum.cast::<i16>()
        } else {
            temp
        };
        let mut yin = pad_size / 2;
        for yout in 0..pad_size {
            let base = yin * padx;
            let mut dst = stemp.add((yout * pad_size + pad_size / 2) as usize);
            for i in (base..base + pad_size).step_by(2) {
                let val = ((*fft.add(i as usize) * *fft.add(i as usize)
                    + *fft.add((i + 1) as usize) * *fft.add((i + 1) as usize))
                    as f64)
                    .sqrt();
                // C assigns the double expression through an `int` to a
                // `short int *`.  On the x86 reference build, `cvttsd2si`
                // produces the integer-indefinite value for non-finite or
                // out-of-range input, then the short store retains its low
                // 16 bits.  Rust casts otherwise saturate.
                let converted = scale * (log_scale * val + 1.).ln();
                *dst = if converted.is_finite()
                    && converted >= i32::MIN as f64
                    && converted <= i32::MAX as f64
                {
                    converted as i32 as i16
                } else {
                    i32::MIN as i16
                };
                dst = dst.add(1);
            }
            let i = base + pad_size;
            let val = ((*fft.add(i as usize) * *fft.add(i as usize)
                + *fft.add((i + 1) as usize) * *fft.add((i + 1) as usize))
                as f64)
                .sqrt();
            let converted = scale * (log_scale * val + 1.).ln();
            *stemp.add((((pad_size - yout) % pad_size) * pad_size) as usize) = if converted
                .is_finite()
                && converted >= i32::MIN as f64
                && converted <= i32::MAX as f64
            {
                converted as i32 as i16
            } else {
                i32::MIN as i16
            };
            yin = (yin + 1) % pad_size;
        }
        for yout in 0..pad_size {
            let yin = if yout != 0 { pad_size - yout } else { 0 };
            for i in 0..pad_size / 2 - 1 {
                *stemp.add((yout * pad_size + pad_size / 2 - 1 - i) as usize) =
                    *stemp.add((yin * pad_size + pad_size / 2 + 1 + i) as usize);
            }
        }
        *stemp.add((pad_size * pad_size / 2 + pad_size / 2) as usize) = 32000;
        let mut ret = 0;
        if reducing {
            // `zoomWithFilter` takes typed line and output slices now; the
            // `makeLinePointers` block above still runs for its error-5 path.
            let line_vec: Vec<&[i16]> = (0..pad_size as usize)
                .map(|i| {
                    core::slice::from_raw_parts(temp.add(i * pad_size as usize), pad_size as usize)
                })
                .collect();
            ret = crate::imod::libcfshr::zoomdown::zoom_with_filter(
                crate::imod::libcfshr::zoomdown::ZoomLines::Short(&line_vec),
                pad_size,
                pad_size,
                0.,
                0.,
                final_size,
                final_size,
                final_size,
                0,
                1,
                &mut crate::imod::libcfshr::zoomdown::ZoomOut::Short(
                    core::slice::from_raw_parts_mut(
                        if bkgd_gray > 0 {
                            fft.cast::<i16>()
                        } else {
                            spectrum.cast::<i16>()
                        },
                        (final_size * final_size) as usize,
                    ),
                ),
                None,
                None,
            );
        }
        if ret == 0 && bkgd_gray > 0 {
            let scalein = if reducing { fft.cast::<i16>() } else { temp };
            let (mut min, mut max) = (0_f32, 32000_f32);
            if final_size > 50 {
                let mut b = 0.;
                for y in 0..final_size {
                    for x in 2..5 {
                        b += *scalein.add((x + y * final_size) as usize) as f64;
                    }
                }
                let b = b / (3 * final_size) as f64;
                let rad = ((final_size * final_size) as f64).sqrt() * trunc_diam as f64 / 2.;
                let hi = rad.ceil() as i32 + 1;
                let lo = (0.7 * rad) as i32 - 1;
                let (mut rings, mut count) = (0., 0);
                for y in -hi..=hi {
                    for x in -hi..=hi {
                        if y < -lo || y > lo || x < -lo || x > lo {
                            if (((x * x + y * y) as f64).sqrt() - rad).abs() < 0.71 {
                                rings += *scalein.add(
                                    (final_size / 2 + x + (final_size / 2 + y) * final_size)
                                        as usize,
                                ) as f64;
                                count += 1;
                            }
                        }
                    }
                }
                if count > 3 {
                    max = (rings / count as f64) as f32;
                    let f = bkgd_gray as f32 / 256.;
                    min = (b as f32 - max * f) / (1. - f);
                }
            }
            let bytes = spectrum.cast::<u8>();
            let s = 255. / (max - min);
            for i in 0..final_size * final_size {
                *bytes.add(i as usize) =
                    ((s * (*scalein.add(i as usize) as f32 - min)) as i32).clamp(0, 255) as u8;
            }
        }
        if !temp.is_null() {
            libc::free(temp.cast())
        }
        libc::free(fft.cast());
        if !lines.is_null() {
            libc::free(lines.cast())
        };
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn amplitude_is_centered() {
        unsafe {
            let mut fft = [0_f32; 24];
            fft[0] = 3.;
            fft[1] = 4.;
            let mut out = [0_f32; 16];
            make_amplitude_spectrum(fft.as_mut_ptr(), out.as_mut_ptr(), 4, 4);
            assert_eq!(out[2 + 2 * 4], 5.);
        }
    }

    #[test]
    fn scaled_spectrum_has_center_peak() {
        unsafe {
            let mut image = [0_f32; 64];
            image[9] = 100.;
            let mut output = [0_i16; 64];
            assert_eq!(
                spectrum_scaled(
                    image.as_mut_ptr().cast(),
                    2,
                    8,
                    8,
                    output.as_mut_ptr().cast(),
                    8,
                    8,
                    0,
                    0.02,
                    3,
                    crate::imod::libfft::todfft
                ),
                0
            );
            assert_eq!(output[4 + 4 * 8], 32000);
        }
    }

    #[test]
    fn byte_scaling_also_applies_to_small_output() {
        unsafe {
            let image = [2_f32; 16];
            let mut output = [0_u8; 16];
            assert_eq!(
                spectrum_scaled(
                    image.as_ptr().cast_mut().cast(),
                    2,
                    4,
                    4,
                    output.as_mut_ptr().cast(),
                    4,
                    4,
                    96,
                    0.02,
                    3,
                    crate::imod::libfft::todfft,
                ),
                0
            );
            assert_eq!(output[2 + 2 * 4], 255);
        }
    }

    #[test]
    fn two_by_two_scaled_spectrum_retains_native_integer_indefinite_narrowing() {
        unsafe {
            let mut image = [1_u8, 2, 3, 4];
            let mut output = [0_i16; 4];
            assert_eq!(
                spectrum_scaled(
                    image.as_mut_ptr().cast(),
                    0,
                    2,
                    2,
                    output.as_mut_ptr().cast(),
                    2,
                    2,
                    0,
                    0.02,
                    3,
                    crate::imod::libfft::todfft,
                ),
                0
            );
            assert_eq!(output, [0, 0, 0, 32000]);
        }
    }
}
