//! Translation of `IMOD/libcfshr/spectrumscaled.c`.

/// Image samples accepted by `spectrum_scaled`.
pub enum SpectrumInput<'a> {
    Byte(&'a [u8]),
    Short(&'a [i16]),
    UShort(&'a [u16]),
    Float(&'a [f32]),
    Rgb(&'a [u8]),
}

/// Destination samples accepted by `spectrum_scaled`.
pub enum SpectrumOutput<'a> {
    Byte(&'a mut [u8]),
    Short(&'a mut [i16]),
    Float(&'a mut [f32]),
}

/// C `makeAmplitudeSpectrum` (`spectrumscaled.c:276`).
pub fn make_amplitude_spectrum(
    fft_array: &[f32],
    spectrum: &mut [f32],
    mut pad_size: i32,
    out_xdim: i32,
) {
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
                spectrum[(iyout * out_xdim + pad_size / 2 + i) as usize] =
                    fft_array[(ixbase + i) as usize];
            }
            spectrum[(((pad_size - iyout) % pad_size) * out_xdim) as usize] =
                fft_array[(ixbase + pad_size / 2) as usize];
        } else {
            for i in (0..pad_size).step_by(2) {
                let real = fft_array[(ixbase + i) as usize];
                let imag = fft_array[(ixbase + i + 1) as usize];
                spectrum[(iyout * out_xdim + pad_size / 2 + i / 2) as usize] =
                    (real * real + imag * imag).sqrt();
            }
            let real = fft_array[(ixbase + pad_size) as usize];
            let imag = fft_array[(ixbase + pad_size + 1) as usize];
            spectrum[(((pad_size - iyout) % pad_size) * out_xdim) as usize] =
                (real * real + imag * imag).sqrt();
        }
        iyin = (iyin + 1) % pad_size;
    }
    for iyout in 0..pad_size {
        let iyin = if iyout != 0 { pad_size - iyout } else { 0 };
        for i in 0..pad_size / 2 - 1 {
            spectrum[(iyout * out_xdim + pad_size / 2 - 1 - i) as usize] =
                spectrum[(iyin * out_xdim + pad_size / 2 + 1 + i) as usize];
        }
    }
}

/// C static `FFTMagnitude`.
fn fft_magnitude(array: &[f32], nx: i32, _ny: i32, ix: i32, iy: i32) -> f64 {
    let i = (ix * 2 + iy * (nx + 2)) as usize;
    ((array[i] * array[i] + array[i + 1] * array[i + 1]) as f64).sqrt()
}

/// C `spectrumScaled`.
pub fn spectrum_scaled(
    image: SpectrumInput<'_>,
    nx: i32,
    ny: i32,
    mut spectrum: SpectrumOutput<'_>,
    mut pad_size: i32,
    final_size: i32,
    bkgd_gray: i32,
    trunc_diam: f32,
    filt_type: i32,
    two_d_fft: fn(&mut [f32], i32, i32, i32),
) -> i32 {
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
    let Some(work_size) = usize::try_from(padx).ok().and_then(|width| {
        usize::try_from(pad_size)
            .ok()
            .and_then(|height| width.checked_mul(height))
    }) else {
        return -3;
    };
    let mut fft = Vec::new();
    if fft.try_reserve_exact(work_size).is_err() {
        return -3;
    }
    fft.resize(work_size, 0.);
    let mut temp = None;
    if bkgd_gray > 0 || reducing {
        let mut storage = Vec::new();
        if storage.try_reserve_exact(work_size).is_err() {
            return -3;
        }
        storage.resize(work_size, 0_i16);
        temp = Some(storage);
    }
    let mut crop = None;
    if cropping {
        let mut storage = Vec::new();
        if storage.try_reserve_exact(work_size).is_err() {
            return -3;
        }
        storage.resize(work_size, 0.);
        crop = Some(storage);
    }
    let (input, typ) = match image {
        SpectrumInput::Byte(data) => (crate::imod::libcfshr::taperpad::PadIn::Byte(data), 0),
        SpectrumInput::Short(data) => (crate::imod::libcfshr::taperpad::PadIn::Short(data), 1),
        SpectrumInput::UShort(data) => (crate::imod::libcfshr::taperpad::PadIn::UShort(data), 6),
        SpectrumInput::Float(data) => (crate::imod::libcfshr::taperpad::PadIn::Float(data), 2),
        SpectrumInput::Rgb(data) => (crate::imod::libcfshr::taperpad::PadIn::Rgb(data), 16),
    };
    crate::imod::libcfshr::taperpad::slice_taper_in_pad(
        input,
        typ,
        nx,
        0,
        nx - 1,
        0,
        ny - 1,
        &mut fft,
        padx,
        pad_size,
        pad_size,
        (nx as f32 * taper) as i32,
        (ny as f32 * taper) as i32,
    );
    let (px, py, mut idir) = (pad_size, pad_size, 0);
    two_d_fft(&mut fft, px, py, idir);
    if filt_type < 0 {
        if cropping {
            let output = crop.as_mut().unwrap();
            make_amplitude_spectrum(&fft, output, pad_size, padx);
            output[(padx * pad_size / 2 + pad_size / 2) as usize] = 0.;
            idir = 0;
            two_d_fft(crop.as_mut().unwrap(), px, py, idir);
            let shift = 0.5 * (pad_size as f32 / final_size as f32 - 1.);
            let SpectrumOutput::Float(output) = &mut spectrum else {
                return -4;
            };
            crate::imod::libcfshr::filtxcorr::fourier_reduce_image(
                crop.as_ref().unwrap(),
                pad_size,
                pad_size,
                output,
                final_size,
                final_size,
                shift,
                shift,
                Some(&mut fft),
            );
            let (fx, fy, inv) = (final_size, final_size, 1);
            two_d_fft(output, fx, fy, inv);
        } else {
            let SpectrumOutput::Float(output) = &mut spectrum else {
                return -4;
            };
            make_amplitude_spectrum(&fft, output, pad_size, padx);
            output[(padx * pad_size / 2 + pad_size / 2) as usize] = 0.;
        }
        return 0;
    }
    let mut cen = 0_f64;
    let lim = (pad_size / 10).max(5).min(pad_size / 2 - 1);
    for x in 0..lim {
        for y in 0..lim {
            if x == 0 && y == 0 {
                continue;
            }
            cen = cen.max(fft_magnitude(&fft, pad_size, pad_size, x, y));
            cen = cen.max(fft_magnitude(&fft, pad_size, pad_size, x, pad_size - 1 - y));
        }
    }
    let mut sum = 0.;
    for y in 0..pad_size {
        sum += fft_magnitude(&fft, pad_size, pad_size, pad_size / 2, y);
    }
    let log_scale = 5. / (sum / pad_size as f64);
    let scale = 32000. / (log_scale * cen + 1.).ln();
    {
        let stemp: &mut [i16] = if let Some(temp) = temp.as_mut() {
            temp
        } else if let SpectrumOutput::Short(output) = &mut spectrum {
            output
        } else {
            return -4;
        };
        let mut yin = pad_size / 2;
        for yout in 0..pad_size {
            let base = yin * padx;
            for i in (base..base + pad_size).step_by(2) {
                let val = ((fft[i as usize] * fft[i as usize]
                    + fft[(i + 1) as usize] * fft[(i + 1) as usize])
                    as f64)
                    .sqrt();
                // C assigns the double expression through an `int` to a
                // `short int *`.  On the x86 reference build, `cvttsd2si`
                // produces the integer-indefinite value for non-finite or
                // out-of-range input, then the short store retains its low
                // 16 bits.  Rust casts otherwise saturate.
                let converted = scale * (log_scale * val + 1.).ln();
                stemp[(yout * pad_size + pad_size / 2 + (i - base) / 2) as usize] = if converted
                    .is_finite()
                    && converted >= i32::MIN as f64
                    && converted <= i32::MAX as f64
                {
                    converted as i32 as i16
                } else {
                    i32::MIN as i16
                };
            }
            let i = base + pad_size;
            let val = ((fft[i as usize] * fft[i as usize]
                + fft[(i + 1) as usize] * fft[(i + 1) as usize]) as f64)
                .sqrt();
            let converted = scale * (log_scale * val + 1.).ln();
            stemp[(((pad_size - yout) % pad_size) * pad_size) as usize] = if converted.is_finite()
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
                stemp[(yout * pad_size + pad_size / 2 - 1 - i) as usize] =
                    stemp[(yin * pad_size + pad_size / 2 + 1 + i) as usize];
            }
        }
        stemp[(pad_size * pad_size / 2 + pad_size / 2) as usize] = 32000;
    }
    let mut ret = 0;
    let mut reduced = Vec::new();
    if reducing {
        let reduced_size = (final_size * final_size) as usize;
        if reduced.try_reserve_exact(reduced_size).is_err() {
            return -3;
        }
        reduced.resize(reduced_size, 0_i16);
        let mut line_vec = Vec::new();
        if line_vec.try_reserve_exact(pad_size as usize).is_err() {
            return -3;
        }
        let temp_data = temp.as_ref().unwrap();
        for row in 0..pad_size as usize {
            let start = row * pad_size as usize;
            line_vec.push(&temp_data[start..start + pad_size as usize]);
        }
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
            &mut crate::imod::libcfshr::zoomdown::ZoomOut::Short(if bkgd_gray > 0 {
                &mut reduced
            } else if let SpectrumOutput::Short(output) = &mut spectrum {
                output
            } else {
                return -4;
            }),
            None,
            None,
        );
    }
    if ret == 0 && bkgd_gray > 0 {
        let scalein: &[i16] = if reducing {
            &reduced
        } else {
            temp.as_ref().unwrap()
        };
        let (mut min, mut max) = (0_f32, 32000_f32);
        if final_size > 50 {
            let mut b = 0.;
            for y in 0..final_size {
                for x in 2..5 {
                    b += scalein[(x + y * final_size) as usize] as f64;
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
                            rings += scalein
                                [(final_size / 2 + x + (final_size / 2 + y) * final_size) as usize]
                                as f64;
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
        let SpectrumOutput::Byte(bytes) = &mut spectrum else {
            return -4;
        };
        let s = 255. / (max - min);
        for i in 0..final_size * final_size {
            bytes[i as usize] =
                ((s * (scalein[i as usize] as f32 - min)) as i32).clamp(0, 255) as u8;
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn amplitude_is_centered() {
        let mut fft = [0_f32; 24];
        fft[0] = 3.;
        fft[1] = 4.;
        let mut out = [0_f32; 16];
        make_amplitude_spectrum(&fft, &mut out, 4, 4);
        assert_eq!(out[2 + 2 * 4], 5.);
    }

    #[test]
    fn scaled_spectrum_has_center_peak() {
        let mut image = [0_f32; 64];
        image[9] = 100.;
        let mut output = [0_i16; 64];
        assert_eq!(
            spectrum_scaled(
                SpectrumInput::Float(&image),
                8,
                8,
                SpectrumOutput::Short(&mut output),
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

    #[test]
    fn byte_scaling_also_applies_to_small_output() {
        let image = [2_f32; 16];
        let mut output = [0_u8; 16];
        assert_eq!(
            spectrum_scaled(
                SpectrumInput::Float(&image),
                4,
                4,
                SpectrumOutput::Byte(&mut output),
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

    #[test]
    fn two_by_two_scaled_spectrum_retains_native_integer_indefinite_narrowing() {
        let image = [1_u8, 2, 3, 4];
        let mut output = [0_i16; 4];
        assert_eq!(
            spectrum_scaled(
                SpectrumInput::Byte(&image),
                2,
                2,
                SpectrumOutput::Short(&mut output),
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
