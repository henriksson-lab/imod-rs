//! Translation of `IMOD/libcfshr/samplemeansd.c`.
#![allow(dead_code)]

/// C `sampleMinMaxMeanSD` (`samplemeansd.c:76`).
pub unsafe fn sample_min_max_mean_sd(
    image: *mut *mut u8,
    data_type: i32,
    _nx: i32,
    _ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: *mut f32,
    sd: *mut f32,
    amin: *mut f32,
    amax: *mut f32,
) -> i32 {
    unsafe {
        if image.is_null() || mean.is_null() || (amin.is_null() != amax.is_null()) {
            return -1;
        }
        let n_pix_use = nx_use as f64 * ny_use as f64;
        let mut n_sample = (sample as f64 * n_pix_use) as i32;
        if nx_use < 2 || ny_use < 2 || n_sample < 5 {
            return 1;
        }
        let (dx_sample, dy_sample) = if n_sample as f64 >= n_pix_use {
            n_sample = n_pix_use as i32;
            (1, 1)
        } else {
            let mut dy = (1.0 / sample.sqrt()).round() as i32;
            dy = dy.max(1).min(ny_use / 4);
            let ny_sample = ny_use / dy;
            let mut dx =
                (nx_use as f32 / 1.0_f32.max(n_sample as f32 / ny_sample as f32)).round() as i32;
            dx = dx.max(1).min(nx_use / 4);
            (dx, dy)
        };
        if !matches!(data_type, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 9) {
            return 2;
        }
        let nchan = if data_type == 9 { 4 } else { 3 };
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        let mut nsum = 0_i32;
        let mut tmin = 1.0e37_f32;
        let mut tmax = -tmin;
        let mut j = iy_start;
        while j < iy_start + ny_use {
            let mut tsum = 0.0_f64;
            let mut tsumsq = 0.0_f64;
            let mut itsum = 0_i32;
            let mut itsumsq = 0_i32;
            let mut itmin = 2_147_000_000_i32;
            let mut itmax = -itmin;
            let xstart = (j % 17) % dx_sample;
            nsum += (nx_use - xstart + dx_sample - 1) / dx_sample;
            let mut i = ix_start + xstart;
            match data_type {
                0 => {
                    let ptr = *image.add(j as usize);
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize) as i32;
                        itsum += val;
                        itsumsq += val * val;
                        if !amin.is_null() {
                            itmin = itmin.min(val);
                            itmax = itmax.max(val);
                        }
                        i += dx_sample;
                    }
                    tmin = tmin.min(itmin as f32);
                    tmax = tmax.max(itmax as f32);
                    tsum = itsum as f64;
                    tsumsq = itsumsq as f64;
                }
                1 => {
                    let ptr = (*image.add(j as usize)).cast::<i8>();
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize) as i32;
                        itsum += val;
                        itsumsq += val * val;
                        if !amin.is_null() {
                            itmin = itmin.min(val);
                            itmax = itmax.max(val);
                        }
                        i += dx_sample;
                    }
                    tmin = tmin.min(itmin as f32);
                    tmax = tmax.max(itmax as f32);
                    tsum = itsum as f64;
                    tsumsq = itsumsq as f64;
                }
                2 => {
                    let ptr = (*image.add(j as usize)).cast::<u16>();
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize) as i32;
                        tsum += val as f64;
                        tsumsq += ((val as f32) * val as f32) as f64;
                        if !amin.is_null() {
                            tmin = tmin.min(val as f32);
                            tmax = tmax.max(val as f32);
                        }
                        i += dx_sample;
                    }
                }
                3 => {
                    let ptr = (*image.add(j as usize)).cast::<i16>();
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize) as i32;
                        tsum += val as f64;
                        tsumsq += ((val as f32) * val as f32) as f64;
                        if !amin.is_null() {
                            tmin = tmin.min(val as f32);
                            tmax = tmax.max(val as f32);
                        }
                        i += dx_sample;
                    }
                }
                6 => {
                    let ptr = (*image.add(j as usize)).cast::<f32>();
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize);
                        tsum += val as f64;
                        tsumsq += (val * val) as f64;
                        if !amin.is_null() {
                            tmin = tmin.min(val);
                            tmax = tmax.max(val);
                        }
                        i += dx_sample;
                    }
                }
                7 => {
                    let ptr = (*image.add(j as usize)).cast::<i32>();
                    while i < ix_start + nx_use {
                        let val = *ptr.add(i as usize) as f32;
                        tsum += val as f64;
                        tsumsq += (val * val) as f64;
                        if !amin.is_null() {
                            tmin = tmin.min(val);
                            tmax = tmax.max(val);
                        }
                        i += dx_sample;
                    }
                }
                8 | 9 => {
                    let ptr = *image.add(j as usize);
                    while i < ix_start + nx_use {
                        let val = 0.3_f32 * *ptr.add((nchan * i) as usize) as f32
                            + 0.59_f32 * *ptr.add((nchan * i + 1) as usize) as f32
                            + 0.11_f32 * *ptr.add((nchan * i + 2) as usize) as f32;
                        tsum += val as f64;
                        tsumsq += (val * val) as f64;
                        if !amin.is_null() {
                            tmin = tmin.min(val);
                            tmax = tmax.max(val);
                        }
                        i += dx_sample;
                    }
                }
                _ => unreachable!(),
            }
            sum += tsum;
            sumsq += tsumsq;
            j += dy_sample;
        }
        sum /= nsum as f64;
        if nsum > 1 && !sd.is_null() {
            sumsq = (sumsq - nsum as f64 * sum * sum) / (nsum as f64 - 1.0);
            if sumsq < 0.0 {
                sumsq = 0.0;
            }
            sumsq = sumsq.sqrt();
        } else {
            sumsq = 0.0;
        }
        *mean = sum as f32;
        if !sd.is_null() {
            *sd = sumsq as f32;
        }
        if !amin.is_null() {
            *amin = tmin;
        }
        if !amax.is_null() {
            *amax = tmax;
        }
        0
    }
}

/// C `sampleMeanOnly` (`samplemeansd.c:272`).
pub unsafe fn sample_mean_only(
    image: *mut *mut u8,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: *mut f32,
) -> i32 {
    unsafe {
        sample_min_max_mean_sd(
            image,
            data_type,
            nx,
            ny,
            sample,
            ix_start,
            iy_start,
            nx_use,
            ny_use,
            mean,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    }
}

/// C `sampleMeanSD` (`samplemeansd.c:283`).
pub unsafe fn sample_mean_sd(
    image: *mut *mut u8,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: *mut f32,
    sd: *mut f32,
) -> i32 {
    unsafe {
        sample_min_max_mean_sd(
            image,
            data_type,
            nx,
            ny,
            sample,
            ix_start,
            iy_start,
            nx_use,
            ny_use,
            mean,
            sd,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    }
}

/// C `sampleMinMaxMean` (`samplemeansd.c:294`).
pub unsafe fn sample_min_max_mean(
    image: *mut *mut u8,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: *mut f32,
    amin: *mut f32,
    amax: *mut f32,
) -> i32 {
    unsafe {
        sample_min_max_mean_sd(
            image,
            data_type,
            nx,
            ny,
            sample,
            ix_start,
            iy_start,
            nx_use,
            ny_use,
            mean,
            core::ptr::null_mut(),
            amin,
            amax,
        )
    }
}

/// C Fortran wrapper `sampleminmaxmeansd` (`samplemeansd.c:307`).
pub unsafe fn sample_min_max_mean_sd_fortran(
    image: *mut f32,
    nx: *const i32,
    ny: *const i32,
    sample: *const f32,
    ix_start: *const i32,
    iy_start: *const i32,
    nx_use: *const i32,
    ny_use: *const i32,
    which: i32,
    mean: *mut f32,
    sd: *mut f32,
    dmin: *mut f32,
    dmax: *mut f32,
) -> i32 {
    unsafe {
        if which < 1 || which > 4 {
            return -2;
        }
        let mut lines = Vec::with_capacity(*ny as usize);
        for i in 0..*ny {
            lines.push(image.add((i * *nx) as usize).cast::<u8>());
        }
        match which {
            1 => sample_min_max_mean_sd(
                lines.as_mut_ptr(),
                6,
                *nx,
                *ny,
                *sample,
                *ix_start,
                *iy_start,
                *nx_use,
                *ny_use,
                mean,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
            2 => sample_min_max_mean_sd(
                lines.as_mut_ptr(),
                6,
                *nx,
                *ny,
                *sample,
                *ix_start,
                *iy_start,
                *nx_use,
                *ny_use,
                mean,
                sd,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
            3 => sample_min_max_mean_sd(
                lines.as_mut_ptr(),
                6,
                *nx,
                *ny,
                *sample,
                *ix_start,
                *iy_start,
                *nx_use,
                *ny_use,
                mean,
                core::ptr::null_mut(),
                dmin,
                dmax,
            ),
            _ => sample_min_max_mean_sd(
                lines.as_mut_ptr(),
                6,
                *nx,
                *ny,
                *sample,
                *ix_start,
                *iy_start,
                *nx_use,
                *ny_use,
                mean,
                sd,
                dmin,
                dmax,
            ),
        }
    }
}

/// C Fortran wrapper `samplemeansd` (`samplemeansd.c:337`).
pub unsafe fn sample_mean_sd_fortran(
    image: *mut f32,
    nx: *const i32,
    ny: *const i32,
    sample: *const f32,
    ix_start: *const i32,
    iy_start: *const i32,
    nx_use: *const i32,
    ny_use: *const i32,
    mean: *mut f32,
    sd: *mut f32,
    dmin: *mut f32,
    dmax: *mut f32,
) -> i32 {
    unsafe {
        sample_min_max_mean_sd_fortran(
            image, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, 2, mean, sd, dmin, dmax,
        )
    }
}

/// C `typeForSampleMean` (`samplemeansd.c:350`).
pub fn type_for_sample_mean(mrc_mode: i32) -> i32 {
    match mrc_mode {
        0 => 0,
        6 => 2,
        1 => 3,
        2 => 6,
        16 => 8,
        _ => -1,
    }
}

/// C Fortran wrapper `typeforsamplemean` (`samplemeansd.c:369`).
pub unsafe fn type_for_sample_mean_fortran(mrc_mode: *const i32) -> i32 {
    unsafe { type_for_sample_mean(*mrc_mode) }
}

/// C `getSampleOfArray` (`samplemeansd.c:387`).
pub unsafe fn get_sample_of_array(
    image: *const core::ffi::c_void,
    mode: i32,
    nx: i32,
    _ny: i32,
    sample_frac: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    fill_to_exclude: f32,
    samples: *mut f32,
    max_samples: i32,
    num_samples: *mut i32,
) -> i32 {
    unsafe {
        let int_max = 2_147_000_000_i32;
        let ifill = if fill_to_exclude > int_max as f32 || fill_to_exclude < -(int_max as f32) {
            int_max
        } else {
            fill_to_exclude.round() as i32
        };
        let n_pix_use = nx_use as f64 * ny_use as f64;
        let mut n_sample = ((sample_frac as f64 * n_pix_use) as i32).min(max_samples);
        if nx_use < 2 || ny_use < 2 || n_sample < 5 {
            return 1;
        }
        let (mut dx_sample, mut dy_sample) = if n_sample as f64 >= n_pix_use {
            n_sample = n_pix_use as i32;
            (1, 1)
        } else {
            let mut dy = (1.0 / (n_sample as f64 / n_pix_use).sqrt()).round() as i32;
            dy = dy.max(1).min(ny_use / 4);
            let ny_sample = ny_use / dy;
            let mut dx =
                (nx_use as f32 / 1.0_f32.max(n_sample as f32 / ny_sample as f32)).round() as i32;
            dx = dx.max(1).min(nx_use / 4);
            (dx, dy)
        };
        while (ny_use / dy_sample + 1) * (nx_use / dx_sample + 1) > max_samples {
            dx_sample = nx_use.min(dx_sample + 1);
            dy_sample = ny_use.min(dy_sample + 1);
        }
        let mut ind_samp = 0_usize;
        let mut j = iy_start;
        while j < iy_start + ny_use {
            let xstart = (j % 17) % dx_sample;
            let mut i = ix_start + xstart;
            match mode {
                0 => {
                    while i < ix_start + nx_use {
                        let value = *image.cast::<u8>().add((j * nx + i) as usize);
                        if value as i32 != ifill {
                            *samples.add(ind_samp) = value as f32;
                            ind_samp += 1;
                        }
                        i += dx_sample;
                    }
                }
                1 => {
                    while i < ix_start + nx_use {
                        let value = *image.cast::<i16>().add((j * nx + i) as usize);
                        if value as i32 != ifill {
                            *samples.add(ind_samp) = value as f32;
                            ind_samp += 1;
                        }
                        i += dx_sample;
                    }
                }
                6 => {
                    while i < ix_start + nx_use {
                        let value = *image.cast::<u16>().add((j * nx + i) as usize);
                        if value as i32 != ifill {
                            *samples.add(ind_samp) = value as f32;
                            ind_samp += 1;
                        }
                        i += dx_sample;
                    }
                }
                2 => {
                    while i < ix_start + nx_use {
                        let value = *image.cast::<f32>().add((j * nx + i) as usize);
                        if (value - fill_to_exclude).abs() > 1.0e-5 {
                            *samples.add(ind_samp) = value;
                            ind_samp += 1;
                        }
                        i += dx_sample;
                    }
                }
                _ => return 2,
            }
            j += dy_sample;
        }
        *num_samples = ind_samp as i32;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        get_sample_of_array, sample_min_max_mean_sd, sample_min_max_mean_sd_fortran,
        type_for_sample_mean,
    };
    #[test]
    fn byte_and_rgb_statistics_match_source_rules() {
        let mut byte = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut lines = [
            byte.as_mut_ptr(),
            unsafe { byte.as_mut_ptr().add(3) },
            unsafe { byte.as_mut_ptr().add(6) },
        ];
        let mut mean = 0.;
        let mut sd = 0.;
        let mut low = 0.;
        let mut high = 0.;
        assert_eq!(
            unsafe {
                sample_min_max_mean_sd(
                    lines.as_mut_ptr(),
                    0,
                    3,
                    3,
                    1.,
                    0,
                    0,
                    3,
                    3,
                    &mut mean,
                    &mut sd,
                    &mut low,
                    &mut high,
                )
            },
            0
        );
        assert_eq!((mean, low, high), (5., 1., 9.));
        assert!((sd - 2.738613).abs() < 1.0e-5);
        let mut rgb = [
            100_u8, 0, 0, 0, 100, 0, 0, 0, 100, 100, 100, 100, 10, 20, 30, 30, 20, 10,
        ];
        let mut rgb_lines = [rgb.as_mut_ptr(), unsafe { rgb.as_mut_ptr().add(9) }];
        assert_eq!(
            unsafe {
                sample_min_max_mean_sd(
                    rgb_lines.as_mut_ptr(),
                    8,
                    3,
                    2,
                    1.,
                    0,
                    0,
                    3,
                    2,
                    &mut mean,
                    core::ptr::null_mut(),
                    &mut low,
                    &mut high,
                )
            },
            0
        );
        assert!((mean - 40.0).abs() < 1.0e-5);
        assert!((low - 11.0).abs() < 1.0e-5);
        assert!((high - 100.).abs() < 1.0e-5);
    }
    #[test]
    fn fortran_wrapper_and_array_sampling_obey_source_modes() {
        let mut image = [1_f32, 2., 3., 4., 5., 6., 7., 8., 9.];
        let n = 3;
        let sample = 1.;
        let zero = 0;
        let mut mean = 0.;
        let mut sd = 0.;
        assert_eq!(
            unsafe {
                sample_min_max_mean_sd_fortran(
                    image.as_mut_ptr(),
                    &n,
                    &n,
                    &sample,
                    &zero,
                    &zero,
                    &n,
                    &n,
                    2,
                    &mut mean,
                    &mut sd,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            },
            0
        );
        assert_eq!(mean, 5.);
        assert!((sd - 2.738613).abs() < 1.0e-5);
        let data = [1_u8, 2, 9, 4, 5, 6, 7, 8, 9];
        let mut out = [0_f32; 9];
        let mut count = 0;
        assert_eq!(
            unsafe {
                get_sample_of_array(
                    data.as_ptr().cast(),
                    0,
                    3,
                    3,
                    1.,
                    0,
                    0,
                    3,
                    3,
                    9.,
                    out.as_mut_ptr(),
                    9,
                    &mut count,
                )
            },
            0
        );
        // The source increases both grid intervals to honor max_samples.
        assert_eq!(count, 2);
        assert_eq!(&out[..2], &[1., 7.]);
        assert_eq!(
            (
                type_for_sample_mean(0),
                type_for_sample_mean(6),
                type_for_sample_mean(42)
            ),
            (0, 2, -1)
        );
    }
}
