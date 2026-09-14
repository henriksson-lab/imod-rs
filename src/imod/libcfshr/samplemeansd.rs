//! Translation of `IMOD/libcfshr/samplemeansd.c`.
#![allow(dead_code)]

/// C `sampleMinMaxMeanSD` (`samplemeansd.c:76`).
///
/// The source's `unsigned char **image` is an array of line pointers that each
/// case of the `type` switch re-types; here it is the array of line byte views,
/// and each arm reads its own element type out of those bytes.  A NULL `image`,
/// `mean`, `sd`, `amin` or `amax` in the source is `None`.
pub fn sample_min_max_mean_sd(
    image: Option<&[&[u8]]>,
    data_type: i32,
    _nx: i32,
    _ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: Option<&mut f32>,
    sd: Option<&mut f32>,
    amin: Option<&mut f32>,
    amax: Option<&mut f32>,
) -> i32 {
    let (image, mean) = match (image, mean) {
        (Some(image), Some(mean)) if amin.is_some() == amax.is_some() => (image, mean),
        _ => return -1,
    };
    let n_pix_use = nx_use as f64 * ny_use as f64;
    let mut n_sample = (sample as f64 * n_pix_use) as i32;
    if nx_use < 2 || ny_use < 2 || n_sample < 5 {
        return 1;
    }
    let (dx_sample, dy_sample) = if n_sample as f64 >= n_pix_use {
        n_sample = n_pix_use as i32;
        (1, 1)
    } else {
        // `fracY` is a `float` in the source; `sqrt` is the double routine
        // and its result rounds to single before `1. / fracY` widens it.
        // `B3DNINT` is `floor(a + 0.5)` and `B3DCLAMP` is
        // `B3DMAX(minv, B3DMIN(maxv, val))`, which is not `max` then `min`.
        let frac_y = (sample as f64).sqrt() as f32;
        let mut dy = (1.0 / frac_y as f64 + 0.5).floor() as i32;
        dy = 1.max((ny_use / 4).min(dy));
        let ny_sample = ny_use / dy;
        let ratio = n_sample as f32 / ny_sample as f32;
        let denom = (if 1.0_f64 > ratio as f64 {
            1.0_f64
        } else {
            ratio as f64
        }) as f32;
        let mut dx = ((nx_use as f32 / denom) as f64 + 0.5).floor() as i32;
        dx = 1.max((nx_use / 4).min(dx));
        (dx, dy)
    };
    if !matches!(data_type, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 9) {
        return 2;
    }
    let nchan = if data_type == 9 { 4 } else { 3 };
    let want_min = amin.is_some();
    // The source's ROW_SUM macro accumulates the sum of squares only in the
    // `sd` arms and the min/max only in the `amin` arms (`samplemeansd.c:31`).
    let want_sd = sd.is_some();
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
        let line = image[j as usize];
        match data_type {
            0 => {
                while i < ix_start + nx_use {
                    let val = line[i as usize] as i32;
                    itsum += val;
                    if want_sd {
                        itsumsq += val * val;
                    }
                    if want_min {
                        itmin = itmin.min(val);
                        itmax = itmax.max(val);
                    }
                    i += dx_sample;
                }
                tmin = if tmin < itmin as f32 {
                    tmin
                } else {
                    itmin as f32
                };
                tmax = if tmax > itmax as f32 {
                    tmax
                } else {
                    itmax as f32
                };
                tsum = itsum as f64;
                tsumsq = itsumsq as f64;
            }
            1 => {
                while i < ix_start + nx_use {
                    let val = line[i as usize] as i8 as i32;
                    itsum += val;
                    if want_sd {
                        itsumsq += val * val;
                    }
                    if want_min {
                        itmin = itmin.min(val);
                        itmax = itmax.max(val);
                    }
                    i += dx_sample;
                }
                tmin = if tmin < itmin as f32 {
                    tmin
                } else {
                    itmin as f32
                };
                tmax = if tmax > itmax as f32 {
                    tmax
                } else {
                    itmax as f32
                };
                tsum = itsum as f64;
                tsumsq = itsumsq as f64;
            }
            2 => {
                while i < ix_start + nx_use {
                    let k = 2 * i as usize;
                    let val = u16::from_ne_bytes([line[k], line[k + 1]]) as i32;
                    tsum += val as f64;
                    if want_sd {
                        tsumsq += ((val as f32) * val as f32) as f64;
                    }
                    if want_min {
                        tmin = if tmin < val as f32 { tmin } else { val as f32 };
                        tmax = if tmax > val as f32 { tmax } else { val as f32 };
                    }
                    i += dx_sample;
                }
            }
            3 => {
                while i < ix_start + nx_use {
                    let k = 2 * i as usize;
                    let val = i16::from_ne_bytes([line[k], line[k + 1]]) as i32;
                    tsum += val as f64;
                    if want_sd {
                        tsumsq += ((val as f32) * val as f32) as f64;
                    }
                    if want_min {
                        tmin = if tmin < val as f32 { tmin } else { val as f32 };
                        tmax = if tmax > val as f32 { tmax } else { val as f32 };
                    }
                    i += dx_sample;
                }
            }
            6 => {
                while i < ix_start + nx_use {
                    let k = 4 * i as usize;
                    let val = f32::from_ne_bytes([line[k], line[k + 1], line[k + 2], line[k + 3]]);
                    tsum += val as f64;
                    if want_sd {
                        tsumsq += (val * val) as f64;
                    }
                    if want_min {
                        tmin = if tmin < val { tmin } else { val };
                        tmax = if tmax > val { tmax } else { val };
                    }
                    i += dx_sample;
                }
            }
            7 => {
                while i < ix_start + nx_use {
                    let k = 4 * i as usize;
                    let val =
                        i32::from_ne_bytes([line[k], line[k + 1], line[k + 2], line[k + 3]]) as f32;
                    tsum += val as f64;
                    if want_sd {
                        tsumsq += (val * val) as f64;
                    }
                    if want_min {
                        tmin = if tmin < val { tmin } else { val };
                        tmax = if tmax > val { tmax } else { val };
                    }
                    i += dx_sample;
                }
            }
            8 | 9 => {
                while i < ix_start + nx_use {
                    let val = 0.3_f32 * line[(nchan * i) as usize] as f32
                        + 0.59_f32 * line[(nchan * i + 1) as usize] as f32
                        + 0.11_f32 * line[(nchan * i + 2) as usize] as f32;
                    tsum += val as f64;
                    if want_sd {
                        tsumsq += (val * val) as f64;
                    }
                    if want_min {
                        tmin = if tmin < val { tmin } else { val };
                        tmax = if tmax > val { tmax } else { val };
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
    if nsum > 1 && sd.is_some() {
        sumsq = (sumsq - nsum as f64 * sum * sum) / (nsum as f64 - 1.0);
        if sumsq < 0.0 {
            sumsq = 0.0;
        }
        sumsq = sumsq.sqrt();
    } else {
        sumsq = 0.0;
    }
    *mean = sum as f32;
    if let Some(sd) = sd {
        *sd = sumsq as f32;
    }
    if let Some(amin) = amin {
        *amin = tmin;
    }
    if let Some(amax) = amax {
        *amax = tmax;
    }
    0
}

/// C `sampleMeanOnly` (`samplemeansd.c:272`).
pub fn sample_mean_only(
    image: Option<&[&[u8]]>,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: Option<&mut f32>,
) -> i32 {
    sample_min_max_mean_sd(
        image, data_type, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, mean, None, None,
        None,
    )
}

/// C `sampleMeanSD` (`samplemeansd.c:283`).
pub fn sample_mean_sd(
    image: Option<&[&[u8]]>,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: Option<&mut f32>,
    sd: Option<&mut f32>,
) -> i32 {
    sample_min_max_mean_sd(
        image, data_type, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, mean, sd, None, None,
    )
}

/// C `sampleMinMaxMean` (`samplemeansd.c:294`).
pub fn sample_min_max_mean(
    image: Option<&[&[u8]]>,
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: Option<&mut f32>,
    amin: Option<&mut f32>,
    amax: Option<&mut f32>,
) -> i32 {
    sample_min_max_mean_sd(
        image, data_type, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, mean, None, amin,
        amax,
    )
}

/// C Fortran wrapper `sampleminmaxmeansd` (`samplemeansd.c:307`).
///
/// `image` is the byte view of the Fortran real*4 array; the wrapper's own
/// `makeLinePointers` call becomes the line slices built below.  The source's
/// `-1` return for a failed allocation has no equivalent.
pub fn sample_min_max_mean_sd_fortran(
    image: &[u8],
    nx: &i32,
    ny: &i32,
    sample: &f32,
    ix_start: &i32,
    iy_start: &i32,
    nx_use: &i32,
    ny_use: &i32,
    which: i32,
    mean: &mut f32,
    sd: &mut f32,
    dmin: &mut f32,
    dmax: &mut f32,
) -> i32 {
    if which < 1 || which > 4 {
        return -2;
    }
    let mut lines: Vec<&[u8]> = Vec::with_capacity(*ny as usize);
    for i in 0..*ny {
        lines.push(&image[(*nx as usize * i as usize * 4)..]);
    }
    match which {
        1 => sample_min_max_mean_sd(
            Some(&lines),
            6,
            *nx,
            *ny,
            *sample,
            *ix_start,
            *iy_start,
            *nx_use,
            *ny_use,
            Some(mean),
            None,
            None,
            None,
        ),
        2 => sample_min_max_mean_sd(
            Some(&lines),
            6,
            *nx,
            *ny,
            *sample,
            *ix_start,
            *iy_start,
            *nx_use,
            *ny_use,
            Some(mean),
            Some(sd),
            None,
            None,
        ),
        3 => sample_min_max_mean_sd(
            Some(&lines),
            6,
            *nx,
            *ny,
            *sample,
            *ix_start,
            *iy_start,
            *nx_use,
            *ny_use,
            Some(mean),
            None,
            Some(dmin),
            Some(dmax),
        ),
        _ => sample_min_max_mean_sd(
            Some(&lines),
            6,
            *nx,
            *ny,
            *sample,
            *ix_start,
            *iy_start,
            *nx_use,
            *ny_use,
            Some(mean),
            Some(sd),
            Some(dmin),
            Some(dmax),
        ),
    }
}

/// C Fortran wrapper `samplemeansd` (`samplemeansd.c:337`).
pub fn sample_mean_sd_fortran(
    image: &[u8],
    nx: &i32,
    ny: &i32,
    sample: &f32,
    ix_start: &i32,
    iy_start: &i32,
    nx_use: &i32,
    ny_use: &i32,
    mean: &mut f32,
    sd: &mut f32,
    dmin: &mut f32,
    dmax: &mut f32,
) -> i32 {
    sample_min_max_mean_sd_fortran(
        image, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, 2, mean, sd, dmin, dmax,
    )
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
pub fn type_for_sample_mean_fortran(mrc_mode: &i32) -> i32 {
    type_for_sample_mean(*mrc_mode)
}

/// C `getSampleOfArray` (`samplemeansd.c:387`).
///
/// The source's `void *image` is the raw byte view of the image; `mode` alone
/// says how to read an element from it.
pub fn get_sample_of_array(
    image: &[u8],
    mode: i32,
    nx: i32,
    _ny: i32,
    sample_frac: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    fill_to_exclude: f32,
    samples: &mut [f32],
    max_samples: i32,
    num_samples: &mut i32,
) -> i32 {
    let int_max = 2_147_000_000_i32;
    let ifill = if fill_to_exclude > int_max as f32 || fill_to_exclude < -(int_max as f32) {
        int_max
    } else {
        (fill_to_exclude as f64 + 0.5).floor() as i32
    };
    let n_pix_use = nx_use as f64 * ny_use as f64;
    let mut n_sample = ((sample_frac as f64 * n_pix_use) as i32).min(max_samples);
    if nx_use < 2 || ny_use < 2 || n_sample < 5 {
        return 1;
    }
    let (dx_sample, dy_sample) = if n_sample as f64 >= n_pix_use {
        n_sample = n_pix_use as i32;
        (1, 1)
    } else {
        // `fracY` is a `float` in the source, so the double `sqrt` rounds back
        // to single before `1. / fracY` widens it again.  `B3DNINT` is
        // `floor(a + 0.5)` and `B3DCLAMP` is `B3DMAX(minv, B3DMIN(maxv, val))`.
        let frac_y = (n_sample as f64 / n_pix_use).sqrt() as f32;
        let mut dy = (1.0 / frac_y as f64 + 0.5).floor() as i32;
        dy = 1.max((ny_use / 4).min(dy));
        let ny_sample = ny_use / dy;
        let ratio = n_sample as f32 / ny_sample as f32;
        let denom = (if 1.0_f64 > ratio as f64 {
            1.0_f64
        } else {
            ratio as f64
        }) as f32;
        let mut dx = ((nx_use as f32 / denom) as f64 + 0.5).floor() as i32;
        dx = 1.max((nx_use / 4).min(dx));
        // `samplemeansd.c:432`: the source runs this only in the subsampling
        // branch, never when every pixel is taken.
        while (ny_use / dy + 1) * (nx_use / dx + 1) > max_samples {
            dx = nx_use.min(dx + 1);
            dy = ny_use.min(dy + 1);
        }
        (dx, dy)
    };
    let mut ind_samp = 0_usize;
    let mut j = iy_start;
    while j < iy_start + ny_use {
        let xstart = (j % 17) % dx_sample;
        let mut i = ix_start + xstart;
        match mode {
            0 => {
                while i < ix_start + nx_use {
                    let value = image[(j * nx + i) as usize];
                    if value as i32 != ifill {
                        samples[ind_samp] = value as f32;
                        ind_samp += 1;
                    }
                    i += dx_sample;
                }
            }
            1 => {
                while i < ix_start + nx_use {
                    let k = 2 * (j * nx + i) as usize;
                    let value = i16::from_ne_bytes([image[k], image[k + 1]]);
                    if value as i32 != ifill {
                        samples[ind_samp] = value as f32;
                        ind_samp += 1;
                    }
                    i += dx_sample;
                }
            }
            6 => {
                while i < ix_start + nx_use {
                    let k = 2 * (j * nx + i) as usize;
                    let value = u16::from_ne_bytes([image[k], image[k + 1]]);
                    if value as i32 != ifill {
                        samples[ind_samp] = value as f32;
                        ind_samp += 1;
                    }
                    i += dx_sample;
                }
            }
            2 => {
                while i < ix_start + nx_use {
                    let k = 4 * (j * nx + i) as usize;
                    let value =
                        f32::from_ne_bytes([image[k], image[k + 1], image[k + 2], image[k + 3]]);
                    if (value - fill_to_exclude).abs() > 1.0e-5 {
                        samples[ind_samp] = value;
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

#[cfg(test)]
mod tests {
    use super::{
        get_sample_of_array, sample_min_max_mean_sd, sample_min_max_mean_sd_fortran,
        type_for_sample_mean,
    };
    #[test]
    fn byte_and_rgb_statistics_match_source_rules() {
        let byte = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let lines = [&byte[0..], &byte[3..], &byte[6..]];
        let mut mean = 0.;
        let mut sd = 0.;
        let mut low = 0.;
        let mut high = 0.;
        assert_eq!(
            sample_min_max_mean_sd(
                Some(&lines),
                0,
                3,
                3,
                1.,
                0,
                0,
                3,
                3,
                Some(&mut mean),
                Some(&mut sd),
                Some(&mut low),
                Some(&mut high),
            ),
            0
        );
        assert_eq!((mean, low, high), (5., 1., 9.));
        assert!((sd - 2.738613).abs() < 1.0e-5);
        let rgb = [
            100_u8, 0, 0, 0, 100, 0, 0, 0, 100, 100, 100, 100, 10, 20, 30, 30, 20, 10,
        ];
        let rgb_lines = [&rgb[0..], &rgb[9..]];
        assert_eq!(
            sample_min_max_mean_sd(
                Some(&rgb_lines),
                8,
                3,
                2,
                1.,
                0,
                0,
                3,
                2,
                Some(&mut mean),
                None,
                Some(&mut low),
                Some(&mut high),
            ),
            0
        );
        assert!((mean - 40.0).abs() < 1.0e-5);
        assert!((low - 11.0).abs() < 1.0e-5);
        assert!((high - 100.).abs() < 1.0e-5);
    }
    #[test]
    fn fortran_wrapper_and_array_sampling_obey_source_modes() {
        let image: Vec<u8> = [1_f32, 2., 3., 4., 5., 6., 7., 8., 9.]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let n = 3;
        let sample = 1.;
        let zero = 0;
        let mut mean = 0.;
        let mut sd = 0.;
        let mut dmin = 0.;
        let mut dmax = 0.;
        assert_eq!(
            sample_min_max_mean_sd_fortran(
                &image, &n, &n, &sample, &zero, &zero, &n, &n, 2, &mut mean, &mut sd, &mut dmin,
                &mut dmax,
            ),
            0
        );
        assert_eq!(mean, 5.);
        assert!((sd - 2.738613).abs() < 1.0e-5);
        let data = [1_u8, 2, 9, 4, 5, 6, 7, 8, 9];
        let mut out = [0_f32; 9];
        let mut count = 0;
        assert_eq!(
            get_sample_of_array(&data, 0, 3, 3, 1., 0, 0, 3, 3, 9., &mut out, 9, &mut count),
            0
        );
        // Every pixel is sampled because `nSample >= nPixUse`; the source runs
        // its `maxSamples` thinning loop only in the subsampling branch
        // (`samplemeansd.c:432`).  Verified against the native routine:
        // `rc=0 n=7: 1 2 4 5 6 7 8`.
        assert_eq!(count, 7);
        assert_eq!(&out[..7], &[1., 2., 4., 5., 6., 7., 8.]);
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
