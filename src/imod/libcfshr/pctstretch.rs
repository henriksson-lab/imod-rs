//! Translation of `IMOD/libcfshr/pctstretch.c`.
#![allow(dead_code)]

pub const SLICE_MODE_BYTE: i32 = 0;
pub const SLICE_MODE_SHORT: i32 = 1;
pub const SLICE_MODE_FLOAT: i32 = 2;
pub const SLICE_MODE_USHORT: i32 = 6;

/// Original `percentileStretch` (`pctstretch.c:35`).  The source's
/// `unsigned char **image` line-pointer array is a slice of line slices here;
/// each line holds the raw bytes of one image row and is reinterpreted in the
/// host's byte order according to `data_type`, exactly as the C casts of
/// `image` to `b3dInt16 **`, `b3dUInt16 **` and `float **` do.
pub fn percentile_stretch(
    image: &[&[u8]],
    data_type: i32,
    nx: i32,
    ny: i32,
    sample: f32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    pct_lo: f32,
    pct_hi: f32,
    scale_lo: &mut f32,
    scale_hi: &mut f32,
) -> i32 {
    let n_pix_use = nx_use as f64 * ny_use as f64;
    let mut n_sample = (sample as f64 * n_pix_use) as i32;
    if n_sample as f64 >= n_pix_use {
        n_sample = n_pix_use as i32;
    }
    if nx_use < 2 || ny_use < 2 || n_sample < 5 {
        return 1;
    }
    let mut dx_sample = (n_pix_use / n_sample as f64) as i32;
    if dx_sample == 0 {
        dx_sample = 1;
    }
    if dx_sample > 5 && n_pix_use < 2.0e9 && dx_sample % 2 == (n_pix_use as i32) % 2 {
        dx_sample -= 1;
    }
    n_sample = (n_pix_use / dx_sample as f64) as i32;
    // `pctstretch.c:66-67`: `0.01` is a double literal, so `pctLo` and
    // `nSample` widen and the whole product is evaluated in double before the
    // truncation.  Computing it in f32 puts `0.01f * 5.f * 700` one ulp below
    // 35 and truncates to 34, which selects a different histogram bin.
    let n_lo = ((0.01 * pct_lo as f64 * n_sample as f64) as i32).max(1);
    let n_hi = ((0.01 * pct_hi as f64 * n_sample as f64) as i32).max(1);
    let (nbins, base, factor, shift, f_base, f_factor) = match data_type {
        SLICE_MODE_BYTE => (256, 0, 1, 0, 0., 0.),
        SLICE_MODE_SHORT => (16384, 32768, 4, 2, 0., 0.),
        SLICE_MODE_USHORT => (16384, 0, 4, 2, 0., 0.),
        SLICE_MODE_FLOAT => {
            let mut min = 1.0e38_f32;
            let mut max = -1.0e38_f32;
            let mut ix = 0;
            let mut iy = 0;
            for _ in 0..n_sample {
                let line = image[(iy + iy_start) as usize];
                let offset = (ix + ix_start) as usize * 4;
                let value = f32::from_ne_bytes(line[offset..offset + 4].try_into().unwrap());
                min = min.min(value);
                max = max.max(value);
                ix += dx_sample;
                while ix >= nx_use {
                    ix -= nx_use;
                    iy += 1;
                }
            }
            let pad = if 0.01 * (max - min) == 0. {
                1.
            } else {
                0.01 * (max - min)
            };
            let range = 2. * pad + max - min;
            (16384, 0, 1, 2, -(min - pad), 16384. / range)
        }
        _ => return 2,
    };
    let mut hist = Vec::new();
    if hist.try_reserve_exact(nbins).is_err() {
        return 3;
    }
    hist.resize(nbins, 0_i32);
    let special_full_scan = dx_sample == 1 && ix_start == 0 && iy_start == 0;
    let histogram_samples = if special_full_scan {
        ny_use * (nx_use - 1)
    } else {
        n_sample
    };
    let mut ix = if special_full_scan { 1 } else { 0 };
    let mut iy = 0;
    for _ in 0..histogram_samples {
        let line = image[(iy + iy_start) as usize];
        let index = (ix + ix_start) as usize;
        let value = match data_type {
            SLICE_MODE_BYTE => line[index] as i32,
            SLICE_MODE_SHORT => {
                i16::from_ne_bytes(line[index * 2..index * 2 + 2].try_into().unwrap()) as i32
            }
            SLICE_MODE_USHORT => {
                u16::from_ne_bytes(line[index * 2..index * 2 + 2].try_into().unwrap()) as i32
            }
            _ => {
                ((f32::from_ne_bytes(line[index * 4..index * 4 + 4].try_into().unwrap()) + f_base)
                    * f_factor) as i32
            }
        };
        let bin = match data_type {
            SLICE_MODE_SHORT => (value + 32768) >> 2,
            SLICE_MODE_USHORT => value >> 2,
            SLICE_MODE_FLOAT => value,
            _ => value,
        };
        hist[bin as usize] += 1;
        ix += dx_sample;
        while ix >= nx_use {
            ix -= nx_use;
            iy += 1;
            if special_full_scan {
                ix = 1;
            }
        }
    }
    let mut cumulative = 0;
    for (index, count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative >= n_lo {
            *scale_lo = if data_type == SLICE_MODE_FLOAT {
                index as f32 / f_factor - f_base
            } else {
                (index as i32 * factor - base) as f32
            };
            break;
        }
    }
    cumulative = 0;
    for index in (0..nbins).rev() {
        cumulative += hist[index];
        if cumulative >= n_hi {
            *scale_hi = if data_type == SLICE_MODE_FLOAT {
                index as f32 / f_factor - f_base
            } else {
                (index as i32 * factor - base + (1 << shift) - 1) as f32
            };
            break;
        }
    }
    if *scale_lo >= *scale_hi {
        let mean = 0.5 * (*scale_lo + *scale_hi);
        *scale_lo = mean;
        *scale_hi = mean;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_sampling_and_invalid_input_follow_source() {
        let row0 = [0_u8, 1, 2, 3];
        let row1 = [4_u8, 5, 6, 7];
        let row2 = [8_u8, 9, 10, 11];
        let row3 = [12_u8, 13, 14, 15];
        let rows: [&[u8]; 4] = [&row0, &row1, &row2, &row3];
        let mut lo = 0.;
        let mut hi = 0.;
        assert_eq!(
            percentile_stretch(
                &rows,
                SLICE_MODE_BYTE,
                4,
                4,
                1.,
                0,
                0,
                4,
                4,
                1.,
                1.,
                &mut lo,
                &mut hi,
            ),
            0
        );
        assert!(lo <= hi);
        assert_eq!(
            percentile_stretch(&rows, 99, 4, 4, 1., 0, 0, 4, 4, 1., 1., &mut lo, &mut hi,),
            2
        );
    }

    /// `nLo`/`nHi` are `(int)(0.01 * pct * nSample)` with `0.01` a *double*
    /// literal (`pctstretch.c:66-67`).  Evaluating the product in f32 makes
    /// `0.01 * 5 * 700` land one ulp below 35 and truncate to 34, which stops
    /// the top-down cumulative one bin early.  Verified against
    /// `libcfshr.so`'s `percentileStretch`, which returns 249 here; the f32
    /// form returned 250.
    #[test]
    fn high_percentile_count_is_computed_in_double_like_the_source() {
        let (nx, ny) = (100usize, 7usize);
        let mut buf = vec![0_u8; nx * ny];
        for x in 1..=34 {
            buf[x] = 250;
        }
        buf[35] = 249;
        let lines: Vec<&[u8]> = (0..ny).map(|y| &buf[y * nx..(y + 1) * nx]).collect();
        let mut lo = -1.5;
        let mut hi = 2.5;
        assert_eq!(
            percentile_stretch(
                &lines,
                SLICE_MODE_BYTE,
                nx as i32,
                ny as i32,
                1.0,
                0,
                0,
                nx as i32,
                ny as i32,
                0.1,
                5.0,
                &mut lo,
                &mut hi,
            ),
            0
        );
        assert_eq!((lo, hi), (0., 249.));
    }
}
