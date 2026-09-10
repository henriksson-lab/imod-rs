//! Translation of `IMOD/libcfshr/reduce_by_binning.c`.
#![allow(dead_code)]

pub const SLICE_MODE_BYTE: i32 = 0;
pub const SLICE_MODE_SHORT: i32 = 1;
pub const SLICE_MODE_FLOAT: i32 = 2;
pub const SLICE_MODE_USHORT: i32 = 6;
pub const SLICE_MODE_RGB: i32 = 16;

/// `extractAndBinIntoArray` (`reduce_by_binning.c:53`).
pub unsafe fn extract_and_bin_into_array(
    array: *mut core::ffi::c_void,
    typ: i32,
    nx_dim: i32,
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    nbin: i32,
    bray: *mut core::ffi::c_void,
    mut nx_bdim: i32,
    mut bx_offset: i32,
    by_offset: i32,
    keep_byte: i32,
    nxr: *mut i32,
    nyr: *mut i32,
) -> i32 {
    unsafe {
        if typ != SLICE_MODE_BYTE
            && typ != SLICE_MODE_SHORT
            && typ != SLICE_MODE_FLOAT
            && typ != SLICE_MODE_USHORT
            && typ != SLICE_MODE_RGB
        {
            return 1;
        }
        let nxin = x_end + 1 - x_start;
        let nyin = y_end + 1 - y_start;
        let nxout = nxin / nbin;
        let nyout = nyin / nbin;
        if nx_bdim == 0 {
            nx_bdim = nxout;
        }
        if nxin > nx_dim {
            return 2;
        }
        if nxout + bx_offset > nx_bdim {
            return 3;
        }
        bx_offset += by_offset * nx_bdim;
        let ixofs = (nxin % nbin) / 2;
        let iyofs = (nyin % nbin) / 2;
        let npix = nbin * nbin;
        // The C source has dedicated 2/3/4 loops only to optimize this exact
        // row-major accumulation; this loop retains their conversion/rounding rules.
        for oy in 0..nyout {
            for ox in 0..nxout {
                let output = (bx_offset + ox + oy * nx_bdim) as usize;
                if typ == SLICE_MODE_FLOAT {
                    let mut sum = 0_f32;
                    for iy in 0..nbin {
                        for ix in 0..nbin {
                            sum += *(array.cast::<f32>()).add(
                                (x_start
                                    + ixofs
                                    + ox * nbin
                                    + ix
                                    + nx_dim * (y_start + iyofs + oy * nbin + iy))
                                    as usize,
                            );
                        }
                    }
                    *bray.cast::<f32>().add(output) = sum / npix as f32;
                } else if typ == SLICE_MODE_SHORT {
                    let mut sum = npix / 2;
                    for iy in 0..nbin {
                        for ix in 0..nbin {
                            sum += *(array.cast::<i16>()).add(
                                (x_start
                                    + ixofs
                                    + ox * nbin
                                    + ix
                                    + nx_dim * (y_start + iyofs + oy * nbin + iy))
                                    as usize,
                            ) as i32;
                        }
                    }
                    *bray.cast::<i16>().add(output) = (sum / npix) as i16;
                } else if typ == SLICE_MODE_USHORT {
                    let mut sum = npix / 2;
                    for iy in 0..nbin {
                        for ix in 0..nbin {
                            sum += *(array.cast::<u16>()).add(
                                (x_start
                                    + ixofs
                                    + ox * nbin
                                    + ix
                                    + nx_dim * (y_start + iyofs + oy * nbin + iy))
                                    as usize,
                            ) as i32;
                        }
                    }
                    *bray.cast::<u16>().add(output) = (sum / npix) as u16;
                } else if typ == SLICE_MODE_BYTE {
                    let mut sum = 0_i32;
                    for iy in 0..nbin {
                        for ix in 0..nbin {
                            sum += *(array.cast::<u8>()).add(
                                (x_start
                                    + ixofs
                                    + ox * nbin
                                    + ix
                                    + nx_dim * (y_start + iyofs + oy * nbin + iy))
                                    as usize,
                            ) as i32;
                        }
                    }
                    if keep_byte > 0 {
                        *bray.cast::<u8>().add(output) = ((sum + npix / 2) / npix) as u8;
                    } else if keep_byte == 0 {
                        *bray.cast::<i16>().add(output) = sum as i16;
                    } else {
                        *bray.cast::<i16>().add(output) = ((sum + npix / 2) / npix) as i16;
                    }
                } else {
                    let (mut red, mut green, mut blue) = (0_i32, 0_i32, 0_i32);
                    for iy in 0..nbin {
                        for ix in 0..nbin {
                            let input = 3
                                * (x_start
                                    + ixofs
                                    + ox * nbin
                                    + ix
                                    + nx_dim * (y_start + iyofs + oy * nbin + iy));
                            red += *array.cast::<u8>().add(input as usize) as i32;
                            green += *array.cast::<u8>().add((input + 1) as usize) as i32;
                            blue += *array.cast::<u8>().add((input + 2) as usize) as i32;
                        }
                    }
                    if keep_byte > 0 {
                        let out = 3 * output;
                        *bray.cast::<u8>().add(out) = ((red + npix / 2) / npix) as u8;
                        *bray.cast::<u8>().add(out + 1) = ((green + npix / 2) / npix) as u8;
                        *bray.cast::<u8>().add(out + 2) = ((blue + npix / 2) / npix) as u8;
                    } else if keep_byte == 0 {
                        *bray.cast::<i16>().add(output) = (red + green + blue) as i16;
                    } else {
                        *bray.cast::<i16>().add(output) =
                            ((red + green + blue + npix / 2) / npix) as i16;
                    }
                }
            }
        }
        *nxr = nxout;
        *nyr = nyout;
        0
    }
}

pub unsafe fn extract_with_binning(
    array: *mut core::ffi::c_void,
    typ: i32,
    nx_dim: i32,
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    nbin: i32,
    bray: *mut core::ffi::c_void,
    keep_byte: i32,
    nxr: *mut i32,
    nyr: *mut i32,
) -> i32 {
    unsafe {
        extract_and_bin_into_array(
            array, typ, nx_dim, x_start, x_end, y_start, y_end, nbin, bray, 0, 0, 0, keep_byte,
            nxr, nyr,
        )
    }
}
pub unsafe fn extractwithbinning(
    array: *mut f32,
    nx_dim: *const i32,
    xs: *const i32,
    xe: *const i32,
    ys: *const i32,
    ye: *const i32,
    nbin: *const i32,
    bray: *mut f32,
    keep_byte: *const i32,
    nxr: *mut i32,
    nyr: *mut i32,
) -> i32 {
    unsafe {
        extract_with_binning(
            array.cast(),
            SLICE_MODE_FLOAT,
            *nx_dim,
            *xs,
            *xe,
            *ys,
            *ye,
            *nbin,
            bray.cast(),
            *keep_byte,
            nxr,
            nyr,
        )
    }
}
pub unsafe fn repack_float_image(
    bray: *mut core::ffi::c_void,
    array: *mut core::ffi::c_void,
    nxin: i32,
    xs: i32,
    xe: i32,
    ys: i32,
    ye: i32,
) {
    unsafe {
        let mut nxr = 0;
        let mut nyr = 0;
        extract_with_binning(
            array,
            SLICE_MODE_FLOAT,
            nxin,
            xs,
            xe,
            ys,
            ye,
            1,
            bray,
            1,
            &mut nxr,
            &mut nyr,
        );
    }
}
pub unsafe fn irepak(
    bray: *mut core::ffi::c_void,
    array: *mut core::ffi::c_void,
    nxin: *const i32,
    _nyin: *const i32,
    xs: *const i32,
    xe: *const i32,
    ys: *const i32,
    ye: *const i32,
) {
    unsafe { repack_float_image(bray, array, *nxin, *xs, *xe, *ys, *ye) }
}
pub unsafe fn reduce_by_binning(
    array: *mut core::ffi::c_void,
    typ: i32,
    nxin: i32,
    nyin: i32,
    nbin: i32,
    bray: *mut core::ffi::c_void,
    keep_byte: i32,
    nxr: *mut i32,
    nyr: *mut i32,
) -> i32 {
    unsafe {
        extract_with_binning(
            array,
            typ,
            nxin,
            0,
            nxin - 1,
            0,
            nyin - 1,
            nbin,
            bray,
            keep_byte,
            nxr,
            nyr,
        )
    }
}
pub unsafe fn reduce_by_binning_f(
    array: *mut f32,
    nx: *mut i32,
    ny: *mut i32,
    nbin: *const i32,
    bray: *mut f32,
    nxr: *mut i32,
    nyr: *mut i32,
) {
    unsafe {
        reduce_by_binning(
            array.cast(),
            SLICE_MODE_FLOAT,
            *nx,
            *ny,
            *nbin,
            bray.cast(),
            0,
            nxr,
            nyr,
        );
    }
}
pub unsafe fn bin_into_slice(
    array: *mut f32,
    nx_dim: i32,
    bray: *mut f32,
    nx_bin: i32,
    ny_bin: i32,
    bin_x: i32,
    bin_y: i32,
    z_weight: f32,
) {
    unsafe {
        let factor = z_weight / (bin_x * bin_y) as f32;
        for oy in 0..ny_bin {
            for by in 0..bin_y {
                for ox in 0..nx_bin {
                    let mut sum = 0.;
                    for bx in 0..bin_x {
                        sum += *array.add((ox * bin_x + bx + nx_dim * (oy * bin_y + by)) as usize);
                    }
                    *bray.add((ox + nx_bin * oy) as usize) += sum * factor;
                }
            }
        }
    }
}
pub unsafe fn binintoslice(
    array: *mut f32,
    nx_dim: *const i32,
    bray: *mut f32,
    nx_bin: *const i32,
    ny_bin: *const i32,
    bin_x: *const i32,
    bin_y: *const i32,
    z_weight: *const f32,
) {
    unsafe {
        bin_into_slice(
            array, *nx_dim, bray, *nx_bin, *ny_bin, *bin_x, *bin_y, *z_weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_binning_centers_remainder_and_averages() {
        let mut input = (0..35).map(|x| x as f32).collect::<Vec<_>>();
        let mut out = [0_f32; 6];
        let (mut x, mut y) = (0, 0);
        unsafe {
            assert_eq!(
                reduce_by_binning(
                    input.as_mut_ptr().cast(),
                    SLICE_MODE_FLOAT,
                    7,
                    5,
                    2,
                    out.as_mut_ptr().cast(),
                    0,
                    &mut x,
                    &mut y
                ),
                0
            )
        };
        assert_eq!((x, y), (3, 2));
        assert_eq!(out[0], (0. + 1. + 7. + 8.) / 4.);
    }
    #[test]
    fn byte_sum_and_average_match_keep_byte_modes() {
        let mut input = vec![1_u8, 2, 3, 4];
        let mut sum = [0_i16; 1];
        let mut avg = [0_u8; 1];
        let (mut x, mut y) = (0, 0);
        unsafe {
            extract_with_binning(
                input.as_mut_ptr().cast(),
                SLICE_MODE_BYTE,
                2,
                0,
                1,
                0,
                1,
                2,
                sum.as_mut_ptr().cast(),
                0,
                &mut x,
                &mut y,
            );
            extract_with_binning(
                input.as_mut_ptr().cast(),
                SLICE_MODE_BYTE,
                2,
                0,
                1,
                0,
                1,
                2,
                avg.as_mut_ptr().cast(),
                1,
                &mut x,
                &mut y,
            );
        };
        assert_eq!(sum[0], 10);
        assert_eq!(avg[0], 3);
    }
}
