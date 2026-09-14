//! Translation of `IMOD/libcfshr/reduce_by_binning.c`.
#![allow(dead_code)]

pub const SLICE_MODE_BYTE: i32 = 0;
pub const SLICE_MODE_SHORT: i32 = 1;
pub const SLICE_MODE_FLOAT: i32 = 2;
pub const SLICE_MODE_USHORT: i32 = 6;
pub const SLICE_MODE_RGB: i32 = 16;

/// `extractAndBinIntoArray` (`reduce_by_binning.c:53`).
///
/// The source's `void *array` and `void *brray` are the raw byte views of the
/// input and output; `type` and `keepByte` alone decide how an element is read
/// and written, exactly as in the C.  The source documents that `brray` may be
/// the same buffer as `array`; Rust cannot express that with one shared and one
/// mutable borrow, so an in-place caller passes a copy of the input.
pub fn extract_and_bin_into_array(
    array: &[u8],
    typ: i32,
    nx_dim: i32,
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    nbin: i32,
    bray: &mut [u8],
    mut nx_bdim: i32,
    mut bx_offset: i32,
    by_offset: i32,
    keep_byte: i32,
    nxr: &mut i32,
    nyr: &mut i32,
) -> i32 {
    let nxin = x_end + 1 - x_start;
    let nyin = y_end + 1 - y_start;
    let nbinsq = nbin * nbin;
    let nxout = nxin / nbin;
    let nyout = nyin / nbin;
    let ixofs = (nxin % nbin) / 2;
    let iyofs = (nyin % nbin) / 2;
    let bin234ok = if typ == SLICE_MODE_RGB || (typ == SLICE_MODE_BYTE && keep_byte < 0) {
        0
    } else {
        1
    };

    if typ != SLICE_MODE_BYTE
        && typ != SLICE_MODE_SHORT
        && typ != SLICE_MODE_USHORT
        && typ != SLICE_MODE_FLOAT
        && typ != SLICE_MODE_RGB
    {
        return 1;
    }
    if nx_bdim == 0 {
        nx_bdim = nxout;
    }
    if nxin > nx_dim {
        return 2;
    }
    if nxout + bx_offset > nx_bdim {
        return 3;
    }

    // Get the bytes per pixel
    let pix_size: usize = if typ == SLICE_MODE_BYTE {
        1
    } else if typ == SLICE_MODE_FLOAT {
        4
    } else if typ == SLICE_MODE_RGB {
        3
    } else {
        2
    };
    let mut out_pix_size = pix_size;
    if (typ == SLICE_MODE_BYTE || typ == SLICE_MODE_RGB) && keep_byte <= 0 {
        out_pix_size = 2;
    }

    // Advance array pointer to the beginning of the data and incorporate Y offset into X
    let abase = (x_start as usize + y_start as usize * nx_dim as usize) * pix_size;
    bx_offset += by_offset * nx_bdim;

    if nbin == 1 && pix_size == out_pix_size {
        // Binning 1: copy the data line by line
        for iy in 0..nyout as usize {
            let cline1 = abase + pix_size * iy * nx_dim as usize;
            let bdata = pix_size * (bx_offset as usize + iy * nx_bdim as usize);
            for ix in 0..nxout as usize * pix_size {
                bray[bdata + ix] = array[cline1 + ix];
            }
        }
    } else if nbin == 2 && bin234ok != 0 {
        // Binning by 2
        match typ {
            SLICE_MODE_BYTE => {
                for iy in 0..nyout as usize {
                    let mut cline1 = abase + 2 * iy * nx_dim as usize;
                    let mut cline2 = cline1 + nx_dim as usize;
                    if keep_byte != 0 {
                        let mut bdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let sum = 2
                                + array[cline1] as i32
                                + array[cline1 + 1] as i32
                                + array[cline2] as i32
                                + array[cline2 + 1] as i32;
                            bray[bdata] = (sum / 4) as u8;
                            bdata += 1;
                            cline1 += 2;
                            cline2 += 2;
                        }
                    } else {
                        let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let sum = array[cline1] as i32
                                + array[cline1 + 1] as i32
                                + array[cline2] as i32
                                + array[cline2 + 1] as i32;
                            bray[2 * sdata..2 * sdata + 2]
                                .copy_from_slice(&(sum as i16).to_ne_bytes());
                            sdata += 1;
                            cline1 += 2;
                            cline2 += 2;
                        }
                    }
                }
            }
            SLICE_MODE_SHORT => {
                for iy in 0..nyout as usize {
                    let mut sline1 = abase / 2 + 2 * iy * nx_dim as usize;
                    let mut sline2 = sline1 + nx_dim as usize;
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let sum = 2
                            + i16::from_ne_bytes([array[2 * sline1], array[2 * sline1 + 1]]) as i32
                            + i16::from_ne_bytes([array[2 * sline1 + 2], array[2 * sline1 + 3]])
                                as i32
                            + i16::from_ne_bytes([array[2 * sline2], array[2 * sline2 + 1]]) as i32
                            + i16::from_ne_bytes([array[2 * sline2 + 2], array[2 * sline2 + 3]])
                                as i32;
                        sline1 += 2;
                        sline2 += 2;
                        bray[2 * sdata..2 * sdata + 2]
                            .copy_from_slice(&((sum / 4) as i16).to_ne_bytes());
                        sdata += 1;
                    }
                }
            }
            SLICE_MODE_USHORT => {
                for iy in 0..nyout as usize {
                    let mut usline1 = abase / 2 + 2 * iy * nx_dim as usize;
                    let mut usline2 = usline1 + nx_dim as usize;
                    let mut usdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let sum = 2
                            + u16::from_ne_bytes([array[2 * usline1], array[2 * usline1 + 1]])
                                as i32
                            + u16::from_ne_bytes([array[2 * usline1 + 2], array[2 * usline1 + 3]])
                                as i32
                            + u16::from_ne_bytes([array[2 * usline2], array[2 * usline2 + 1]])
                                as i32
                            + u16::from_ne_bytes([array[2 * usline2 + 2], array[2 * usline2 + 3]])
                                as i32;
                        usline1 += 2;
                        usline2 += 2;
                        bray[2 * usdata..2 * usdata + 2]
                            .copy_from_slice(&((sum / 4) as u16).to_ne_bytes());
                        usdata += 1;
                    }
                }
            }
            SLICE_MODE_FLOAT => {
                for iy in 0..nyout as usize {
                    let mut fline1 = abase / 4 + 2 * iy * nx_dim as usize;
                    let mut fline2 = fline1 + nx_dim as usize;
                    let mut fdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let fsum = f32::from_ne_bytes([
                            array[4 * fline1],
                            array[4 * fline1 + 1],
                            array[4 * fline1 + 2],
                            array[4 * fline1 + 3],
                        ]) + f32::from_ne_bytes([
                            array[4 * fline1 + 4],
                            array[4 * fline1 + 5],
                            array[4 * fline1 + 6],
                            array[4 * fline1 + 7],
                        ]) + f32::from_ne_bytes([
                            array[4 * fline2],
                            array[4 * fline2 + 1],
                            array[4 * fline2 + 2],
                            array[4 * fline2 + 3],
                        ]) + f32::from_ne_bytes([
                            array[4 * fline2 + 4],
                            array[4 * fline2 + 5],
                            array[4 * fline2 + 6],
                            array[4 * fline2 + 7],
                        ]);
                        fline1 += 2;
                        fline2 += 2;
                        bray[4 * fdata..4 * fdata + 4]
                            .copy_from_slice(&(fsum / 4.0_f32).to_ne_bytes());
                        fdata += 1;
                    }
                }
            }
            _ => {}
        }
    } else if nbin == 3 && bin234ok != 0 {
        // Binning by 3
        match typ {
            SLICE_MODE_BYTE => {
                for iy in 0..nyout as usize {
                    let mut cline1 =
                        abase + (3 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut cline2 = cline1 + nx_dim as usize;
                    let mut cline3 = cline2 + nx_dim as usize;
                    if keep_byte != 0 {
                        let mut bdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let sum = 4
                                + array[cline1] as i32
                                + array[cline1 + 1] as i32
                                + array[cline1 + 2] as i32
                                + array[cline2] as i32
                                + array[cline2 + 1] as i32
                                + array[cline2 + 2] as i32
                                + array[cline3] as i32
                                + array[cline3 + 1] as i32
                                + array[cline3 + 2] as i32;
                            cline1 += 3;
                            cline2 += 3;
                            cline3 += 3;
                            bray[bdata] = (sum / 9) as u8;
                            bdata += 1;
                        }
                    } else {
                        let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let sum = array[cline1] as i32
                                + array[cline1 + 1] as i32
                                + array[cline1 + 2] as i32
                                + array[cline2] as i32
                                + array[cline2 + 1] as i32
                                + array[cline2 + 2] as i32
                                + array[cline3] as i32
                                + array[cline3 + 1] as i32
                                + array[cline3 + 2] as i32;
                            bray[2 * sdata..2 * sdata + 2]
                                .copy_from_slice(&(sum as i16).to_ne_bytes());
                            sdata += 1;
                            cline1 += 3;
                            cline2 += 3;
                            cline3 += 3;
                        }
                    }
                }
            }
            SLICE_MODE_SHORT => {
                for iy in 0..nyout as usize {
                    let mut sline1 =
                        abase / 2 + (3 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut sline2 = sline1 + nx_dim as usize;
                    let mut sline3 = sline2 + nx_dim as usize;
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = 4;
                        for base in [sline1, sline2, sline3] {
                            for k in 0..3 {
                                sum += i16::from_ne_bytes([
                                    array[2 * (base + k)],
                                    array[2 * (base + k) + 1],
                                ]) as i32;
                            }
                        }
                        sline1 += 3;
                        sline2 += 3;
                        sline3 += 3;
                        bray[2 * sdata..2 * sdata + 2]
                            .copy_from_slice(&((sum / 9) as i16).to_ne_bytes());
                        sdata += 1;
                    }
                }
            }
            SLICE_MODE_USHORT => {
                for iy in 0..nyout as usize {
                    let mut usline1 =
                        abase / 2 + (3 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut usline2 = usline1 + nx_dim as usize;
                    let mut usline3 = usline2 + nx_dim as usize;
                    let mut usdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = 4;
                        for base in [usline1, usline2, usline3] {
                            for k in 0..3 {
                                sum += u16::from_ne_bytes([
                                    array[2 * (base + k)],
                                    array[2 * (base + k) + 1],
                                ]) as i32;
                            }
                        }
                        usline1 += 3;
                        usline2 += 3;
                        usline3 += 3;
                        bray[2 * usdata..2 * usdata + 2]
                            .copy_from_slice(&((sum / 9) as u16).to_ne_bytes());
                        usdata += 1;
                    }
                }
            }
            SLICE_MODE_FLOAT => {
                for iy in 0..nyout as usize {
                    let mut fline1 =
                        abase / 4 + (3 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut fline2 = fline1 + nx_dim as usize;
                    let mut fline3 = fline2 + nx_dim as usize;
                    let mut fdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut fsum = f32::from_ne_bytes([
                            array[4 * fline1],
                            array[4 * fline1 + 1],
                            array[4 * fline1 + 2],
                            array[4 * fline1 + 3],
                        ]);
                        for (n, base) in [fline1, fline2, fline3].iter().enumerate() {
                            for k in 0..3 {
                                if n == 0 && k == 0 {
                                    continue;
                                }
                                let o = 4 * (base + k);
                                fsum += f32::from_ne_bytes([
                                    array[o],
                                    array[o + 1],
                                    array[o + 2],
                                    array[o + 3],
                                ]);
                            }
                        }
                        fline1 += 3;
                        fline2 += 3;
                        fline3 += 3;
                        bray[4 * fdata..4 * fdata + 4]
                            .copy_from_slice(&(fsum / 9.0_f32).to_ne_bytes());
                        fdata += 1;
                    }
                }
            }
            _ => {}
        }
    } else if nbin == 4 && bin234ok != 0 {
        // Binning by 4
        match typ {
            SLICE_MODE_BYTE => {
                for iy in 0..nyout as usize {
                    let mut cline1 =
                        abase + (4 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut cline2 = cline1 + nx_dim as usize;
                    let mut cline3 = cline2 + nx_dim as usize;
                    let mut cline4 = cline3 + nx_dim as usize;
                    if keep_byte != 0 {
                        let mut bdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let mut sum = 8;
                            for base in [cline1, cline2, cline3, cline4] {
                                for k in 0..4 {
                                    sum += array[base + k] as i32;
                                }
                            }
                            cline1 += 4;
                            cline2 += 4;
                            cline3 += 4;
                            cline4 += 4;
                            bray[bdata] = (sum / 16) as u8;
                            bdata += 1;
                        }
                    } else {
                        let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                        for _ix in 0..nxout {
                            let mut sum = 0;
                            for base in [cline1, cline2, cline3, cline4] {
                                for k in 0..4 {
                                    sum += array[base + k] as i32;
                                }
                            }
                            bray[2 * sdata..2 * sdata + 2]
                                .copy_from_slice(&(sum as i16).to_ne_bytes());
                            sdata += 1;
                            cline1 += 4;
                            cline2 += 4;
                            cline3 += 4;
                            cline4 += 4;
                        }
                    }
                }
            }
            SLICE_MODE_SHORT => {
                for iy in 0..nyout as usize {
                    let mut sline1 =
                        abase / 2 + (4 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut sline2 = sline1 + nx_dim as usize;
                    let mut sline3 = sline2 + nx_dim as usize;
                    let mut sline4 = sline3 + nx_dim as usize;
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = 8;
                        for base in [sline1, sline2, sline3, sline4] {
                            for k in 0..4 {
                                sum += i16::from_ne_bytes([
                                    array[2 * (base + k)],
                                    array[2 * (base + k) + 1],
                                ]) as i32;
                            }
                        }
                        sline1 += 4;
                        sline2 += 4;
                        sline3 += 4;
                        sline4 += 4;
                        bray[2 * sdata..2 * sdata + 2]
                            .copy_from_slice(&((sum / 16) as i16).to_ne_bytes());
                        sdata += 1;
                    }
                }
            }
            SLICE_MODE_USHORT => {
                for iy in 0..nyout as usize {
                    let mut usline1 =
                        abase / 2 + (4 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut usline2 = usline1 + nx_dim as usize;
                    let mut usline3 = usline2 + nx_dim as usize;
                    let mut usline4 = usline3 + nx_dim as usize;
                    let mut usdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = 8;
                        for base in [usline1, usline2, usline3, usline4] {
                            for k in 0..4 {
                                sum += u16::from_ne_bytes([
                                    array[2 * (base + k)],
                                    array[2 * (base + k) + 1],
                                ]) as i32;
                            }
                        }
                        usline1 += 4;
                        usline2 += 4;
                        usline3 += 4;
                        usline4 += 4;
                        bray[2 * usdata..2 * usdata + 2]
                            .copy_from_slice(&((sum / 16) as u16).to_ne_bytes());
                        usdata += 1;
                    }
                }
            }
            SLICE_MODE_FLOAT => {
                for iy in 0..nyout as usize {
                    let mut fline1 =
                        abase / 4 + (4 * iy + iyofs as usize) * nx_dim as usize + ixofs as usize;
                    let mut fline2 = fline1 + nx_dim as usize;
                    let mut fline3 = fline2 + nx_dim as usize;
                    let mut fline4 = fline3 + nx_dim as usize;
                    let mut fdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut fsum = f32::from_ne_bytes([
                            array[4 * fline1],
                            array[4 * fline1 + 1],
                            array[4 * fline1 + 2],
                            array[4 * fline1 + 3],
                        ]);
                        for (n, base) in [fline1, fline2, fline3, fline4].iter().enumerate() {
                            for k in 0..4 {
                                if n == 0 && k == 0 {
                                    continue;
                                }
                                let o = 4 * (base + k);
                                fsum += f32::from_ne_bytes([
                                    array[o],
                                    array[o + 1],
                                    array[o + 2],
                                    array[o + 3],
                                ]);
                            }
                        }
                        fline1 += 4;
                        fline2 += 4;
                        fline3 += 4;
                        fline4 += 4;
                        bray[4 * fdata..4 * fdata + 4]
                            .copy_from_slice(&(fsum / 16.0_f32).to_ne_bytes());
                        fdata += 1;
                    }
                }
            }
            _ => {}
        }
    } else {
        // Bin by arbitrary number or bin RGB
        match typ {
            SLICE_MODE_BYTE => {
                for iy in 0..nyout as usize {
                    let mut cline1 = abase
                        + (nbin as usize * iy + iyofs as usize) * nx_dim as usize
                        + ixofs as usize;
                    let mut bdata = bx_offset as usize + iy * nx_bdim as usize;
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = 0;
                        let mut cline2 = cline1;
                        for _j in 0..nbin {
                            for i in 0..nbin as usize {
                                sum += array[cline2 + i] as i32;
                            }
                            cline2 += nx_dim as usize;
                        }
                        if keep_byte > 0 {
                            bray[bdata] = ((sum + nbinsq / 2) / nbinsq) as u8;
                            bdata += 1;
                        } else if keep_byte == 0 {
                            bray[2 * sdata..2 * sdata + 2]
                                .copy_from_slice(&(sum as i16).to_ne_bytes());
                            sdata += 1;
                        } else {
                            bray[2 * sdata..2 * sdata + 2].copy_from_slice(
                                &(((sum + nbinsq / 2) / nbinsq) as i16).to_ne_bytes(),
                            );
                            sdata += 1;
                        }
                        cline1 += nbin as usize;
                    }
                }
            }
            SLICE_MODE_SHORT => {
                for iy in 0..nyout as usize {
                    let mut sline1 = abase / 2
                        + (nbin as usize * iy + iyofs as usize) * nx_dim as usize
                        + ixofs as usize;
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = nbinsq / 2;
                        let mut sline2 = sline1;
                        for _j in 0..nbin {
                            for i in 0..nbin as usize {
                                sum += i16::from_ne_bytes([
                                    array[2 * (sline2 + i)],
                                    array[2 * (sline2 + i) + 1],
                                ]) as i32;
                            }
                            sline2 += nx_dim as usize;
                        }
                        bray[2 * sdata..2 * sdata + 2]
                            .copy_from_slice(&((sum / nbinsq) as i16).to_ne_bytes());
                        sdata += 1;
                        sline1 += nbin as usize;
                    }
                }
            }
            SLICE_MODE_USHORT => {
                for iy in 0..nyout as usize {
                    let mut usline1 = abase / 2
                        + (nbin as usize * iy + iyofs as usize) * nx_dim as usize
                        + ixofs as usize;
                    let mut usdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut sum = nbinsq / 2;
                        let mut usline2 = usline1;
                        for _j in 0..nbin {
                            for i in 0..nbin as usize {
                                sum += u16::from_ne_bytes([
                                    array[2 * (usline2 + i)],
                                    array[2 * (usline2 + i) + 1],
                                ]) as i32;
                            }
                            usline2 += nx_dim as usize;
                        }
                        bray[2 * usdata..2 * usdata + 2]
                            .copy_from_slice(&((sum / nbinsq) as u16).to_ne_bytes());
                        usdata += 1;
                        usline1 += nbin as usize;
                    }
                }
            }
            SLICE_MODE_FLOAT => {
                for iy in 0..nyout as usize {
                    let mut fline1 = abase / 4
                        + (nbin as usize * iy + iyofs as usize) * nx_dim as usize
                        + ixofs as usize;
                    let mut fdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let mut fsum = 0.0_f32;
                        let mut fline2 = fline1;
                        for _j in 0..nbin {
                            for i in 0..nbin as usize {
                                let o = 4 * (fline2 + i);
                                fsum += f32::from_ne_bytes([
                                    array[o],
                                    array[o + 1],
                                    array[o + 2],
                                    array[o + 3],
                                ]);
                            }
                            fline2 += nx_dim as usize;
                        }
                        bray[4 * fdata..4 * fdata + 4]
                            .copy_from_slice(&(fsum / nbinsq as f32).to_ne_bytes());
                        fdata += 1;
                        fline1 += nbin as usize;
                    }
                }
            }
            SLICE_MODE_RGB => {
                for iy in 0..nyout as usize {
                    let mut cline1 = abase
                        + 3 * ((nbin as usize * iy + iyofs as usize) * nx_dim as usize
                            + ixofs as usize);
                    let mut bdata = 3 * (bx_offset as usize + iy * nx_bdim as usize);
                    let mut sdata = bx_offset as usize + iy * nx_bdim as usize;
                    for _ix in 0..nxout {
                        let (mut red, mut green, mut blue) = (0_i32, 0_i32, 0_i32);
                        let mut cline2 = cline1;
                        for _j in 0..nbin {
                            for i in 0..nbin as usize {
                                red += array[cline2 + 3 * i] as i32;
                                green += array[cline2 + 3 * i + 1] as i32;
                                blue += array[cline2 + 3 * i + 2] as i32;
                            }
                            cline2 += nx_dim as usize * 3;
                        }
                        if keep_byte > 0 {
                            bray[bdata] = ((red + nbinsq / 2) / nbinsq) as u8;
                            bray[bdata + 1] = ((green + nbinsq / 2) / nbinsq) as u8;
                            bray[bdata + 2] = ((blue + nbinsq / 2) / nbinsq) as u8;
                            bdata += 3;
                        } else if keep_byte == 0 {
                            bray[2 * sdata..2 * sdata + 2]
                                .copy_from_slice(&((red + green + blue) as i16).to_ne_bytes());
                            sdata += 1;
                        } else {
                            bray[2 * sdata..2 * sdata + 2].copy_from_slice(
                                &(((red + green + blue + nbinsq / 2) / nbinsq) as i16)
                                    .to_ne_bytes(),
                            );
                            sdata += 1;
                        }
                        cline1 += nbin as usize * 3;
                    }
                }
            }
            _ => {}
        }
    }
    *nxr = nxout;
    *nyr = nyout;
    0
}

/// `extractWithBinning` (`reduce_by_binning.c:434`).
pub fn extract_with_binning(
    array: &[u8],
    typ: i32,
    nx_dim: i32,
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    nbin: i32,
    bray: &mut [u8],
    keep_byte: i32,
    nxr: &mut i32,
    nyr: &mut i32,
) -> i32 {
    extract_and_bin_into_array(
        array, typ, nx_dim, x_start, x_end, y_start, y_end, nbin, bray, 0, 0, 0, keep_byte, nxr,
        nyr,
    )
}

/// C Fortran wrapper `extractwithbinning` (`reduce_by_binning.c:444`).
/// The source's `float *` arrays reach `extractWithBinning` as `void *`, so
/// they are the byte views here as well.
pub fn extractwithbinning(
    array: &[u8],
    nx_dim: &i32,
    xs: &i32,
    xe: &i32,
    ys: &i32,
    ye: &i32,
    nbin: &i32,
    bray: &mut [u8],
    keep_byte: &i32,
    nxr: &mut i32,
    nyr: &mut i32,
) -> i32 {
    extract_with_binning(
        array,
        SLICE_MODE_FLOAT,
        *nx_dim,
        *xs,
        *xe,
        *ys,
        *ye,
        *nbin,
        bray,
        *keep_byte,
        nxr,
        nyr,
    )
}

/// `repackFloatImage` (`reduce_by_binning.c:456`).
pub fn repack_float_image(
    bray: &mut [u8],
    array: &[u8],
    nxin: i32,
    xs: i32,
    xe: i32,
    ys: i32,
    ye: i32,
) {
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

/// C Fortran wrapper `irepak` (`reduce_by_binning.c:467`).
pub fn irepak(
    bray: &mut [u8],
    array: &[u8],
    nxin: &i32,
    _nyin: &i32,
    xs: &i32,
    xe: &i32,
    ys: &i32,
    ye: &i32,
) {
    repack_float_image(bray, array, *nxin, *xs, *xe, *ys, *ye)
}

/// `reduceByBinning` (`reduce_by_binning.c:482`).
pub fn reduce_by_binning(
    array: &[u8],
    typ: i32,
    nxin: i32,
    nyin: i32,
    nbin: i32,
    bray: &mut [u8],
    keep_byte: i32,
    nxr: &mut i32,
    nyr: &mut i32,
) -> i32 {
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

/// C Fortran wrapper `reduce_by_binning` (`reduce_by_binning.c:494`).
pub fn reduce_by_binning_f(
    array: &[u8],
    nx: &i32,
    ny: &i32,
    nbin: &i32,
    bray: &mut [u8],
    nxr: &mut i32,
    nyr: &mut i32,
) {
    reduce_by_binning(array, SLICE_MODE_FLOAT, *nx, *ny, *nbin, bray, 0, nxr, nyr);
}

/// `binIntoSlice` (`reduce_by_binning.c:507`).
pub fn bin_into_slice(
    array: &[f32],
    nx_dim: i32,
    bray: &mut [f32],
    nx_bin: i32,
    ny_bin: i32,
    bin_x: i32,
    bin_y: i32,
    z_weight: f32,
) {
    let factor = z_weight / (bin_x * bin_y) as f32;
    if bin_x * bin_y == 1 {
        // `reduce_by_binning.c:513-519`: this branch indexes the output with
        // `nxDim`, not `nxBin`, and it is not the same as the general loop
        // whenever the two differ.
        for iy_bin in 0..ny_bin {
            for ix_bin in 0..nx_bin {
                let ind = (ix_bin + nx_dim * iy_bin) as usize;
                bray[ind] += array[ind] * factor;
            }
        }
    } else {
        // The `numOMPthreads(8)` call and the OpenMP `parallel for` over
        // `iyBin` are not reproduced; the rows are disjoint in the output so
        // the accumulation order within a row is what matters.
        for iy_bin in 0..ny_bin {
            for iy in iy_bin * bin_y..(iy_bin + 1) * bin_y {
                let ix_base = nx_dim * iy;
                let ind_base = nx_bin * iy_bin;
                if bin_x == 1 {
                    for ix_bin in 0..nx_bin {
                        bray[(ix_bin + ind_base) as usize] +=
                            array[(ix_bin + ix_base) as usize] * factor;
                    }
                } else if bin_x == 2 {
                    for ix_bin in 0..nx_bin {
                        let ind = (2 * ix_bin + ix_base) as usize;
                        bray[(ix_bin + ind_base) as usize] +=
                            (array[ind] + array[ind + 1]) * factor;
                    }
                } else if bin_x == 3 {
                    for ix_bin in 0..nx_bin {
                        let ind = (3 * ix_bin + ix_base) as usize;
                        bray[(ix_bin + ind_base) as usize] +=
                            (array[ind] + array[ind + 1] + array[ind + 2]) * factor;
                    }
                } else if bin_x == 4 {
                    for ix_bin in 0..nx_bin {
                        let ind = (4 * ix_bin + ix_base) as usize;
                        bray[(ix_bin + ind_base) as usize] +=
                            (array[ind] + array[ind + 1] + array[ind + 2] + array[ind + 3])
                                * factor;
                    }
                } else {
                    for ix_bin in 0..nx_bin {
                        let ind = (ix_bin + ind_base) as usize;
                        for ix in ix_bin * bin_x..(ix_bin + 1) * bin_x {
                            bray[ind] += array[(ix + ix_base) as usize] * factor;
                        }
                    }
                }
            }
        }
    }
}

/// C Fortran wrapper `binintoslice` (`reduce_by_binning.c:566`).
pub fn binintoslice(
    array: &[f32],
    nx_dim: &i32,
    bray: &mut [f32],
    nx_bin: &i32,
    ny_bin: &i32,
    bin_x: &i32,
    bin_y: &i32,
    z_weight: &f32,
) {
    bin_into_slice(
        array, *nx_dim, bray, *nx_bin, *ny_bin, *bin_x, *bin_y, *z_weight,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_binning_centers_remainder_and_averages() {
        let input: Vec<u8> = (0..35)
            .map(|x| x as f32)
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let mut out = [0_u8; 24];
        let (mut x, mut y) = (0, 0);
        assert_eq!(
            reduce_by_binning(
                &input,
                SLICE_MODE_FLOAT,
                7,
                5,
                2,
                &mut out,
                0,
                &mut x,
                &mut y
            ),
            0
        );
        assert_eq!((x, y), (3, 2));
        assert_eq!(
            f32::from_ne_bytes([out[0], out[1], out[2], out[3]]),
            (0. + 1. + 7. + 8.) / 4.
        );
    }
    #[test]
    fn byte_sum_and_average_match_keep_byte_modes() {
        let input = vec![1_u8, 2, 3, 4];
        let mut sum = [0_u8; 2];
        let mut avg = [0_u8; 1];
        let (mut x, mut y) = (0, 0);
        extract_with_binning(
            &input,
            SLICE_MODE_BYTE,
            2,
            0,
            1,
            0,
            1,
            2,
            &mut sum,
            0,
            &mut x,
            &mut y,
        );
        extract_with_binning(
            &input,
            SLICE_MODE_BYTE,
            2,
            0,
            1,
            0,
            1,
            2,
            &mut avg,
            1,
            &mut x,
            &mut y,
        );
        assert_eq!(i16::from_ne_bytes([sum[0], sum[1]]), 10);
        assert_eq!(avg[0], 3);
    }
}
