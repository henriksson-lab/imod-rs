//! Temporary c2rust parity baseline for `IMOD/libiimod/unit_reduced.c`.
//! It retains all source reduction functions pending native unit-I/O integration.
#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_variables
)]
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_with_filter};
pub const SLICE_MODE_FLOAT: i32 = 2 as i32;
pub fn iiu_read_binned(
    imUnit: i32,
    iz: i32,
    array: &mut [f32],
    ixDim: i32,
    iyDim: i32,
    ixUBstart: i32,
    iyUBstart: i32,
    nbin: i32,
    nxBin: i32,
    nyBin: i32,
    temp: &mut [f32],
    lenTemp: i32,
    ierr: &mut i32,
) {
    let mut ix0: i32 = 0;
    let mut ix1: i32 = 0;
    let mut iy0: i32 = 0;
    let mut iy1: i32 = 0;
    let mut nx: i32 = 0;
    let mut ny: i32 = 0;
    let mut ixOffset: i32 = 0;
    let mut iyOffset: i32 = 0;
    let mut nxyz: [i32; 3] = [0; 3];
    let mut mxyz: [i32; 3] = [0; 3];
    let mut nxyzst: [i32; 3] = [0; 3];
    let mut ixb: i32 = 0;
    let mut nxLoad: i32 = 0;
    let mut nyLoad: i32 = 0;
    let mut maxLineLoad: i32 = 0;
    let mut maxColLoad: i32 = 0;
    let mut loadYoffset: i32 = 0;
    let mut iyStart: i32 = 0;
    let mut iyDone: i32 = 0;
    let mut nBinLines: i32 = 0;
    let mut loadXoffset: i32 = 0;
    let mut ixStart: i32 = 0;
    let mut ixDone: i32 = 0;
    let mut nBinCols: i32 = 0;
    let mut iyb: i32 = 0;
    let mut iFastStrt: i32 = 0;
    let mut iFastEnd: i32 = 0;
    let mut ixEnd: i32 = 0;
    let mut iyEnd: i32 = 0;
    let mut ix: i32 = 0;
    let mut iy: i32 = 0;
    let mut iCheckStrt: i32 = 0;
    let mut iCheckEnd: i32 = 0;
    let mut iCheck: i32 = 0;
    let mut nsum: i32 = 0;
    let mut sum: f32 = 0.;
    let mut binsq: f32 = 0.;
    crate::imod::libiimod::unit_header::iiu_ret_size(imUnit, &mut nxyz, &mut mxyz, &mut nxyzst);
    nx = nxyz[0 as i32 as usize];
    ny = nxyz[1 as i32 as usize];
    *ierr = 1 as i32;
    ix1 = (if nx < ixUBstart + nxBin * nbin {
        nx
    } else {
        ixUBstart + nxBin * nbin
    }) - 1 as i32;
    ixOffset = if (0 as i32) < ixUBstart {
        0 as i32
    } else {
        ixUBstart
    };
    ix0 = if 0 as i32 > ixUBstart {
        0 as i32
    } else {
        ixUBstart
    };
    iy1 = (if ny < iyUBstart + nyBin * nbin {
        ny
    } else {
        iyUBstart + nyBin * nbin
    }) - 1 as i32;
    iyOffset = if (0 as i32) < iyUBstart {
        0 as i32
    } else {
        iyUBstart
    };
    iy0 = if 0 as i32 > iyUBstart {
        0 as i32
    } else {
        iyUBstart
    };
    if nbin == 1 as i32 {
        unsafe { crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0) };
        if ixDim == nx && nxBin == nx && nyBin == ny {
            *ierr = unsafe {
                crate::imod::libiimod::unit_fileio::iiu_read_section(
                    imUnit,
                    array.as_mut_ptr().cast(),
                )
            };
        } else {
            *ierr = unsafe {
                crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                    imUnit,
                    array.as_mut_ptr().cast(),
                    ixDim,
                    ix0,
                    ix1,
                    iy0,
                    iy1,
                )
            };
        }
        return;
    }
    binsq = (nbin * nbin) as f32;
    if (lenTemp as f32) < binsq {
        {
            // Still on the C stream: this program's other output goes through
            // libc stdio and a Rust write here would reorder a redirected
            // capture.
            use std::io::Write;
            let _ = crate::imod::libcfshr::b3dutil::ImodFile::Stdout
                .write_all(b"\nERROR: iiuReadBinned - Binning too large for temporary array\n");
        }
        return;
    }
    nxLoad = ix1 + 1 as i32 - ix0;
    maxLineLoad = lenTemp / nxLoad;
    maxColLoad = nxLoad;
    if maxLineLoad < nbin {
        maxLineLoad = nbin;
        maxColLoad = lenTemp / nbin;
    }
    loadYoffset = iyOffset;
    iyStart = iy0;
    iyDone = 0 as i32;
    while iyStart <= iy1 {
        if maxLineLoad < iy1 + 1 as i32 - iyStart {
            nyLoad = nbin * ((maxLineLoad - loadYoffset) / nbin) + loadYoffset;
        } else {
            nyLoad = iy1 + 1 as i32 - iyStart;
        }
        nBinLines = (nyLoad - loadYoffset + nbin - 1 as i32) / nbin;
        loadXoffset = ixOffset;
        ixStart = ix0;
        ixDone = 0 as i32;
        while ixStart <= ix1 {
            if maxColLoad < ix1 + 1 as i32 - ixStart {
                nxLoad = nbin * ((maxColLoad - loadXoffset) / nbin) + loadXoffset;
            } else {
                nxLoad = ix1 + 1 as i32 - ixStart;
            }
            nBinCols = (nxLoad - loadXoffset + nbin - 1 as i32) / nbin;
            unsafe { crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0) };
            *ierr = unsafe {
                crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                    imUnit,
                    temp.as_mut_ptr().cast(),
                    nxLoad,
                    ixStart,
                    ixStart + nxLoad - 1,
                    iyStart,
                    iyStart + nyLoad - 1,
                )
            };
            if *ierr != 0 as i32 {
                return;
            }
            iyb = 1 as i32;
            while iyb <= nBinLines {
                if (iyb - 1 as i32) * nbin + loadYoffset < 0 as i32
                    || iyb * nbin + loadYoffset > nyLoad
                {
                    iFastStrt = nBinCols + 1 as i32;
                    iFastEnd = nBinCols;
                } else {
                    iFastStrt = 1 as i32;
                    iFastEnd = nBinCols;
                    if loadXoffset != 0 as i32 {
                        iFastStrt = 2 as i32;
                    }
                    if nBinCols * nbin + loadXoffset > nxLoad {
                        iFastEnd = nBinCols - 1 as i32;
                    }
                }
                ixb = iFastStrt;
                while ixb <= iFastEnd {
                    sum = 0.0f32;
                    iyEnd = iyb * nbin + loadYoffset;
                    ixEnd = ixb * nbin + loadXoffset;
                    iy = iyEnd - nbin;
                    while iy < iyEnd {
                        ix = ixEnd - nbin;
                        while ix < ixEnd {
                            sum += temp[(ix + iy * nxLoad) as usize];
                            ix += 1;
                        }
                        iy += 1;
                    }
                    array[(ixDone + ixb - 1 as i32 + ixDim * (iyDone + iyb - 1 as i32)) as usize] =
                        sum / binsq;
                    ixb += 1;
                }
                iCheckStrt = 1 as i32;
                iCheckEnd = iFastStrt - 1 as i32;
                iCheck = 1 as i32;
                while iCheck <= 2 as i32 {
                    ixb = iCheckStrt;
                    while ixb <= iCheckEnd {
                        sum = 0.0f32;
                        nsum = 0 as i32;
                        iyEnd = iyb * nbin + loadYoffset;
                        ixEnd = ixb * nbin + loadXoffset;
                        iy = iyEnd - nbin;
                        while iy < iyEnd {
                            ix = ixEnd - nbin;
                            while ix < ixEnd {
                                if ix >= 0 as i32 && ix < nxLoad && iy >= 0 as i32 && iy < nyLoad {
                                    sum += temp[(ix + iy * nxLoad) as usize];
                                    nsum += 1 as i32;
                                }
                                ix += 1;
                            }
                            iy += 1;
                        }
                        array[(ixDone + ixb - 1 as i32 + ixDim * (iyDone + iyb - 1 as i32))
                            as usize] = sum / nsum as f32;
                        ixb += 1;
                    }
                    iCheckStrt = iFastEnd + 1 as i32;
                    iCheckEnd = nBinCols;
                    iCheck += 1;
                }
                iyb += 1;
            }
            ixDone += nBinCols;
            ixStart += nxLoad;
            loadXoffset = 0 as i32;
        }
        iyDone += nBinLines;
        iyStart += nyLoad;
        loadYoffset = 0 as i32;
    }
}
pub unsafe extern "C" fn iiureadbinned_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut f32,
    mut ixDim: *mut i32,
    mut iyDim: *mut i32,
    mut ixUBstart: *mut i32,
    mut iyUBstart: *mut i32,
    mut nbin: *mut i32,
    mut nxBin: *mut i32,
    mut nyBin: *mut i32,
    mut temp: *mut f32,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    let Some(array_len) = (*ixDim)
        .checked_mul(*iyDim)
        .and_then(|length| usize::try_from(length).ok())
    else {
        *ierr = 1;
        return;
    };
    let Some(temp_len) = usize::try_from(*lenTemp).ok() else {
        *ierr = 1;
        return;
    };
    iiu_read_binned(
        *imUnit,
        *iz,
        core::slice::from_raw_parts_mut(array, array_len),
        *ixDim,
        *iyDim,
        *ixUBstart,
        *iyUBstart,
        *nbin,
        *nxBin,
        *nyBin,
        core::slice::from_raw_parts_mut(temp, temp_len),
        *lenTemp,
        &mut *ierr,
    );
}
pub unsafe extern "C" fn irdbinned_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut f32,
    mut ixDim: *mut i32,
    mut iyDim: *mut i32,
    mut ixUBstart: *mut i32,
    mut iyUBstart: *mut i32,
    mut nbin: *mut i32,
    mut nxBin: *mut i32,
    mut nyBin: *mut i32,
    mut temp: *mut f32,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    let Some(array_len) = (*ixDim)
        .checked_mul(*iyDim)
        .and_then(|length| usize::try_from(length).ok())
    else {
        *ierr = 1;
        return;
    };
    let Some(temp_len) = usize::try_from(*lenTemp).ok() else {
        *ierr = 1;
        return;
    };
    iiu_read_binned(
        *imUnit,
        *iz,
        core::slice::from_raw_parts_mut(array, array_len),
        *ixDim,
        *iyDim,
        *ixUBstart,
        *iyUBstart,
        *nbin,
        *nxBin,
        *nyBin,
        core::slice::from_raw_parts_mut(temp, temp_len),
        *lenTemp,
        &mut *ierr,
    );
}
pub fn iiu_read_reduced(
    imUnit: i32,
    iz: i32,
    array: &mut [f32],
    nxDim: i32,
    xUBstart: f32,
    yUBstart: f32,
    redFac: f32,
    nxRed: i32,
    nyRed: i32,
    ifiltType: i32,
    temp: &mut [f32],
    lenTemp: i32,
    ierr: &mut i32,
) {
    let mut ix0: i32 = 0;
    let mut ix1: i32 = 0;
    let mut iy0: i32 = 0;
    let mut iy1: i32 = 0;
    let mut nx: i32 = 0;
    let mut ny: i32 = 0;
    let mut nxyz: [i32; 3] = [0; 3];
    let mut mxyz: [i32; 3] = [0; 3];
    let mut nxyzst: [i32; 3] = [0; 3];
    let mut chunkYstart: f32 = 0.;
    let mut zoomFac: f32 = 0.;
    let mut xUseStart: f32 = 0.;
    let mut yUseStart: f32 = 0.;
    let mut nxLoad: i32 = 0;
    let mut nyLoad: i32 = 0;
    let mut maxLineLoad: i32 = 0;
    let mut loadYstart: i32 = 0;
    let mut ifiltWidth: i32 = 0;
    let mut ihalfWidth: i32 = 0;
    let mut ix: i32 = 0;
    let mut iyStart: i32 = 0;
    let mut maxChunkLines: i32 = 0;
    let mut lastY1: i32 = 0;
    let mut lastY0: i32 = 0;
    let mut indStart: i32 = 0;
    let mut numCopy: i32 = 0;
    let mut iyEnd: i32 = 0;
    let mut ierr2: i32 = 0;
    let mut ibXoffset: i32 = 0;
    let mut ibYoffset: i32 = 0;
    let mut nxRedUse: i32 = 0;
    let mut nyRedUse: i32 = 0;
    let mut nbin: i32 = 0;
    let mut loadXoffset: i32 = 0;
    let mut loadXextra: i32 = 0;
    let mut loadYoffset: i32 = 0;
    let mut loadYextra: i32 = 0;
    let mut iyBinStart: i32 = 0;
    let mut iyBinEnd: i32 = 0;
    let mut iyEdgeStart: i32 = 0;
    let mut iyEdgeOffset: i32 = 0;
    let mut ixEdgeOffset: i32 = 0;
    let mut fillXend: i32 = 0;
    let mut fillYend: i32 = 0;
    crate::imod::libiimod::unit_header::iiu_ret_size(imUnit, &mut nxyz, &mut mxyz, &mut nxyzst);
    nx = nxyz[0 as i32 as usize];
    ny = nxyz[1 as i32 as usize];
    zoomFac = (1.0 / redFac as f64) as f32;
    ird_red_sizes_for_load(
        xUBstart,
        redFac,
        nxRed,
        nx,
        &mut xUseStart,
        &mut nxRedUse,
        &mut ibXoffset,
        &mut fillXend,
        &mut nbin,
        &mut loadXoffset,
        &mut loadXextra,
        &mut ixEdgeOffset,
        ierr,
    );
    if *ierr != 0 {
        return;
    }
    ird_red_sizes_for_load(
        yUBstart,
        redFac,
        nyRed,
        ny,
        &mut yUseStart,
        &mut nyRedUse,
        &mut ibYoffset,
        &mut fillYend,
        &mut nbin,
        &mut loadYoffset,
        &mut loadYextra,
        &mut iyEdgeOffset,
        ierr,
    );
    if *ierr != 0 {
        return;
    }
    iyEdgeStart = 1 as i32;
    *ierr = select_zoom_filter(ifiltType, zoomFac as f64, &mut ifiltWidth);
    if *ierr != 0 {
        return;
    }
    ihalfWidth = (ifiltWidth + 3 as i32) / 2 as i32;
    ix0 = (if 0.0 > (xUseStart - ihalfWidth as f32).floor() as f64 {
        0.0
    } else {
        (xUseStart - ihalfWidth as f32).floor() as f64
    }) as i32;
    ix1 = (if ((nx - 1) as f64)
        < ((xUseStart + redFac * nxRedUse as f32 + ihalfWidth as f32).ceil() as f64)
    {
        (nx - 1) as f64
    } else {
        ((xUseStart + redFac * nxRedUse as f32 + ihalfWidth as f32).ceil() as f64)
    }) as i32;
    nxLoad = ix1 + 1 as i32 - ix0;
    maxLineLoad = lenTemp / nxLoad;
    maxChunkLines = (zoomFac * (maxLineLoad - 2 * ihalfWidth) as f32) as i32;
    *ierr = 3 as i32;
    if redFac <= 32.0 && maxChunkLines < 10 || maxChunkLines < 2 as i32 {
        return;
    }
    *ierr = 5 as i32;
    iyStart = 0 as i32;
    lastY1 = -(1 as i32);
    while iyStart < nyRedUse {
        iyEnd = if nyRedUse < iyStart + maxChunkLines {
            nyRedUse
        } else {
            iyStart + maxChunkLines
        };
        iy0 = (if 0.0 > (yUseStart + redFac * iyStart as f32 - ihalfWidth as f32).floor() as f64 {
            0.0
        } else {
            (yUseStart + redFac * iyStart as f32 - ihalfWidth as f32).floor() as f64
        }) as i32;
        iy1 = (if ((ny - 1) as f64)
            < (yUseStart + redFac * iyEnd as f32 + ihalfWidth as f32).ceil() as f64
        {
            (ny - 1) as f64
        } else {
            (yUseStart + redFac * iyEnd as f32 + ihalfWidth as f32).ceil() as f64
        }) as i32;
        while iy1 >= iy0 + maxLineLoad {
            iyEnd -= 1;
            iy1 = (if ((ny - 1) as f64)
                < (yUseStart + redFac * iyEnd as f32 + ihalfWidth as f32).ceil() as f64
            {
                (ny - 1) as f64
            } else {
                (yUseStart + redFac * iyEnd as f32 + ihalfWidth as f32).ceil() as f64
            }) as i32;
        }
        indStart = 1 as i32;
        loadYstart = iy0;
        if iy0 <= lastY1 && lastY1 < ny - 1 as i32 {
            indStart = (iy0 - lastY0) * nxLoad;
            numCopy = (lastY1 + 1 as i32 - iy0) * nxLoad;
            ix = 0 as i32;
            while ix < numCopy {
                temp[ix as usize] = temp[(ix + indStart) as usize];
                ix += 1;
            }
            loadYstart = lastY1 + 1 as i32;
            indStart = numCopy + 1 as i32;
        }
        lastY0 = iy0;
        lastY1 = iy1;
        unsafe { crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0) };
        *ierr = -(1 as i32);
        ierr2 = unsafe {
            crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                imUnit,
                temp[(indStart - 1) as usize..].as_mut_ptr().cast(),
                nxLoad,
                ix0,
                ix1,
                loadYstart,
                iy1,
            )
        };
        if ierr2 != 0 as i32 {
            return;
        }
        chunkYstart = yUseStart + iyStart as f32 * redFac - iy0 as f32;
        // `zoomWithFilter` takes typed line and output slices now; the
        // `makeLinePointers` block above still runs for its error-5 path.
        let linePtrVec: Vec<&[f32]> = temp[..(nxLoad * (iy1 + 1 - iy0)) as usize]
            .chunks(nxLoad as usize)
            .collect();
        *ierr = zoom_with_filter(
            crate::imod::libcfshr::zoomdown::ZoomLines::Float(&linePtrVec),
            nxLoad,
            iy1 + 1 as i32 - iy0,
            xUseStart - ix0 as f32,
            chunkYstart,
            nxRedUse,
            iyEnd - iyStart,
            nxDim,
            ibXoffset,
            SLICE_MODE_FLOAT,
            &mut crate::imod::libcfshr::zoomdown::ZoomOut::Float(
                &mut array[((iyStart + ibYoffset) * nxDim) as usize
                    ..((iyStart + ibYoffset) * nxDim
                        + ((iyEnd - iyStart - 1) * nxDim + ibXoffset + nxRedUse))
                        as usize],
            ),
            None,
            None,
        );
        if *ierr != 0 {
            return;
        }
        if nbin > 0 as i32 {
            iyEdgeStart = (chunkYstart as f64 + 0.5).floor() as i32 + 1;
            if iyStart == 0 as i32 && loadYoffset > 0 as i32 {
                iyEdgeStart = 1 as i32;
            }
            iyBinStart = iyStart + ibYoffset + 1 as i32;
            if iyStart == 0 as i32 {
                iyBinStart = 1 as i32;
            }
            iyBinEnd = iyEnd + ibYoffset;
            if iyEnd >= nyRedUse {
                iyBinEnd = nyRed;
            }
            if iyStart == 0 as i32 && loadYoffset > 0 as i32 {
                ird_red_bin_edge(
                    &temp[..(nxLoad * (iy1 + 1 - iy0)) as usize],
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    ixEdgeOffset,
                    1 as i32,
                    0 as i32,
                    nbin,
                    loadYoffset,
                    &mut array[..(nxDim * nyRed) as usize],
                    nxDim,
                    1 as i32,
                    nxRed,
                    1 as i32,
                    1 as i32,
                );
            }
            if loadXoffset > 0 as i32 {
                ird_red_bin_edge(
                    &temp[..(nxLoad * (iy1 + 1 - iy0)) as usize],
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    0 as i32,
                    iyEdgeStart,
                    iyEdgeOffset,
                    loadXoffset,
                    nbin,
                    &mut array[..(nxDim * nyRed) as usize],
                    nxDim,
                    1 as i32,
                    1 as i32,
                    iyBinStart,
                    iyBinEnd,
                );
            }
            if loadXextra > 0 as i32 {
                ird_red_bin_edge(
                    &temp[..(nxLoad * (iy1 + 1 - iy0)) as usize],
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    nxLoad + 1 as i32 - loadXextra,
                    0 as i32,
                    iyEdgeStart,
                    iyEdgeOffset,
                    loadXoffset,
                    nbin,
                    &mut array[..(nxDim * nyRed) as usize],
                    nxDim,
                    nxRed,
                    nxRed,
                    iyBinStart,
                    iyBinEnd,
                );
            }
            if iyEnd >= nyRedUse && loadYextra > 0 as i32 {
                ird_red_bin_edge(
                    &temp[..(nxLoad * (iy1 + 1 - iy0)) as usize],
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    ixEdgeOffset,
                    iy1 + 2 as i32 - iy0 - loadYextra,
                    0 as i32,
                    nbin,
                    loadYextra,
                    &mut array[..(nxDim * nyRed) as usize],
                    nxDim,
                    1 as i32,
                    nxRed,
                    nyRed,
                    nyRed,
                );
            }
            iyEdgeOffset = 0 as i32;
        }
        iyStart = iyEnd;
    }
    if nbin == 0 as i32 {
        if yUBstart < 0.0 {
            array.copy_within(nxDim as usize..(nxDim + nxRed) as usize, 0);
        }
        if fillYend != 0 {
            array.copy_within(
                (nxDim * (nyRed - 2)) as usize..(nxDim * (nyRed - 2) + nxRed) as usize,
                (nxDim * (nyRed - 1)) as usize,
            );
        }
        if xUBstart < 0.0 {
            ix = 0 as i32;
            while ix < nyRed {
                array[(ix * nxDim) as usize] = array[(ix * nxDim + 1) as usize];
                ix += 1;
            }
        }
        if fillXend != 0 {
            ix = 0 as i32;
            while ix < nyRed {
                array[(ix * nxDim + nxRed - 1) as usize] = array[(ix * nxDim + nxRed - 2) as usize];
                ix += 1;
            }
        }
    }
    *ierr = 0 as i32;
}
pub unsafe extern "C" fn irdreduced_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut f32,
    mut nxDim: *mut i32,
    mut xUBstart: *mut f32,
    mut yUBstart: *mut f32,
    mut redFac: *mut f32,
    mut nxRed: *mut i32,
    mut nyRed: *mut i32,
    mut ifiltType: *mut i32,
    mut temp: *mut f32,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    let Some(array_len) = (*nxDim)
        .checked_mul(*nyRed)
        .and_then(|length| usize::try_from(length).ok())
    else {
        *ierr = 1;
        return;
    };
    let Some(temp_len) = usize::try_from(*lenTemp).ok() else {
        *ierr = 1;
        return;
    };
    iiu_read_reduced(
        *imUnit,
        *iz,
        core::slice::from_raw_parts_mut(array, array_len),
        *nxDim,
        *xUBstart,
        *yUBstart,
        *redFac,
        *nxRed,
        *nyRed,
        *ifiltType,
        core::slice::from_raw_parts_mut(temp, temp_len),
        *lenTemp,
        &mut *ierr,
    );
}
pub unsafe extern "C" fn iiureadreduced_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut f32,
    mut nxDim: *mut i32,
    mut xUBstart: *mut f32,
    mut yUBstart: *mut f32,
    mut redFac: *mut f32,
    mut nxRed: *mut i32,
    mut nyRed: *mut i32,
    mut ifiltType: *mut i32,
    mut temp: *mut f32,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    let Some(array_len) = (*nxDim)
        .checked_mul(*nyRed)
        .and_then(|length| usize::try_from(length).ok())
    else {
        *ierr = 1;
        return;
    };
    let Some(temp_len) = usize::try_from(*lenTemp).ok() else {
        *ierr = 1;
        return;
    };
    iiu_read_reduced(
        *imUnit,
        *iz,
        core::slice::from_raw_parts_mut(array, array_len),
        *nxDim,
        *xUBstart,
        *yUBstart,
        *redFac,
        *nxRed,
        *nyRed,
        *ifiltType,
        core::slice::from_raw_parts_mut(temp, temp_len),
        *lenTemp,
        &mut *ierr,
    );
}
fn ird_red_sizes_for_load(
    x_ub_start: f32,
    red_fac: f32,
    nx_red: i32,
    nx: i32,
    x_use_start: &mut f32,
    nx_red_use: &mut i32,
    ib_x_offset: &mut i32,
    fill_end: &mut i32,
    nbin: &mut i32,
    load_offset: &mut i32,
    load_extra: &mut i32,
    ix_edge_offset: &mut i32,
    ierr: &mut i32,
) {
    *x_use_start = x_ub_start;
    *nx_red_use = nx_red;
    *ib_x_offset = 0;
    *nbin = 0;
    *load_offset = 0;
    *load_extra = 0;
    *ix_edge_offset = 0;
    *fill_end = 0;
    *ierr = 4;
    if (x_ub_start as f64) < -(red_fac as f64 - 0.99)
        || (x_ub_start + red_fac * nx_red as f32) as i32 as f64
            > (nx as f32 + red_fac) as f64 - 0.99
    {
        return;
    }
    *ierr = 0;
    if ((if (red_fac as f64 + 0.5).floor() as i32 as f32 - red_fac >= 0.0 {
        (red_fac as f64 + 0.5).floor() as i32 as f32 - red_fac
    } else {
        -((red_fac as f64 + 0.5).floor() as i32 as f32 - red_fac)
    }) as f64)
        < 1.0e-4
    {
        *nbin = (red_fac as f64 + 0.5).floor() as i32;
    }
    if x_ub_start < 0.0 {
        *x_use_start = x_ub_start + red_fac;
        *nx_red_use = nx_red - 1;
        *ib_x_offset = 1;
        if *nbin > 0 {
            *load_offset = (*x_use_start as f64 + 0.5).floor() as i32;
            *ix_edge_offset = *nbin - *load_offset;
        }
    }
    if (*x_use_start + red_fac * *nx_red_use as f32) as i32 > nx {
        *nx_red_use -= 1;
        *fill_end = 1;
        if *nbin > 0 {
            *load_extra = nx - (*x_use_start + red_fac * *nx_red_use as f32) as i32;
        }
    }
}
fn ird_red_bin_edge(
    temp: &[f32],
    nx_in: i32,
    ny_in: i32,
    ix_in_start: i32,
    ix_in_offset: i32,
    iy_in_start: i32,
    iy_in_offset: i32,
    nbin_x: i32,
    nbin_y: i32,
    array: &mut [f32],
    nx_dim_out: i32,
    ix_out_start: i32,
    ix_out_end: i32,
    iy_out_start: i32,
    iy_out_end: i32,
) {
    let mut ix: i32 = 0;
    let mut iy: i32 = 0;
    let mut ixb: i32 = 0;
    let mut iyb: i32 = 0;
    let mut ixStart: i32 = 0;
    let mut ixEnd: i32 = 0;
    let mut iyStart: i32 = 0;
    let mut iyEnd: i32 = 0;
    let mut ixUse: i32 = 0;
    let mut iyUse: i32 = 0;
    let mut sum: f32 = 0.;
    iyStart = iy_in_start - iy_in_offset;
    iyb = iy_out_start;
    while iyb <= iy_out_end {
        ixStart = ix_in_start - ix_in_offset;
        iyEnd = if ny_in < iyStart + nbin_y - 1 {
            ny_in
        } else {
            iyStart + nbin_y - 1
        };
        iyUse = if 1 as i32 > iyStart {
            1 as i32
        } else {
            iyStart
        };
        ixb = ix_out_start;
        while ixb <= ix_out_end {
            ixEnd = if nx_in < ixStart + nbin_x - 1 {
                nx_in
            } else {
                ixStart + nbin_x - 1
            };
            ixUse = if 1 as i32 > ixStart {
                1 as i32
            } else {
                ixStart
            };
            sum = 0.0f32;
            iy = iyUse;
            while iy <= iyEnd {
                ix = ixUse;
                while ix <= ixEnd {
                    sum += temp[((iy - 1) * nx_in + ix - 1) as usize];
                    ix += 1;
                }
                iy += 1;
            }
            array[(ixb + (iyb - 1) * nx_dim_out - 1) as usize] =
                sum / ((ixEnd + 1 - ixUse) * (iyEnd + 1 - iyUse)) as f32;
            ixStart += nbin_x;
            ixb += 1;
        }
        iyStart += nbin_y;
        iyb += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_load_sizes_retain_integer_negative_and_end_edge_adjustments() {
        {
            let mut x_use = 0.;
            let mut n_use = 0;
            let mut bin_offset = 0;
            let mut fill_end = 0;
            let mut nbin = 0;
            let mut load_offset = 0;
            let mut load_extra = 0;
            let mut edge_offset = 0;
            let mut error = 0;
            ird_red_sizes_for_load(
                -1.,
                2.,
                4,
                6,
                &mut x_use,
                &mut n_use,
                &mut bin_offset,
                &mut fill_end,
                &mut nbin,
                &mut load_offset,
                &mut load_extra,
                &mut edge_offset,
                &mut error,
            );
            assert_eq!(error, 0);
            assert_eq!((x_use, n_use, bin_offset, fill_end), (1., 2, 1, 1));
            assert_eq!((nbin, load_offset, load_extra, edge_offset), (2, 1, 1, 1));

            ird_red_sizes_for_load(
                -2.,
                2.,
                3,
                6,
                &mut x_use,
                &mut n_use,
                &mut bin_offset,
                &mut fill_end,
                &mut nbin,
                &mut load_offset,
                &mut load_extra,
                &mut edge_offset,
                &mut error,
            );
            assert_eq!(error, 4);
        }
    }

    #[test]
    fn reduced_edge_binning_uses_one_based_source_limits_and_partial_averages() {
        {
            let input: Vec<f32> = (1..=16).map(|value| value as f32).collect();
            let mut output = [0.; 4];
            ird_red_bin_edge(&input, 4, 4, 1, 0, 1, 0, 2, 2, &mut output, 2, 1, 2, 1, 2);
            assert_eq!(output, [3.5, 5.5, 11.5, 13.5]);
        }
    }
}
