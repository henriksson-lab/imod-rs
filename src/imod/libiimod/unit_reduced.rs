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
unsafe extern "C" {
    fn ceil(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn floor(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn memcpy(
        __dest: *mut ::core::ffi::c_void,
        __src: *const ::core::ffi::c_void,
        __n: size_t,
    ) -> *mut ::core::ffi::c_void;
}
pub type size_t = usize;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub type b3dUInt32 = ::core::ffi::c_uint;
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
pub const SLICE_MODE_FLOAT: i32 = 2 as i32;
pub unsafe extern "C" fn iiu_read_binned(
    mut imUnit: i32,
    mut iz: i32,
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: i32,
    mut iyDim: i32,
    mut ixUBstart: i32,
    mut iyUBstart: i32,
    mut nbin: i32,
    mut nxBin: i32,
    mut nyBin: i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: i32,
    mut ierr: *mut i32,
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
    let mut sum: ::core::ffi::c_float = 0.;
    let mut binsq: ::core::ffi::c_float = 0.;
    crate::imod::libiimod::unit_header::iiu_ret_size(
        imUnit,
        &raw mut nxyz as *mut i32,
        &raw mut mxyz as *mut i32,
        &raw mut nxyzst as *mut i32,
    );
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
        crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0 as i32);
        if ixDim == nx && nxBin == nx && nyBin == ny {
            *ierr = crate::imod::libiimod::unit_fileio::iiu_read_section(
                imUnit,
                array as *mut ::core::ffi::c_void,
            );
        } else {
            *ierr = crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                imUnit,
                array as *mut ::core::ffi::c_void,
                ixDim,
                ix0,
                ix1,
                iy0,
                iy1,
            );
        }
        return;
    }
    binsq = (nbin * nbin) as ::core::ffi::c_float;
    if (lenTemp as ::core::ffi::c_float) < binsq {
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
            crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0 as i32);
            *ierr = crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                imUnit,
                temp as *mut ::core::ffi::c_void,
                nxLoad,
                ixStart,
                ixStart + nxLoad - 1 as i32,
                iyStart,
                iyStart + nyLoad - 1 as i32,
            );
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
                            sum += *temp.offset((ix + iy * nxLoad) as isize);
                            ix += 1;
                        }
                        iy += 1;
                    }
                    *array.offset(
                        (ixDone + ixb - 1 as i32 + ixDim * (iyDone + iyb - 1 as i32)) as isize,
                    ) = sum / binsq;
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
                                    sum += *temp.offset((ix + iy * nxLoad) as isize);
                                    nsum += 1 as i32;
                                }
                                ix += 1;
                            }
                            iy += 1;
                        }
                        *array.offset(
                            (ixDone + ixb - 1 as i32 + ixDim * (iyDone + iyb - 1 as i32)) as isize,
                        ) = sum / nsum as ::core::ffi::c_float;
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
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: *mut i32,
    mut iyDim: *mut i32,
    mut ixUBstart: *mut i32,
    mut iyUBstart: *mut i32,
    mut nbin: *mut i32,
    mut nxBin: *mut i32,
    mut nyBin: *mut i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    iiu_read_binned(
        *imUnit, *iz, array, *ixDim, *iyDim, *ixUBstart, *iyUBstart, *nbin, *nxBin, *nyBin, temp,
        *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn irdbinned_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: *mut i32,
    mut iyDim: *mut i32,
    mut ixUBstart: *mut i32,
    mut iyUBstart: *mut i32,
    mut nbin: *mut i32,
    mut nxBin: *mut i32,
    mut nyBin: *mut i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    iiu_read_binned(
        *imUnit, *iz, array, *ixDim, *iyDim, *ixUBstart, *iyUBstart, *nbin, *nxBin, *nyBin, temp,
        *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn iiu_read_reduced(
    mut imUnit: i32,
    mut iz: i32,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: i32,
    mut xUBstart: ::core::ffi::c_float,
    mut yUBstart: ::core::ffi::c_float,
    mut redFac: ::core::ffi::c_float,
    mut nxRed: i32,
    mut nyRed: i32,
    mut ifiltType: i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: i32,
    mut ierr: *mut i32,
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
    let mut chunkYstart: ::core::ffi::c_float = 0.;
    let mut zoomFac: ::core::ffi::c_float = 0.;
    let mut xUseStart: ::core::ffi::c_float = 0.;
    let mut yUseStart: ::core::ffi::c_float = 0.;
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
    crate::imod::libiimod::unit_header::iiu_ret_size(
        imUnit,
        &raw mut nxyz as *mut i32,
        &raw mut mxyz as *mut i32,
        &raw mut nxyzst as *mut i32,
    );
    nx = nxyz[0 as i32 as usize];
    ny = nxyz[1 as i32 as usize];
    zoomFac = (1.0f64 / redFac as ::core::ffi::c_double) as ::core::ffi::c_float;
    ird_red_sizes_for_load(
        xUBstart,
        redFac,
        nxRed,
        nx,
        &raw mut xUseStart,
        &raw mut nxRedUse,
        &raw mut ibXoffset,
        &raw mut fillXend,
        &raw mut nbin,
        &raw mut loadXoffset,
        &raw mut loadXextra,
        &raw mut ixEdgeOffset,
        ierr,
    );
    if *ierr != 0 as i32 {
        return;
    }
    ird_red_sizes_for_load(
        yUBstart,
        redFac,
        nyRed,
        ny,
        &raw mut yUseStart,
        &raw mut nyRedUse,
        &raw mut ibYoffset,
        &raw mut fillYend,
        &raw mut nbin,
        &raw mut loadYoffset,
        &raw mut loadYextra,
        &raw mut iyEdgeOffset,
        ierr,
    );
    if *ierr != 0 as i32 {
        return;
    }
    iyEdgeStart = 1 as i32;
    *ierr = select_zoom_filter(ifiltType, zoomFac as ::core::ffi::c_double, &mut ifiltWidth);
    if *ierr != 0 as i32 {
        return;
    }
    ihalfWidth = (ifiltWidth + 3 as i32) / 2 as i32;
    ix0 = (if 0 as i32 as ::core::ffi::c_double
        > floor((xUseStart - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double)
    {
        0 as i32 as ::core::ffi::c_double
    } else {
        floor((xUseStart - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double)
    }) as i32;
    ix1 = (if ((nx - 1 as i32) as ::core::ffi::c_double)
        < ceil(
            (xUseStart
                + redFac * nxRedUse as ::core::ffi::c_float
                + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
        ) {
        (nx - 1 as i32) as ::core::ffi::c_double
    } else {
        ceil(
            (xUseStart
                + redFac * nxRedUse as ::core::ffi::c_float
                + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
        )
    }) as i32;
    nxLoad = ix1 + 1 as i32 - ix0;
    maxLineLoad = lenTemp / nxLoad;
    maxChunkLines =
        (zoomFac * (maxLineLoad - 2 as i32 * ihalfWidth) as ::core::ffi::c_float) as i32;
    *ierr = 3 as i32;
    if redFac <= 32 as i32 as ::core::ffi::c_float && maxChunkLines < 10 as i32
        || maxChunkLines < 2 as i32
    {
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
        iy0 = (if 0 as i32 as ::core::ffi::c_double
            > floor(
                (yUseStart + redFac * iyStart as ::core::ffi::c_float
                    - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            ) {
            0 as i32 as ::core::ffi::c_double
        } else {
            floor(
                (yUseStart + redFac * iyStart as ::core::ffi::c_float
                    - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            )
        }) as i32;
        iy1 = (if ((ny - 1 as i32) as ::core::ffi::c_double)
            < ceil(
                (yUseStart
                    + redFac * iyEnd as ::core::ffi::c_float
                    + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            ) {
            (ny - 1 as i32) as ::core::ffi::c_double
        } else {
            ceil(
                (yUseStart
                    + redFac * iyEnd as ::core::ffi::c_float
                    + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            )
        }) as i32;
        while iy1 >= iy0 + maxLineLoad {
            iyEnd -= 1;
            iy1 = (if ((ny - 1 as i32) as ::core::ffi::c_double)
                < ceil(
                    (yUseStart
                        + redFac * iyEnd as ::core::ffi::c_float
                        + ihalfWidth as ::core::ffi::c_float)
                        as ::core::ffi::c_double,
                ) {
                (ny - 1 as i32) as ::core::ffi::c_double
            } else {
                ceil(
                    (yUseStart
                        + redFac * iyEnd as ::core::ffi::c_float
                        + ihalfWidth as ::core::ffi::c_float)
                        as ::core::ffi::c_double,
                )
            }) as i32;
        }
        indStart = 1 as i32;
        loadYstart = iy0;
        if iy0 <= lastY1 && lastY1 < ny - 1 as i32 {
            indStart = (iy0 - lastY0) * nxLoad;
            numCopy = (lastY1 + 1 as i32 - iy0) * nxLoad;
            ix = 0 as i32;
            while ix < numCopy {
                *temp.offset(ix as isize) = *temp.offset((ix + indStart) as isize);
                ix += 1;
            }
            loadYstart = lastY1 + 1 as i32;
            indStart = numCopy + 1 as i32;
        }
        lastY0 = iy0;
        lastY1 = iy1;
        crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0 as i32);
        *ierr = -(1 as i32);
        ierr2 = crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
            imUnit,
            temp.offset((indStart - 1 as i32) as isize) as *mut ::core::ffi::c_float
                as *mut ::core::ffi::c_void,
            nxLoad,
            ix0,
            ix1,
            loadYstart,
            iy1,
        );
        if ierr2 != 0 as i32 {
            return;
        }
        chunkYstart =
            yUseStart + iyStart as ::core::ffi::c_float * redFac - iy0 as ::core::ffi::c_float;
        // `zoomWithFilter` takes typed line and output slices now; the
        // `makeLinePointers` block above still runs for its error-5 path.
        let linePtrVec: Vec<&[::core::ffi::c_float]> = (0..(iy1 + 1 - iy0) as usize)
            .map(|i| ::core::slice::from_raw_parts(temp.add(i * nxLoad as usize), nxLoad as usize))
            .collect();
        *ierr = zoom_with_filter(
            crate::imod::libcfshr::zoomdown::ZoomLines::Float(&linePtrVec),
            nxLoad,
            iy1 + 1 as i32 - iy0,
            xUseStart - ix0 as ::core::ffi::c_float,
            chunkYstart,
            nxRedUse,
            iyEnd - iyStart,
            nxDim,
            ibXoffset,
            SLICE_MODE_FLOAT,
            &mut crate::imod::libcfshr::zoomdown::ZoomOut::Float(
                ::core::slice::from_raw_parts_mut(
                    array.offset(((iyStart + ibYoffset) * nxDim) as isize),
                    (((iyEnd - iyStart - 1) * nxDim) + ibXoffset + nxRedUse) as usize,
                ),
            ),
            None,
            None,
        );
        if *ierr != 0 as i32 {
            return;
        }
        if nbin > 0 as i32 {
            iyEdgeStart = floor(chunkYstart as ::core::ffi::c_double + 0.5f64) as i32 + 1 as i32;
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
                    temp,
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    ixEdgeOffset,
                    1 as i32,
                    0 as i32,
                    nbin,
                    loadYoffset,
                    array,
                    nxDim,
                    1 as i32,
                    nxRed,
                    1 as i32,
                    1 as i32,
                );
            }
            if loadXoffset > 0 as i32 {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    0 as i32,
                    iyEdgeStart,
                    iyEdgeOffset,
                    loadXoffset,
                    nbin,
                    array,
                    nxDim,
                    1 as i32,
                    1 as i32,
                    iyBinStart,
                    iyBinEnd,
                );
            }
            if loadXextra > 0 as i32 {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    nxLoad + 1 as i32 - loadXextra,
                    0 as i32,
                    iyEdgeStart,
                    iyEdgeOffset,
                    loadXoffset,
                    nbin,
                    array,
                    nxDim,
                    nxRed,
                    nxRed,
                    iyBinStart,
                    iyBinEnd,
                );
            }
            if iyEnd >= nyRedUse && loadYextra > 0 as i32 {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as i32 - iy0,
                    1 as i32,
                    ixEdgeOffset,
                    iy1 + 2 as i32 - iy0 - loadYextra,
                    0 as i32,
                    nbin,
                    loadYextra,
                    array,
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
        if yUBstart < 0 as i32 as ::core::ffi::c_float {
            memcpy(
                array as *mut ::core::ffi::c_void,
                array.offset(nxDim as isize) as *mut ::core::ffi::c_float
                    as *const ::core::ffi::c_void,
                (nxRed as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
            );
        }
        if fillYend != 0 {
            memcpy(
                array.offset((nxDim * (nyRed - 1 as i32)) as isize) as *mut ::core::ffi::c_float
                    as *mut ::core::ffi::c_void,
                array.offset((nxDim * (nyRed - 2 as i32)) as isize) as *mut ::core::ffi::c_float
                    as *const ::core::ffi::c_void,
                (nxRed as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
            );
        }
        if xUBstart < 0 as i32 as ::core::ffi::c_float {
            ix = 0 as i32;
            while ix < nyRed {
                *array.offset((ix * nxDim) as isize) =
                    *array.offset((ix * nxDim + 1 as i32) as isize);
                ix += 1;
            }
        }
        if fillXend != 0 {
            ix = 0 as i32;
            while ix < nyRed {
                *array.offset((ix * nxDim + nxRed - 1 as i32) as isize) =
                    *array.offset((ix * nxDim + nxRed - 2 as i32) as isize);
                ix += 1;
            }
        }
    }
    *ierr = 0 as i32;
}
pub unsafe extern "C" fn irdreduced_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: *mut i32,
    mut xUBstart: *mut ::core::ffi::c_float,
    mut yUBstart: *mut ::core::ffi::c_float,
    mut redFac: *mut ::core::ffi::c_float,
    mut nxRed: *mut i32,
    mut nyRed: *mut i32,
    mut ifiltType: *mut i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    iiu_read_reduced(
        *imUnit, *iz, array, *nxDim, *xUBstart, *yUBstart, *redFac, *nxRed, *nyRed, *ifiltType,
        temp, *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn iiureadreduced_(
    mut imUnit: *mut i32,
    mut iz: *mut i32,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: *mut i32,
    mut xUBstart: *mut ::core::ffi::c_float,
    mut yUBstart: *mut ::core::ffi::c_float,
    mut redFac: *mut ::core::ffi::c_float,
    mut nxRed: *mut i32,
    mut nyRed: *mut i32,
    mut ifiltType: *mut i32,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut i32,
    mut ierr: *mut i32,
) {
    iiu_read_reduced(
        *imUnit, *iz, array, *nxDim, *xUBstart, *yUBstart, *redFac, *nxRed, *nyRed, *ifiltType,
        temp, *lenTemp, ierr,
    );
}
unsafe extern "C" fn ird_red_sizes_for_load(
    mut xUBstart: ::core::ffi::c_float,
    mut redFac: ::core::ffi::c_float,
    mut nxRed: i32,
    mut nx: i32,
    mut xUseStart: *mut ::core::ffi::c_float,
    mut nxRedUse: *mut i32,
    mut ibXoffset: *mut i32,
    mut fillEnd: *mut i32,
    mut nbin: *mut i32,
    mut loadOffset: *mut i32,
    mut loadExtra: *mut i32,
    mut ixEdgeOffset: *mut i32,
    mut ierr: *mut i32,
) {
    *xUseStart = xUBstart;
    *nxRedUse = nxRed;
    *ibXoffset = 0 as i32;
    *nbin = 0 as i32;
    *loadOffset = 0 as i32;
    *loadExtra = 0 as i32;
    *ixEdgeOffset = 0 as i32;
    *fillEnd = 0 as i32;
    *ierr = 4 as i32;
    if (xUBstart as ::core::ffi::c_double) < -(redFac as ::core::ffi::c_double - 0.99f64)
        || (xUBstart + redFac * nxRed as ::core::ffi::c_float) as i32 as ::core::ffi::c_double
            > (nx as ::core::ffi::c_float + redFac) as ::core::ffi::c_double - 0.99f64
    {
        return;
    }
    *ierr = 0 as i32;
    if ((if floor(redFac as ::core::ffi::c_double + 0.5f64) as i32 as ::core::ffi::c_float - redFac
        >= 0 as i32 as ::core::ffi::c_float
    {
        floor(redFac as ::core::ffi::c_double + 0.5f64) as i32 as ::core::ffi::c_float - redFac
    } else {
        -(floor(redFac as ::core::ffi::c_double + 0.5f64) as i32 as ::core::ffi::c_float - redFac)
    }) as ::core::ffi::c_double)
        < 1.0e-4f64
    {
        *nbin = floor(redFac as ::core::ffi::c_double + 0.5f64) as i32;
    }
    if xUBstart < 0 as i32 as ::core::ffi::c_float {
        *xUseStart = xUBstart + redFac;
        *nxRedUse = nxRed - 1 as i32;
        *ibXoffset = 1 as i32;
        if *nbin > 0 as i32 {
            *loadOffset = floor(*xUseStart as ::core::ffi::c_double + 0.5f64) as i32;
            *ixEdgeOffset = *nbin - *loadOffset;
        }
    }
    if (*xUseStart + redFac * *nxRedUse as ::core::ffi::c_float) as i32 > nx {
        *nxRedUse -= 1 as i32;
        *fillEnd = 1 as i32;
        if *nbin > 0 as i32 {
            *loadExtra = nx - (*xUseStart + redFac * *nxRedUse as ::core::ffi::c_float) as i32;
        }
    }
}
unsafe extern "C" fn ird_red_bin_edge(
    mut temp: *mut ::core::ffi::c_float,
    mut nxIn: i32,
    mut nyIn: i32,
    mut ixInStart: i32,
    mut ixInOffset: i32,
    mut iyInStart: i32,
    mut iyInOffset: i32,
    mut nbinX: i32,
    mut nbinY: i32,
    mut array: *mut ::core::ffi::c_float,
    mut nxDimOut: i32,
    mut ixOutStart: i32,
    mut ixOutEnd: i32,
    mut iyOutStart: i32,
    mut iyOutEnd: i32,
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
    let mut sum: ::core::ffi::c_float = 0.;
    iyStart = iyInStart - iyInOffset;
    iyb = iyOutStart;
    while iyb <= iyOutEnd {
        ixStart = ixInStart - ixInOffset;
        iyEnd = if nyIn < iyStart + nbinY - 1 as i32 {
            nyIn
        } else {
            iyStart + nbinY - 1 as i32
        };
        iyUse = if 1 as i32 > iyStart {
            1 as i32
        } else {
            iyStart
        };
        ixb = ixOutStart;
        while ixb <= ixOutEnd {
            ixEnd = if nxIn < ixStart + nbinX - 1 as i32 {
                nxIn
            } else {
                ixStart + nbinX - 1 as i32
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
                    sum += *temp.offset(((iy - 1 as i32) * nxIn + ix - 1 as i32) as isize);
                    ix += 1;
                }
                iy += 1;
            }
            *array.offset((ixb + (iyb - 1 as i32) * nxDimOut - 1 as i32) as isize) = sum
                / ((ixEnd + 1 as i32 - ixUse) * (iyEnd + 1 as i32 - iyUse)) as ::core::ffi::c_float;
            ixStart += nbinX;
            ixb += 1;
        }
        iyStart += nbinY;
        iyb += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_load_sizes_retain_integer_negative_and_end_edge_adjustments() {
        unsafe {
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
        unsafe {
            let mut input: Vec<f32> = (1..=16).map(|value| value as f32).collect();
            let mut output = [0.; 4];
            ird_red_bin_edge(
                input.as_mut_ptr(),
                4,
                4,
                1,
                0,
                1,
                0,
                2,
                2,
                output.as_mut_ptr(),
                2,
                1,
                2,
                1,
                2,
            );
            assert_eq!(output, [3.5, 5.5, 11.5, 13.5]);
        }
    }
}
