//! Temporary c2rust parity baseline for `IMOD/libiimod/unit_reduced.c`.
//! It retains all source reduction functions pending native unit-I/O integration.
#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_variables
)]
use crate::imod::libcfshr::b3dutil::make_line_pointers;
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_with_filter};
unsafe extern "C" {
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn ceil(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn floor(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn free(__ptr: *mut ::core::ffi::c_void);
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
pub const SLICE_MODE_FLOAT: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub unsafe extern "C" fn iiu_read_binned(
    mut imUnit: ::core::ffi::c_int,
    mut iz: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: ::core::ffi::c_int,
    mut iyDim: ::core::ffi::c_int,
    mut ixUBstart: ::core::ffi::c_int,
    mut iyUBstart: ::core::ffi::c_int,
    mut nbin: ::core::ffi::c_int,
    mut nxBin: ::core::ffi::c_int,
    mut nyBin: ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    let mut ix0: ::core::ffi::c_int = 0;
    let mut ix1: ::core::ffi::c_int = 0;
    let mut iy0: ::core::ffi::c_int = 0;
    let mut iy1: ::core::ffi::c_int = 0;
    let mut nx: ::core::ffi::c_int = 0;
    let mut ny: ::core::ffi::c_int = 0;
    let mut ixOffset: ::core::ffi::c_int = 0;
    let mut iyOffset: ::core::ffi::c_int = 0;
    let mut nxyz: [::core::ffi::c_int; 3] = [0; 3];
    let mut mxyz: [::core::ffi::c_int; 3] = [0; 3];
    let mut nxyzst: [::core::ffi::c_int; 3] = [0; 3];
    let mut ixb: ::core::ffi::c_int = 0;
    let mut nxLoad: ::core::ffi::c_int = 0;
    let mut nyLoad: ::core::ffi::c_int = 0;
    let mut maxLineLoad: ::core::ffi::c_int = 0;
    let mut maxColLoad: ::core::ffi::c_int = 0;
    let mut loadYoffset: ::core::ffi::c_int = 0;
    let mut iyStart: ::core::ffi::c_int = 0;
    let mut iyDone: ::core::ffi::c_int = 0;
    let mut nBinLines: ::core::ffi::c_int = 0;
    let mut loadXoffset: ::core::ffi::c_int = 0;
    let mut ixStart: ::core::ffi::c_int = 0;
    let mut ixDone: ::core::ffi::c_int = 0;
    let mut nBinCols: ::core::ffi::c_int = 0;
    let mut iyb: ::core::ffi::c_int = 0;
    let mut iFastStrt: ::core::ffi::c_int = 0;
    let mut iFastEnd: ::core::ffi::c_int = 0;
    let mut ixEnd: ::core::ffi::c_int = 0;
    let mut iyEnd: ::core::ffi::c_int = 0;
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut iCheckStrt: ::core::ffi::c_int = 0;
    let mut iCheckEnd: ::core::ffi::c_int = 0;
    let mut iCheck: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut sum: ::core::ffi::c_float = 0.;
    let mut binsq: ::core::ffi::c_float = 0.;
    crate::imod::libiimod::unit_header::iiu_ret_size(
        imUnit,
        &raw mut nxyz as *mut ::core::ffi::c_int,
        &raw mut mxyz as *mut ::core::ffi::c_int,
        &raw mut nxyzst as *mut ::core::ffi::c_int,
    );
    nx = nxyz[0 as ::core::ffi::c_int as usize];
    ny = nxyz[1 as ::core::ffi::c_int as usize];
    *ierr = 1 as ::core::ffi::c_int;
    ix1 = (if nx < ixUBstart + nxBin * nbin {
        nx
    } else {
        ixUBstart + nxBin * nbin
    }) - 1 as ::core::ffi::c_int;
    ixOffset = if (0 as ::core::ffi::c_int) < ixUBstart {
        0 as ::core::ffi::c_int
    } else {
        ixUBstart
    };
    ix0 = if 0 as ::core::ffi::c_int > ixUBstart {
        0 as ::core::ffi::c_int
    } else {
        ixUBstart
    };
    iy1 = (if ny < iyUBstart + nyBin * nbin {
        ny
    } else {
        iyUBstart + nyBin * nbin
    }) - 1 as ::core::ffi::c_int;
    iyOffset = if (0 as ::core::ffi::c_int) < iyUBstart {
        0 as ::core::ffi::c_int
    } else {
        iyUBstart
    };
    iy0 = if 0 as ::core::ffi::c_int > iyUBstart {
        0 as ::core::ffi::c_int
    } else {
        iyUBstart
    };
    if nbin == 1 as ::core::ffi::c_int {
        crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0 as ::core::ffi::c_int);
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
        printf(
            b"\nERROR: iiuReadBinned - Binning too large for temporary array\n\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        return;
    }
    nxLoad = ix1 + 1 as ::core::ffi::c_int - ix0;
    maxLineLoad = lenTemp / nxLoad;
    maxColLoad = nxLoad;
    if maxLineLoad < nbin {
        maxLineLoad = nbin;
        maxColLoad = lenTemp / nbin;
    }
    loadYoffset = iyOffset;
    iyStart = iy0;
    iyDone = 0 as ::core::ffi::c_int;
    while iyStart <= iy1 {
        if maxLineLoad < iy1 + 1 as ::core::ffi::c_int - iyStart {
            nyLoad = nbin * ((maxLineLoad - loadYoffset) / nbin) + loadYoffset;
        } else {
            nyLoad = iy1 + 1 as ::core::ffi::c_int - iyStart;
        }
        nBinLines = (nyLoad - loadYoffset + nbin - 1 as ::core::ffi::c_int) / nbin;
        loadXoffset = ixOffset;
        ixStart = ix0;
        ixDone = 0 as ::core::ffi::c_int;
        while ixStart <= ix1 {
            if maxColLoad < ix1 + 1 as ::core::ffi::c_int - ixStart {
                nxLoad = nbin * ((maxColLoad - loadXoffset) / nbin) + loadXoffset;
            } else {
                nxLoad = ix1 + 1 as ::core::ffi::c_int - ixStart;
            }
            nBinCols = (nxLoad - loadXoffset + nbin - 1 as ::core::ffi::c_int) / nbin;
            crate::imod::libiimod::unit_fileio::iiu_set_position(
                imUnit,
                iz,
                0 as ::core::ffi::c_int,
            );
            *ierr = crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
                imUnit,
                temp as *mut ::core::ffi::c_void,
                nxLoad,
                ixStart,
                ixStart + nxLoad - 1 as ::core::ffi::c_int,
                iyStart,
                iyStart + nyLoad - 1 as ::core::ffi::c_int,
            );
            if *ierr != 0 as ::core::ffi::c_int {
                return;
            }
            iyb = 1 as ::core::ffi::c_int;
            while iyb <= nBinLines {
                if (iyb - 1 as ::core::ffi::c_int) * nbin + loadYoffset < 0 as ::core::ffi::c_int
                    || iyb * nbin + loadYoffset > nyLoad
                {
                    iFastStrt = nBinCols + 1 as ::core::ffi::c_int;
                    iFastEnd = nBinCols;
                } else {
                    iFastStrt = 1 as ::core::ffi::c_int;
                    iFastEnd = nBinCols;
                    if loadXoffset != 0 as ::core::ffi::c_int {
                        iFastStrt = 2 as ::core::ffi::c_int;
                    }
                    if nBinCols * nbin + loadXoffset > nxLoad {
                        iFastEnd = nBinCols - 1 as ::core::ffi::c_int;
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
                        (ixDone + ixb - 1 as ::core::ffi::c_int
                            + ixDim * (iyDone + iyb - 1 as ::core::ffi::c_int))
                            as isize,
                    ) = sum / binsq;
                    ixb += 1;
                }
                iCheckStrt = 1 as ::core::ffi::c_int;
                iCheckEnd = iFastStrt - 1 as ::core::ffi::c_int;
                iCheck = 1 as ::core::ffi::c_int;
                while iCheck <= 2 as ::core::ffi::c_int {
                    ixb = iCheckStrt;
                    while ixb <= iCheckEnd {
                        sum = 0.0f32;
                        nsum = 0 as ::core::ffi::c_int;
                        iyEnd = iyb * nbin + loadYoffset;
                        ixEnd = ixb * nbin + loadXoffset;
                        iy = iyEnd - nbin;
                        while iy < iyEnd {
                            ix = ixEnd - nbin;
                            while ix < ixEnd {
                                if ix >= 0 as ::core::ffi::c_int
                                    && ix < nxLoad
                                    && iy >= 0 as ::core::ffi::c_int
                                    && iy < nyLoad
                                {
                                    sum += *temp.offset((ix + iy * nxLoad) as isize);
                                    nsum += 1 as ::core::ffi::c_int;
                                }
                                ix += 1;
                            }
                            iy += 1;
                        }
                        *array.offset(
                            (ixDone + ixb - 1 as ::core::ffi::c_int
                                + ixDim * (iyDone + iyb - 1 as ::core::ffi::c_int))
                                as isize,
                        ) = sum / nsum as ::core::ffi::c_float;
                        ixb += 1;
                    }
                    iCheckStrt = iFastEnd + 1 as ::core::ffi::c_int;
                    iCheckEnd = nBinCols;
                    iCheck += 1;
                }
                iyb += 1;
            }
            ixDone += nBinCols;
            ixStart += nxLoad;
            loadXoffset = 0 as ::core::ffi::c_int;
        }
        iyDone += nBinLines;
        iyStart += nyLoad;
        loadYoffset = 0 as ::core::ffi::c_int;
    }
}
pub unsafe extern "C" fn iiureadbinned_(
    mut imUnit: *mut ::core::ffi::c_int,
    mut iz: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: *mut ::core::ffi::c_int,
    mut iyDim: *mut ::core::ffi::c_int,
    mut ixUBstart: *mut ::core::ffi::c_int,
    mut iyUBstart: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut nxBin: *mut ::core::ffi::c_int,
    mut nyBin: *mut ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    iiu_read_binned(
        *imUnit, *iz, array, *ixDim, *iyDim, *ixUBstart, *iyUBstart, *nbin, *nxBin, *nyBin, temp,
        *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn irdbinned_(
    mut imUnit: *mut ::core::ffi::c_int,
    mut iz: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut ixDim: *mut ::core::ffi::c_int,
    mut iyDim: *mut ::core::ffi::c_int,
    mut ixUBstart: *mut ::core::ffi::c_int,
    mut iyUBstart: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut nxBin: *mut ::core::ffi::c_int,
    mut nyBin: *mut ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    iiu_read_binned(
        *imUnit, *iz, array, *ixDim, *iyDim, *ixUBstart, *iyUBstart, *nbin, *nxBin, *nyBin, temp,
        *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn iiu_read_reduced(
    mut imUnit: ::core::ffi::c_int,
    mut iz: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: ::core::ffi::c_int,
    mut xUBstart: ::core::ffi::c_float,
    mut yUBstart: ::core::ffi::c_float,
    mut redFac: ::core::ffi::c_float,
    mut nxRed: ::core::ffi::c_int,
    mut nyRed: ::core::ffi::c_int,
    mut ifiltType: ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    let mut ix0: ::core::ffi::c_int = 0;
    let mut ix1: ::core::ffi::c_int = 0;
    let mut iy0: ::core::ffi::c_int = 0;
    let mut iy1: ::core::ffi::c_int = 0;
    let mut nx: ::core::ffi::c_int = 0;
    let mut ny: ::core::ffi::c_int = 0;
    let mut nxyz: [::core::ffi::c_int; 3] = [0; 3];
    let mut mxyz: [::core::ffi::c_int; 3] = [0; 3];
    let mut nxyzst: [::core::ffi::c_int; 3] = [0; 3];
    let mut chunkYstart: ::core::ffi::c_float = 0.;
    let mut zoomFac: ::core::ffi::c_float = 0.;
    let mut xUseStart: ::core::ffi::c_float = 0.;
    let mut yUseStart: ::core::ffi::c_float = 0.;
    let mut nxLoad: ::core::ffi::c_int = 0;
    let mut nyLoad: ::core::ffi::c_int = 0;
    let mut maxLineLoad: ::core::ffi::c_int = 0;
    let mut loadYstart: ::core::ffi::c_int = 0;
    let mut ifiltWidth: ::core::ffi::c_int = 0;
    let mut ihalfWidth: ::core::ffi::c_int = 0;
    let mut ix: ::core::ffi::c_int = 0;
    let mut iyStart: ::core::ffi::c_int = 0;
    let mut maxChunkLines: ::core::ffi::c_int = 0;
    let mut lastY1: ::core::ffi::c_int = 0;
    let mut lastY0: ::core::ffi::c_int = 0;
    let mut indStart: ::core::ffi::c_int = 0;
    let mut numCopy: ::core::ffi::c_int = 0;
    let mut iyEnd: ::core::ffi::c_int = 0;
    let mut ierr2: ::core::ffi::c_int = 0;
    let mut ibXoffset: ::core::ffi::c_int = 0;
    let mut ibYoffset: ::core::ffi::c_int = 0;
    let mut nxRedUse: ::core::ffi::c_int = 0;
    let mut nyRedUse: ::core::ffi::c_int = 0;
    let mut nbin: ::core::ffi::c_int = 0;
    let mut loadXoffset: ::core::ffi::c_int = 0;
    let mut loadXextra: ::core::ffi::c_int = 0;
    let mut loadYoffset: ::core::ffi::c_int = 0;
    let mut loadYextra: ::core::ffi::c_int = 0;
    let mut iyBinStart: ::core::ffi::c_int = 0;
    let mut iyBinEnd: ::core::ffi::c_int = 0;
    let mut iyEdgeStart: ::core::ffi::c_int = 0;
    let mut iyEdgeOffset: ::core::ffi::c_int = 0;
    let mut ixEdgeOffset: ::core::ffi::c_int = 0;
    let mut fillXend: ::core::ffi::c_int = 0;
    let mut fillYend: ::core::ffi::c_int = 0;
    let mut linePtrs: *mut *mut ::core::ffi::c_uchar =
        ::core::ptr::null_mut::<*mut ::core::ffi::c_uchar>();
    crate::imod::libiimod::unit_header::iiu_ret_size(
        imUnit,
        &raw mut nxyz as *mut ::core::ffi::c_int,
        &raw mut mxyz as *mut ::core::ffi::c_int,
        &raw mut nxyzst as *mut ::core::ffi::c_int,
    );
    nx = nxyz[0 as ::core::ffi::c_int as usize];
    ny = nxyz[1 as ::core::ffi::c_int as usize];
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
    if *ierr != 0 as ::core::ffi::c_int {
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
    if *ierr != 0 as ::core::ffi::c_int {
        return;
    }
    iyEdgeStart = 1 as ::core::ffi::c_int;
    *ierr = select_zoom_filter(
        ifiltType,
        zoomFac as ::core::ffi::c_double,
        &raw mut ifiltWidth,
    );
    if *ierr != 0 as ::core::ffi::c_int {
        return;
    }
    ihalfWidth = (ifiltWidth + 3 as ::core::ffi::c_int) / 2 as ::core::ffi::c_int;
    ix0 = (if 0 as ::core::ffi::c_int as ::core::ffi::c_double
        > floor((xUseStart - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double)
    {
        0 as ::core::ffi::c_int as ::core::ffi::c_double
    } else {
        floor((xUseStart - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double)
    }) as ::core::ffi::c_int;
    ix1 = (if ((nx - 1 as ::core::ffi::c_int) as ::core::ffi::c_double)
        < ceil(
            (xUseStart
                + redFac * nxRedUse as ::core::ffi::c_float
                + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
        ) {
        (nx - 1 as ::core::ffi::c_int) as ::core::ffi::c_double
    } else {
        ceil(
            (xUseStart
                + redFac * nxRedUse as ::core::ffi::c_float
                + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
        )
    }) as ::core::ffi::c_int;
    nxLoad = ix1 + 1 as ::core::ffi::c_int - ix0;
    maxLineLoad = lenTemp / nxLoad;
    maxChunkLines = (zoomFac
        * (maxLineLoad - 2 as ::core::ffi::c_int * ihalfWidth) as ::core::ffi::c_float)
        as ::core::ffi::c_int;
    *ierr = 3 as ::core::ffi::c_int;
    if redFac <= 32 as ::core::ffi::c_int as ::core::ffi::c_float
        && maxChunkLines < 10 as ::core::ffi::c_int
        || maxChunkLines < 2 as ::core::ffi::c_int
    {
        return;
    }
    *ierr = 5 as ::core::ffi::c_int;
    linePtrs = make_line_pointers(
        temp as *mut ::core::ffi::c_void,
        nxLoad,
        maxLineLoad,
        ::core::mem::size_of::<::core::ffi::c_float>() as ::core::ffi::c_int,
    );
    if linePtrs.is_null() {
        return;
    }
    iyStart = 0 as ::core::ffi::c_int;
    lastY1 = -(1 as ::core::ffi::c_int);
    while iyStart < nyRedUse {
        iyEnd = if nyRedUse < iyStart + maxChunkLines {
            nyRedUse
        } else {
            iyStart + maxChunkLines
        };
        iy0 = (if 0 as ::core::ffi::c_int as ::core::ffi::c_double
            > floor(
                (yUseStart + redFac * iyStart as ::core::ffi::c_float
                    - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            ) {
            0 as ::core::ffi::c_int as ::core::ffi::c_double
        } else {
            floor(
                (yUseStart + redFac * iyStart as ::core::ffi::c_float
                    - ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            )
        }) as ::core::ffi::c_int;
        iy1 = (if ((ny - 1 as ::core::ffi::c_int) as ::core::ffi::c_double)
            < ceil(
                (yUseStart
                    + redFac * iyEnd as ::core::ffi::c_float
                    + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            ) {
            (ny - 1 as ::core::ffi::c_int) as ::core::ffi::c_double
        } else {
            ceil(
                (yUseStart
                    + redFac * iyEnd as ::core::ffi::c_float
                    + ihalfWidth as ::core::ffi::c_float) as ::core::ffi::c_double,
            )
        }) as ::core::ffi::c_int;
        while iy1 >= iy0 + maxLineLoad {
            iyEnd -= 1;
            iy1 = (if ((ny - 1 as ::core::ffi::c_int) as ::core::ffi::c_double)
                < ceil(
                    (yUseStart
                        + redFac * iyEnd as ::core::ffi::c_float
                        + ihalfWidth as ::core::ffi::c_float)
                        as ::core::ffi::c_double,
                ) {
                (ny - 1 as ::core::ffi::c_int) as ::core::ffi::c_double
            } else {
                ceil(
                    (yUseStart
                        + redFac * iyEnd as ::core::ffi::c_float
                        + ihalfWidth as ::core::ffi::c_float)
                        as ::core::ffi::c_double,
                )
            }) as ::core::ffi::c_int;
        }
        indStart = 1 as ::core::ffi::c_int;
        loadYstart = iy0;
        if iy0 <= lastY1 && lastY1 < ny - 1 as ::core::ffi::c_int {
            indStart = (iy0 - lastY0) * nxLoad;
            numCopy = (lastY1 + 1 as ::core::ffi::c_int - iy0) * nxLoad;
            ix = 0 as ::core::ffi::c_int;
            while ix < numCopy {
                *temp.offset(ix as isize) = *temp.offset((ix + indStart) as isize);
                ix += 1;
            }
            loadYstart = lastY1 + 1 as ::core::ffi::c_int;
            indStart = numCopy + 1 as ::core::ffi::c_int;
        }
        lastY0 = iy0;
        lastY1 = iy1;
        crate::imod::libiimod::unit_fileio::iiu_set_position(imUnit, iz, 0 as ::core::ffi::c_int);
        *ierr = -(1 as ::core::ffi::c_int);
        ierr2 = crate::imod::libiimod::unit_fileio::iiu_read_sec_part(
            imUnit,
            temp.offset((indStart - 1 as ::core::ffi::c_int) as isize) as *mut ::core::ffi::c_float
                as *mut ::core::ffi::c_void,
            nxLoad,
            ix0,
            ix1,
            loadYstart,
            iy1,
        );
        if ierr2 != 0 as ::core::ffi::c_int {
            return;
        }
        chunkYstart =
            yUseStart + iyStart as ::core::ffi::c_float * redFac - iy0 as ::core::ffi::c_float;
        *ierr = zoom_with_filter(
            linePtrs,
            nxLoad,
            iy1 + 1 as ::core::ffi::c_int - iy0,
            xUseStart - ix0 as ::core::ffi::c_float,
            chunkYstart,
            nxRedUse,
            iyEnd - iyStart,
            nxDim,
            ibXoffset,
            SLICE_MODE_FLOAT,
            array.offset(((iyStart + ibYoffset) * nxDim) as isize) as *mut ::core::ffi::c_float
                as *mut ::core::ffi::c_void,
            ::core::ptr::null_mut::<b3dUInt32>(),
            ::core::ptr::null_mut::<::core::ffi::c_uchar>(),
        );
        if *ierr != 0 as ::core::ffi::c_int {
            free(linePtrs as *mut ::core::ffi::c_void);
            return;
        }
        if nbin > 0 as ::core::ffi::c_int {
            iyEdgeStart = floor(chunkYstart as ::core::ffi::c_double + 0.5f64)
                as ::core::ffi::c_int
                + 1 as ::core::ffi::c_int;
            if iyStart == 0 as ::core::ffi::c_int && loadYoffset > 0 as ::core::ffi::c_int {
                iyEdgeStart = 1 as ::core::ffi::c_int;
            }
            iyBinStart = iyStart + ibYoffset + 1 as ::core::ffi::c_int;
            if iyStart == 0 as ::core::ffi::c_int {
                iyBinStart = 1 as ::core::ffi::c_int;
            }
            iyBinEnd = iyEnd + ibYoffset;
            if iyEnd >= nyRedUse {
                iyBinEnd = nyRed;
            }
            if iyStart == 0 as ::core::ffi::c_int && loadYoffset > 0 as ::core::ffi::c_int {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as ::core::ffi::c_int - iy0,
                    1 as ::core::ffi::c_int,
                    ixEdgeOffset,
                    1 as ::core::ffi::c_int,
                    0 as ::core::ffi::c_int,
                    nbin,
                    loadYoffset,
                    array,
                    nxDim,
                    1 as ::core::ffi::c_int,
                    nxRed,
                    1 as ::core::ffi::c_int,
                    1 as ::core::ffi::c_int,
                );
            }
            if loadXoffset > 0 as ::core::ffi::c_int {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as ::core::ffi::c_int - iy0,
                    1 as ::core::ffi::c_int,
                    0 as ::core::ffi::c_int,
                    iyEdgeStart,
                    iyEdgeOffset,
                    loadXoffset,
                    nbin,
                    array,
                    nxDim,
                    1 as ::core::ffi::c_int,
                    1 as ::core::ffi::c_int,
                    iyBinStart,
                    iyBinEnd,
                );
            }
            if loadXextra > 0 as ::core::ffi::c_int {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as ::core::ffi::c_int - iy0,
                    nxLoad + 1 as ::core::ffi::c_int - loadXextra,
                    0 as ::core::ffi::c_int,
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
            if iyEnd >= nyRedUse && loadYextra > 0 as ::core::ffi::c_int {
                ird_red_bin_edge(
                    temp,
                    nxLoad,
                    iy1 + 1 as ::core::ffi::c_int - iy0,
                    1 as ::core::ffi::c_int,
                    ixEdgeOffset,
                    iy1 + 2 as ::core::ffi::c_int - iy0 - loadYextra,
                    0 as ::core::ffi::c_int,
                    nbin,
                    loadYextra,
                    array,
                    nxDim,
                    1 as ::core::ffi::c_int,
                    nxRed,
                    nyRed,
                    nyRed,
                );
            }
            iyEdgeOffset = 0 as ::core::ffi::c_int;
        }
        iyStart = iyEnd;
    }
    if nbin == 0 as ::core::ffi::c_int {
        if yUBstart < 0 as ::core::ffi::c_int as ::core::ffi::c_float {
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
                array.offset((nxDim * (nyRed - 1 as ::core::ffi::c_int)) as isize)
                    as *mut ::core::ffi::c_float as *mut ::core::ffi::c_void,
                array.offset((nxDim * (nyRed - 2 as ::core::ffi::c_int)) as isize)
                    as *mut ::core::ffi::c_float as *const ::core::ffi::c_void,
                (nxRed as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
            );
        }
        if xUBstart < 0 as ::core::ffi::c_int as ::core::ffi::c_float {
            ix = 0 as ::core::ffi::c_int;
            while ix < nyRed {
                *array.offset((ix * nxDim) as isize) =
                    *array.offset((ix * nxDim + 1 as ::core::ffi::c_int) as isize);
                ix += 1;
            }
        }
        if fillXend != 0 {
            ix = 0 as ::core::ffi::c_int;
            while ix < nyRed {
                *array.offset((ix * nxDim + nxRed - 1 as ::core::ffi::c_int) as isize) =
                    *array.offset((ix * nxDim + nxRed - 2 as ::core::ffi::c_int) as isize);
                ix += 1;
            }
        }
    }
    free(linePtrs as *mut ::core::ffi::c_void);
    *ierr = 0 as ::core::ffi::c_int;
}
pub unsafe extern "C" fn irdreduced_(
    mut imUnit: *mut ::core::ffi::c_int,
    mut iz: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: *mut ::core::ffi::c_int,
    mut xUBstart: *mut ::core::ffi::c_float,
    mut yUBstart: *mut ::core::ffi::c_float,
    mut redFac: *mut ::core::ffi::c_float,
    mut nxRed: *mut ::core::ffi::c_int,
    mut nyRed: *mut ::core::ffi::c_int,
    mut ifiltType: *mut ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    iiu_read_reduced(
        *imUnit, *iz, array, *nxDim, *xUBstart, *yUBstart, *redFac, *nxRed, *nyRed, *ifiltType,
        temp, *lenTemp, ierr,
    );
}
pub unsafe extern "C" fn iiureadreduced_(
    mut imUnit: *mut ::core::ffi::c_int,
    mut iz: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut nxDim: *mut ::core::ffi::c_int,
    mut xUBstart: *mut ::core::ffi::c_float,
    mut yUBstart: *mut ::core::ffi::c_float,
    mut redFac: *mut ::core::ffi::c_float,
    mut nxRed: *mut ::core::ffi::c_int,
    mut nyRed: *mut ::core::ffi::c_int,
    mut ifiltType: *mut ::core::ffi::c_int,
    mut temp: *mut ::core::ffi::c_float,
    mut lenTemp: *mut ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    iiu_read_reduced(
        *imUnit, *iz, array, *nxDim, *xUBstart, *yUBstart, *redFac, *nxRed, *nyRed, *ifiltType,
        temp, *lenTemp, ierr,
    );
}
unsafe extern "C" fn ird_red_sizes_for_load(
    mut xUBstart: ::core::ffi::c_float,
    mut redFac: ::core::ffi::c_float,
    mut nxRed: ::core::ffi::c_int,
    mut nx: ::core::ffi::c_int,
    mut xUseStart: *mut ::core::ffi::c_float,
    mut nxRedUse: *mut ::core::ffi::c_int,
    mut ibXoffset: *mut ::core::ffi::c_int,
    mut fillEnd: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut loadOffset: *mut ::core::ffi::c_int,
    mut loadExtra: *mut ::core::ffi::c_int,
    mut ixEdgeOffset: *mut ::core::ffi::c_int,
    mut ierr: *mut ::core::ffi::c_int,
) {
    *xUseStart = xUBstart;
    *nxRedUse = nxRed;
    *ibXoffset = 0 as ::core::ffi::c_int;
    *nbin = 0 as ::core::ffi::c_int;
    *loadOffset = 0 as ::core::ffi::c_int;
    *loadExtra = 0 as ::core::ffi::c_int;
    *ixEdgeOffset = 0 as ::core::ffi::c_int;
    *fillEnd = 0 as ::core::ffi::c_int;
    *ierr = 4 as ::core::ffi::c_int;
    if (xUBstart as ::core::ffi::c_double) < -(redFac as ::core::ffi::c_double - 0.99f64)
        || (xUBstart + redFac * nxRed as ::core::ffi::c_float) as ::core::ffi::c_int
            as ::core::ffi::c_double
            > (nx as ::core::ffi::c_float + redFac) as ::core::ffi::c_double - 0.99f64
    {
        return;
    }
    *ierr = 0 as ::core::ffi::c_int;
    if ((if floor(redFac as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
        as ::core::ffi::c_float
        - redFac
        >= 0 as ::core::ffi::c_int as ::core::ffi::c_float
    {
        floor(redFac as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
            as ::core::ffi::c_float
            - redFac
    } else {
        -(floor(redFac as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
            as ::core::ffi::c_float
            - redFac)
    }) as ::core::ffi::c_double)
        < 1.0e-4f64
    {
        *nbin = floor(redFac as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
    }
    if xUBstart < 0 as ::core::ffi::c_int as ::core::ffi::c_float {
        *xUseStart = xUBstart + redFac;
        *nxRedUse = nxRed - 1 as ::core::ffi::c_int;
        *ibXoffset = 1 as ::core::ffi::c_int;
        if *nbin > 0 as ::core::ffi::c_int {
            *loadOffset = floor(*xUseStart as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
            *ixEdgeOffset = *nbin - *loadOffset;
        }
    }
    if (*xUseStart + redFac * *nxRedUse as ::core::ffi::c_float) as ::core::ffi::c_int > nx {
        *nxRedUse -= 1 as ::core::ffi::c_int;
        *fillEnd = 1 as ::core::ffi::c_int;
        if *nbin > 0 as ::core::ffi::c_int {
            *loadExtra = nx
                - (*xUseStart + redFac * *nxRedUse as ::core::ffi::c_float) as ::core::ffi::c_int;
        }
    }
}
unsafe extern "C" fn ird_red_bin_edge(
    mut temp: *mut ::core::ffi::c_float,
    mut nxIn: ::core::ffi::c_int,
    mut nyIn: ::core::ffi::c_int,
    mut ixInStart: ::core::ffi::c_int,
    mut ixInOffset: ::core::ffi::c_int,
    mut iyInStart: ::core::ffi::c_int,
    mut iyInOffset: ::core::ffi::c_int,
    mut nbinX: ::core::ffi::c_int,
    mut nbinY: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_float,
    mut nxDimOut: ::core::ffi::c_int,
    mut ixOutStart: ::core::ffi::c_int,
    mut ixOutEnd: ::core::ffi::c_int,
    mut iyOutStart: ::core::ffi::c_int,
    mut iyOutEnd: ::core::ffi::c_int,
) {
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut ixb: ::core::ffi::c_int = 0;
    let mut iyb: ::core::ffi::c_int = 0;
    let mut ixStart: ::core::ffi::c_int = 0;
    let mut ixEnd: ::core::ffi::c_int = 0;
    let mut iyStart: ::core::ffi::c_int = 0;
    let mut iyEnd: ::core::ffi::c_int = 0;
    let mut ixUse: ::core::ffi::c_int = 0;
    let mut iyUse: ::core::ffi::c_int = 0;
    let mut sum: ::core::ffi::c_float = 0.;
    iyStart = iyInStart - iyInOffset;
    iyb = iyOutStart;
    while iyb <= iyOutEnd {
        ixStart = ixInStart - ixInOffset;
        iyEnd = if nyIn < iyStart + nbinY - 1 as ::core::ffi::c_int {
            nyIn
        } else {
            iyStart + nbinY - 1 as ::core::ffi::c_int
        };
        iyUse = if 1 as ::core::ffi::c_int > iyStart {
            1 as ::core::ffi::c_int
        } else {
            iyStart
        };
        ixb = ixOutStart;
        while ixb <= ixOutEnd {
            ixEnd = if nxIn < ixStart + nbinX - 1 as ::core::ffi::c_int {
                nxIn
            } else {
                ixStart + nbinX - 1 as ::core::ffi::c_int
            };
            ixUse = if 1 as ::core::ffi::c_int > ixStart {
                1 as ::core::ffi::c_int
            } else {
                ixStart
            };
            sum = 0.0f32;
            iy = iyUse;
            while iy <= iyEnd {
                ix = ixUse;
                while ix <= ixEnd {
                    sum += *temp.offset(
                        ((iy - 1 as ::core::ffi::c_int) * nxIn + ix - 1 as ::core::ffi::c_int)
                            as isize,
                    );
                    ix += 1;
                }
                iy += 1;
            }
            *array.offset(
                (ixb + (iyb - 1 as ::core::ffi::c_int) * nxDimOut - 1 as ::core::ffi::c_int)
                    as isize,
            ) = sum
                / ((ixEnd + 1 as ::core::ffi::c_int - ixUse)
                    * (iyEnd + 1 as ::core::ffi::c_int - iyUse))
                    as ::core::ffi::c_float;
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
