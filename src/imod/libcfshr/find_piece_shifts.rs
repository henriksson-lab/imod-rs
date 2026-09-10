#![allow(
    dead_code,
    non_snake_case,
    non_camel_case_types,
    unused_mut,
    unused_assignments,
    unsafe_op_in_unsafe_fn
)]
//! Mechanical C2Rust baseline of `IMOD/libcfshr/find_piece_shifts.c`, wired to direct robust-statistics dependencies.

use super::robuststat::{rs_madn, rs_median, rs_trimmed_mean};

pub enum _IO_wide_data {}
pub enum _IO_codecvt {}
pub enum _IO_marker {}
unsafe extern "C" {
    static mut stdout: *mut FILE;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn sqrt(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn fabs(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
}
pub type size_t = usize;
pub type __off_t = ::core::ffi::c_long;
pub type __off64_t = ::core::ffi::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1],
    pub _lock: *mut ::core::ffi::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: size_t,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
#[unsafe(no_mangle)]
pub unsafe extern "C" fn find_piece_shifts(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut ixpclist: *mut ::core::ffi::c_int,
    mut iypclist: *mut ::core::ffi::c_int,
    mut dxedge: *mut ::core::ffi::c_float,
    mut dyedge: *mut ::core::ffi::c_float,
    mut idir: ::core::ffi::c_int,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: ::core::ffi::c_int,
    mut dxyvar: *mut ::core::ffi::c_float,
    mut varStep: ::core::ffi::c_int,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_float,
    mut fort: ::core::ffi::c_int,
    mut leaveInd: ::core::ffi::c_int,
    mut skipCrit: ::core::ffi::c_int,
    mut robustCrit: ::core::ffi::c_float,
    mut critMaxMove: ::core::ffi::c_float,
    mut critMoveDiff: ::core::ffi::c_float,
    mut maxIter: ::core::ffi::c_int,
    mut numAvgForTest: ::core::ffi::c_int,
    mut intervalForTest: ::core::ffi::c_int,
    mut numIter: *mut ::core::ffi::c_int,
    mut wErrMean: *mut ::core::ffi::c_float,
    mut wErrMax: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut iedge: ::core::ffi::c_int = 0;
    let mut ipc: ::core::ffi::c_int = 0;
    let mut numNeigh: ::core::ffi::c_int = 0;
    let mut isign: ::core::ffi::c_int = 0;
    let mut iter: ::core::ffi::c_int = 0;
    let mut nay: ::core::ffi::c_int = 0;
    let mut list: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut ivar: ::core::ffi::c_int = 0;
    let mut xyStep: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ixy: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut numMedian: ::core::ffi::c_int = 0;
    let mut didWeights: ::core::ffi::c_int = 0;
    let mut computeWeights: ::core::ffi::c_int = 0;
    let mut weightInterval: ::core::ffi::c_int = 0;
    let mut numEdge: [::core::ffi::c_int; 2] = [0; 2];
    let mut debug: ::core::ffi::c_int = 0;
    let mut edgeMedCrit: ::core::ffi::c_int = 6 as ::core::ffi::c_int;
    let mut edgeTrMeanX: [::core::ffi::c_float; 2] = [0.; 2];
    let mut edgeTrMeanY: [::core::ffi::c_float; 2] = [0.; 2];
    let mut edgeDevMed: [::core::ffi::c_float; 2] = [0.; 2];
    let mut edgeDevMADN: [::core::ffi::c_float; 2] = [0.; 2];
    let mut xmovemax: ::core::ffi::c_float = 0.;
    let mut ymovemax: ::core::ffi::c_float = 0.;
    let mut xsum: ::core::ffi::c_float = 0.;
    let mut ysum: ::core::ffi::c_float = 0.;
    let mut ex: ::core::ffi::c_float = 0.;
    let mut ey: ::core::ffi::c_float = 0.;
    let mut erx: [::core::ffi::c_float; 4] = [0.; 4];
    let mut ery: [::core::ffi::c_float; 4] = [0.; 4];
    let mut errd: [::core::ffi::c_float; 4] = [0.; 4];
    let mut xlow: ::core::ffi::c_float = 0.;
    let mut xsec: ::core::ffi::c_float = 0.;
    let mut xthr: ::core::ffi::c_float = 0.;
    let mut ylow: ::core::ffi::c_float = 0.;
    let mut ysec: ::core::ffi::c_float = 0.;
    let mut ythr: ::core::ffi::c_float = 0.;
    let mut elow: ::core::ffi::c_float = 0.;
    let mut esec: ::core::ffi::c_float = 0.;
    let mut ethr: ::core::ffi::c_float = 0.;
    let mut uu: ::core::ffi::c_float = 0.;
    let mut wsum: ::core::ffi::c_float = 0.;
    let mut wgt: ::core::ffi::c_float = 0.;
    let mut MAD: ::core::ffi::c_float = 0.;
    let mut Ktune: ::core::ffi::c_float = 2.0f32 * 4.685f32 * 0.6745f32 * robustCrit;
    let mut medThresh: ::core::ffi::c_float = 2.0f32 * robustCrit;
    let mut sumxmove: ::core::ffi::c_double = 0.;
    let mut sumymove: ::core::ffi::c_double = 0.;
    let mut dxsum: ::core::ffi::c_double = 0.;
    let mut dysum: ::core::ffi::c_double = 0.;
    let mut errsum: ::core::ffi::c_double = 0.;
    let mut errmax: ::core::ffi::c_double = 0.;
    let mut medSum: ::core::ffi::c_double = 0.;
    let mut xmoveAvg: ::core::ffi::c_float = 0.;
    let mut ymoveAvg: ::core::ffi::c_float = 0.;
    let mut xmoveLast: ::core::ffi::c_float = 0.;
    let mut ymoveLast: ::core::ffi::c_float = 0.;
    let mut ivarToList: *mut ::core::ffi::c_int = work as *mut ::core::ffi::c_int;
    let mut listToVar: *mut ::core::ffi::c_int = ivarToList.offset(nvar as isize);
    let mut neighInd: *mut ::core::ffi::c_int = listToVar.offset(nvar as isize);
    let mut neighList: *mut ::core::ffi::c_int = neighInd
        .offset(nvar as isize)
        .offset(1 as ::core::ffi::c_int as isize);
    let mut dxyEdge: *mut ::core::ffi::c_float =
        neighList.offset((4 as ::core::ffi::c_int * nvar) as isize) as *mut ::core::ffi::c_float;
    let mut neighWgt: *mut ::core::ffi::c_float =
        dxyEdge.offset((8 as ::core::ffi::c_int * nvar) as isize);
    let mut edgeDir: *mut ::core::ffi::c_uchar =
        neighWgt.offset((4 as ::core::ffi::c_int * nvar) as isize) as *mut ::core::ffi::c_uchar;
    let mut placed: *mut ::core::ffi::c_uchar =
        edgeDir.offset((4 as ::core::ffi::c_int * nvar) as isize);
    if edgeStep == 1 as ::core::ffi::c_int
        && pcStep == 1 as ::core::ffi::c_int
        && varStep == 1 as ::core::ffi::c_int
    {
        xyStep = 2 as ::core::ffi::c_int;
    } else if edgeStep > 1 as ::core::ffi::c_int
        && pcStep > 1 as ::core::ffi::c_int
        && varStep > 1 as ::core::ffi::c_int
    {
        xyStep = 1 as ::core::ffi::c_int;
    } else {
        return 1 as ::core::ffi::c_int;
    }
    if fort != 0 {
        fort = 1 as ::core::ffi::c_int;
    }
    if robustCrit as ::core::ffi::c_double > 0.0f64 {
        ixy = 0 as ::core::ffi::c_int;
        while ixy < 2 as ::core::ffi::c_int {
            nsum = 0 as ::core::ffi::c_int;
            ivar = 0 as ::core::ffi::c_int;
            while ivar < nvar {
                ipc = *ivarpc.offset(ivar as isize) - fort;
                iedge = *edgeUpper.offset((xyStep * ipc + pcStep * ixy) as isize) - fort;
                ind = xyStep * iedge + edgeStep * ixy;
                if iedge >= 0 as ::core::ffi::c_int
                    && *ifskipEdge.offset(ind as isize) < skipCrit
                    && ind != leaveInd - fort
                {
                    *dxyEdge.offset(nsum as isize) =
                        -idir as ::core::ffi::c_float * *dxedge.offset(ind as isize);
                    let fresh0 = nsum;
                    nsum = nsum + 1;
                    *dxyEdge.offset((fresh0 + nvar) as isize) =
                        -idir as ::core::ffi::c_float * *dyedge.offset(ind as isize);
                }
                ivar += 1;
            }
            numEdge[ixy as usize] = nsum;
            if nsum >= edgeMedCrit {
                rs_trimmed_mean(
                    dxyEdge,
                    nsum,
                    0.2f32,
                    dxyEdge.offset((2 as ::core::ffi::c_int * nvar) as isize)
                        as *mut ::core::ffi::c_float,
                    (&raw mut edgeTrMeanX as *mut ::core::ffi::c_float).offset(ixy as isize)
                        as *mut ::core::ffi::c_float,
                );
                rs_trimmed_mean(
                    dxyEdge.offset(nvar as isize) as *mut ::core::ffi::c_float,
                    nsum,
                    0.2f32,
                    dxyEdge.offset((2 as ::core::ffi::c_int * nvar) as isize)
                        as *mut ::core::ffi::c_float,
                    (&raw mut edgeTrMeanY as *mut ::core::ffi::c_float).offset(ixy as isize)
                        as *mut ::core::ffi::c_float,
                );
                i = 0 as ::core::ffi::c_int;
                while i < nsum {
                    ex = *dxyEdge.offset(i as isize) - edgeTrMeanX[ixy as usize];
                    ey = *dxyEdge.offset((i + nvar) as isize) - edgeTrMeanY[ixy as usize];
                    *dxyEdge.offset((i + 2 as ::core::ffi::c_int * nvar) as isize) = sqrt(
                        ex as ::core::ffi::c_double * ex as ::core::ffi::c_double
                            + (ey * ey) as ::core::ffi::c_double,
                    )
                        as ::core::ffi::c_float;
                    i += 1;
                }
                rs_median(
                    dxyEdge.offset((2 as ::core::ffi::c_int * nvar) as isize)
                        as *mut ::core::ffi::c_float,
                    nsum,
                    dxyEdge,
                    (&raw mut edgeDevMed as *mut ::core::ffi::c_float).offset(ixy as isize)
                        as *mut ::core::ffi::c_float,
                );
                rs_madn(
                    dxyEdge.offset((2 as ::core::ffi::c_int * nvar) as isize)
                        as *mut ::core::ffi::c_float,
                    nsum,
                    edgeDevMed[ixy as usize],
                    dxyEdge,
                    (&raw mut edgeDevMADN as *mut ::core::ffi::c_float).offset(ixy as isize)
                        as *mut ::core::ffi::c_float,
                );
            }
            ixy += 1;
        }
    }
    initialize(
        ixpclist, iypclist, ivarpc, edgeLower, edgeUpper, pieceLower, pieceUpper, ifskipEdge,
        dxedge, dyedge, dxyEdge, dxyvar, neighInd, neighWgt, edgeDir, neighList, placed,
        ivarToList, listToVar, indvar, nvar, fort, idir, leaveInd, skipCrit, xyStep, edgeStep,
        pcStep,
    );
    numNeigh = *neighInd.offset(nvar as isize);
    xmoveLast = 1.0e10f32;
    xmoveAvg = 0.0f32;
    ymoveLast = 1.0e10f32;
    ymoveAvg = 0.0f32;
    MAD = 0.0f32;
    weightInterval = if 1 as ::core::ffi::c_int
        > (if (nvar / 2 as ::core::ffi::c_int) < maxIter / 10 as ::core::ffi::c_int {
            nvar / 2 as ::core::ffi::c_int
        } else {
            maxIter / 10 as ::core::ffi::c_int
        }) {
        1 as ::core::ffi::c_int
    } else if (nvar / 2 as ::core::ffi::c_int) < maxIter / 10 as ::core::ffi::c_int {
        nvar / 2 as ::core::ffi::c_int
    } else {
        maxIter / 10 as ::core::ffi::c_int
    };
    computeWeights = 0 as ::core::ffi::c_int;
    didWeights = 0 as ::core::ffi::c_int;
    iter = 1 as ::core::ffi::c_int;
    while iter <= maxIter {
        sumxmove = 0.0f64;
        sumymove = 0.0f64;
        xmovemax = 0.0f32;
        ymovemax = 0.0f32;
        dxsum = 0.0f64;
        dysum = 0.0f64;
        if robustCrit as ::core::ffi::c_double > 0.0f64 && iter % weightInterval == 0 {
            computeWeights = 1 as ::core::ffi::c_int;
        }
        errsum = 0.0f64;
        errmax = 0.0f64;
        if computeWeights != 0 {
            numMedian = 0 as ::core::ffi::c_int;
            medSum = 0.0f64;
            list = 0 as ::core::ffi::c_int;
            while list < nvar {
                xsum = 0.0f32;
                ysum = 0.0f32;
                debug = 0 as ::core::ffi::c_int;
                i = *neighInd.offset(list as isize);
                while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
                    j = i - *neighInd.offset(list as isize);
                    nay = *neighList.offset(i as isize);
                    erx[j as usize] = *dxyvar.offset((2 as ::core::ffi::c_int * nay) as isize)
                        - *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize)
                        - *dxyEdge.offset((2 as ::core::ffi::c_int * i) as isize);
                    ery[j as usize] = *dxyvar
                        .offset((2 as ::core::ffi::c_int * nay + 1 as ::core::ffi::c_int) as isize)
                        - *dxyvar.offset(
                            (2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize,
                        )
                        - *dxyEdge.offset(
                            (2 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize,
                        );
                    find_lowest_three(
                        erx[j as usize],
                        j,
                        &raw mut xlow,
                        &raw mut xsec,
                        &raw mut xthr,
                    );
                    find_lowest_three(
                        ery[j as usize],
                        j,
                        &raw mut ylow,
                        &raw mut ysec,
                        &raw mut ythr,
                    );
                    xsum += erx[j as usize];
                    ysum += ery[j as usize];
                    i += 1;
                }
                nsum = if 1 as ::core::ffi::c_int
                    > *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize)
                        - *neighInd.offset(list as isize)
                {
                    1 as ::core::ffi::c_int
                } else {
                    *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize)
                        - *neighInd.offset(list as isize)
                };
                if nsum >= 3 as ::core::ffi::c_int {
                    if nsum > 3 as ::core::ffi::c_int {
                        xsec = ((xsec + xthr) as ::core::ffi::c_double / 2.0f64)
                            as ::core::ffi::c_float;
                        ysec = ((ysec + ythr) as ::core::ffi::c_double / 2.0f64)
                            as ::core::ffi::c_float;
                    }
                    j = 0 as ::core::ffi::c_int;
                    while j < nsum {
                        ex = erx[j as usize] - xsec;
                        ey = ery[j as usize] - ysec;
                        errd[j as usize] = sqrt((ex * ex + ey * ey) as ::core::ffi::c_double)
                            as ::core::ffi::c_float;
                        find_lowest_three(
                            errd[j as usize],
                            j,
                            &raw mut elow,
                            &raw mut esec,
                            &raw mut ethr,
                        );
                        j += 1;
                    }
                    if nsum > 3 as ::core::ffi::c_int {
                        esec = ((esec + ethr) as ::core::ffi::c_double / 2.0f64)
                            as ::core::ffi::c_float;
                    }
                    medSum += esec as ::core::ffi::c_double;
                    numMedian += 1;
                    if MAD as ::core::ffi::c_double > 0.0f64 {
                        esec = if MAD > esec { MAD } else { esec };
                        j = 0 as ::core::ffi::c_int;
                        while j < nsum {
                            uu = (errd[j as usize] - medThresh * MAD) / (Ktune * esec);
                            wgt = 0.0f32;
                            if (uu as ::core::ffi::c_double) < 1.0f64 {
                                wgt = (if uu <= 0 as ::core::ffi::c_int as ::core::ffi::c_float {
                                    1.0f64
                                } else {
                                    (1.0f64 - (uu * uu) as ::core::ffi::c_double)
                                        * (1.0f64 - (uu * uu) as ::core::ffi::c_double)
                                }) as ::core::ffi::c_float;
                            }
                            *neighWgt.offset((*neighInd.offset(list as isize) + j) as isize) = wgt;
                            j += 1;
                        }
                    }
                } else if nsum == 2 as ::core::ffi::c_int
                    && (if numEdge[0 as ::core::ffi::c_int as usize]
                        < numEdge[1 as ::core::ffi::c_int as usize]
                    {
                        numEdge[0 as ::core::ffi::c_int as usize]
                    } else {
                        numEdge[1 as ::core::ffi::c_int as usize]
                    }) > edgeMedCrit
                    && (if edgeDevMADN[0 as ::core::ffi::c_int as usize]
                        < edgeDevMADN[1 as ::core::ffi::c_int as usize]
                    {
                        edgeDevMADN[0 as ::core::ffi::c_int as usize]
                    } else {
                        edgeDevMADN[1 as ::core::ffi::c_int as usize]
                    }) as ::core::ffi::c_double
                        > 1.0e-6f64
                {
                    j = 0 as ::core::ffi::c_int;
                    while j < 2 as ::core::ffi::c_int {
                        i = *neighInd.offset(list as isize) + j;
                        ixy = *edgeDir.offset(i as isize) as ::core::ffi::c_int
                            / 2 as ::core::ffi::c_int;
                        isign = if *edgeDir.offset(i as isize) as ::core::ffi::c_int
                            % 2 as ::core::ffi::c_int
                            != 0
                        {
                            1 as ::core::ffi::c_int
                        } else {
                            -(1 as ::core::ffi::c_int)
                        };
                        ex = *dxyEdge.offset((2 as ::core::ffi::c_int * i) as isize)
                            - isign as ::core::ffi::c_float * edgeTrMeanX[ixy as usize];
                        ey = *dxyEdge.offset(
                            (2 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize,
                        ) - isign as ::core::ffi::c_float * edgeTrMeanY[ixy as usize];
                        esec = sqrt((ex * ex + ey * ey) as ::core::ffi::c_double)
                            as ::core::ffi::c_float;
                        uu = (esec - edgeDevMed[ixy as usize])
                            / (edgeDevMADN[ixy as usize] * 4.685f32);
                        *neighWgt.offset(i as isize) = 0.0f32;
                        if (uu as ::core::ffi::c_double) < 1.0f64 {
                            *neighWgt.offset(i as isize) =
                                (if uu as ::core::ffi::c_double <= 0.0f64 {
                                    1.0f64
                                } else {
                                    (1.0f64 - (uu * uu) as ::core::ffi::c_double)
                                        * (1.0f64 - (uu * uu) as ::core::ffi::c_double)
                                }) as ::core::ffi::c_float;
                        }
                        j += 1;
                    }
                    if ((if *neighWgt.offset((i - 1 as ::core::ffi::c_int) as isize)
                        > *neighWgt.offset(i as isize)
                    {
                        *neighWgt.offset((i - 1 as ::core::ffi::c_int) as isize)
                    } else {
                        *neighWgt.offset(i as isize)
                    }) as ::core::ffi::c_double)
                        < 1.0e-2f64
                    {
                        let ref mut fresh1 = *neighWgt.offset(i as isize);
                        *fresh1 = 1.0f32;
                        *neighWgt.offset((i - 1 as ::core::ffi::c_int) as isize) = *fresh1;
                    }
                }
                list += 1;
            }
            if MAD as ::core::ffi::c_double > 0.0f64 {
                didWeights = 1 as ::core::ffi::c_int;
                computeWeights = 0 as ::core::ffi::c_int;
            }
            MAD = 0.0f32;
            if numMedian != 0 {
                MAD = (if 1.0e-5f64 > medSum / numMedian as ::core::ffi::c_double {
                    1.0e-5f64
                } else {
                    medSum / numMedian as ::core::ffi::c_double
                }) as ::core::ffi::c_float;
            }
        }
        list = 0 as ::core::ffi::c_int;
        while list < nvar {
            xsum = 0.0f32;
            ysum = 0.0f32;
            wsum = 0.0f32;
            i = *neighInd.offset(list as isize);
            while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
                nay = *neighList.offset(i as isize);
                wgt = *neighWgt.offset(i as isize);
                ex = *dxyvar.offset((2 as ::core::ffi::c_int * nay) as isize)
                    - *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize)
                    - *dxyEdge.offset((2 as ::core::ffi::c_int * i) as isize);
                ey = *dxyvar
                    .offset((2 as ::core::ffi::c_int * nay + 1 as ::core::ffi::c_int) as isize)
                    - *dxyvar.offset(
                        (2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize,
                    )
                    - *dxyEdge
                        .offset((2 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize);
                xsum += ex * wgt;
                ysum += ey * wgt;
                wsum += wgt;
                i += 1;
            }
            if wsum as ::core::ffi::c_double > 1.0e-6f64 {
                xsum /= wsum;
                ysum /= wsum;
            }
            *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize) += xsum;
            *dxyvar.offset((2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize) +=
                ysum;
            dxsum +=
                *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize) as ::core::ffi::c_double;
            dysum += *dxyvar
                .offset((2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize)
                as ::core::ffi::c_double;
            sumxmove += fabs(xsum as ::core::ffi::c_double);
            sumymove += fabs(ysum as ::core::ffi::c_double);
            xmovemax = (if xmovemax as ::core::ffi::c_double > fabs(xsum as ::core::ffi::c_double) {
                xmovemax as ::core::ffi::c_double
            } else {
                fabs(xsum as ::core::ffi::c_double)
            }) as ::core::ffi::c_float;
            ymovemax = (if ymovemax as ::core::ffi::c_double > fabs(ysum as ::core::ffi::c_double) {
                ymovemax as ::core::ffi::c_double
            } else {
                fabs(ysum as ::core::ffi::c_double)
            }) as ::core::ffi::c_float;
            list += 1;
        }
        ex = (dxsum / nvar as ::core::ffi::c_double) as ::core::ffi::c_float;
        ey = (dysum / nvar as ::core::ffi::c_double) as ::core::ffi::c_float;
        list = 0 as ::core::ffi::c_int;
        while list < nvar {
            *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize) -= ex;
            *dxyvar.offset((2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize) -=
                ey;
            list += 1;
        }
        if xmovemax < critMaxMove && ymovemax < critMaxMove {
            if robustCrit as ::core::ffi::c_double <= 0.0f64 || didWeights != 0 {
                break;
            }
            computeWeights = 1 as ::core::ffi::c_int;
        }
        if iter % intervalForTest >= intervalForTest - numAvgForTest {
            xmoveAvg = (xmoveAvg as ::core::ffi::c_double
                + sumxmove / nvar as ::core::ffi::c_double)
                as ::core::ffi::c_float;
            ymoveAvg = (ymoveAvg as ::core::ffi::c_double
                + sumymove / nvar as ::core::ffi::c_double)
                as ::core::ffi::c_float;
        }
        if iter % intervalForTest == intervalForTest - 1 as ::core::ffi::c_int {
            xmoveAvg /= numAvgForTest as ::core::ffi::c_float;
            ymoveAvg /= numAvgForTest as ::core::ffi::c_float;
            if xmoveLast - xmoveAvg < critMoveDiff && ymoveLast - ymoveAvg < critMoveDiff {
                if robustCrit as ::core::ffi::c_double <= 0.0f64 || didWeights != 0 {
                    break;
                }
                computeWeights = 1 as ::core::ffi::c_int;
            }
            xmoveLast = xmoveAvg;
            xmoveAvg = 0.0f32;
            ymoveLast = ymoveAvg;
            ymoveAvg = 0.0f32;
        }
        iter += 1;
    }
    errsum = 0.0f64;
    errmax = 0.0f64;
    nsum = 0 as ::core::ffi::c_int;
    list = 0 as ::core::ffi::c_int;
    while list < nvar {
        i = *neighInd.offset(list as isize);
        while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
            nay = *neighList.offset(i as isize);
            ex = *dxyvar.offset((2 as ::core::ffi::c_int * nay) as isize)
                - *dxyvar.offset((2 as ::core::ffi::c_int * list) as isize)
                - *dxyEdge.offset((2 as ::core::ffi::c_int * i) as isize);
            ey = *dxyvar.offset((2 as ::core::ffi::c_int * nay + 1 as ::core::ffi::c_int) as isize)
                - *dxyvar
                    .offset((2 as ::core::ffi::c_int * list + 1 as ::core::ffi::c_int) as isize)
                - *dxyEdge.offset((2 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize);
            ex = sqrt(
                ex as ::core::ffi::c_double * ex as ::core::ffi::c_double
                    + (ey * ey) as ::core::ffi::c_double,
            ) as ::core::ffi::c_float
                * *neighWgt.offset(i as isize);
            errsum += ex as ::core::ffi::c_double;
            errmax = if errmax > ex as ::core::ffi::c_double {
                errmax
            } else {
                ex as ::core::ffi::c_double
            };
            nsum += 1;
            i += 1;
        }
        list += 1;
    }
    *wErrMax = errmax as ::core::ffi::c_float;
    *wErrMean = (errsum
        / (if 1 as ::core::ffi::c_int > nsum {
            1 as ::core::ffi::c_int
        } else {
            nsum
        }) as ::core::ffi::c_double) as ::core::ffi::c_float;
    ivar = 0 as ::core::ffi::c_int;
    while ivar < 2 as ::core::ffi::c_int * nvar {
        *dxyEdge.offset(ivar as isize) = *dxyvar.offset(ivar as isize);
        ivar += 1;
    }
    ivar = 0 as ::core::ffi::c_int;
    while ivar < nvar {
        *dxyvar.offset((xyStep * ivar) as isize) =
            *dxyEdge.offset((2 as ::core::ffi::c_int * *ivarToList.offset(ivar as isize)) as isize);
        *dxyvar.offset((xyStep * ivar + varStep) as isize) = *dxyEdge.offset(
            (2 as ::core::ffi::c_int * *ivarToList.offset(ivar as isize) + 1 as ::core::ffi::c_int)
                as isize,
        );
        ivar += 1;
    }
    ivar = 0 as ::core::ffi::c_int;
    while ivar < 2 as ::core::ffi::c_int * nvar {
        *dxyEdge.offset(ivar as isize) = 0.0f32;
        *placed.offset(ivar as isize) = 0 as ::core::ffi::c_uchar;
        ivar += 1;
    }
    *numIter = iter;
    list = 0 as ::core::ffi::c_int;
    while list < nvar {
        i = *neighInd.offset(list as isize);
        while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
            nay = *neighList.offset(i as isize);
            ipc = list;
            ixy = *edgeDir.offset(i as isize) as ::core::ffi::c_int / 2 as ::core::ffi::c_int;
            if *edgeDir.offset(i as isize) as ::core::ffi::c_int % 2 as ::core::ffi::c_int != 0 {
                ipc = nay;
                nay = list;
            }
            ivar = *listToVar.offset(nay as isize);
            *dxyEdge.offset((2 as ::core::ffi::c_int * ivar + ixy) as isize) +=
                *neighWgt.offset(i as isize);
            let ref mut fresh2 = *placed.offset((2 as ::core::ffi::c_int * ivar + ixy) as isize);
            *fresh2 = (*fresh2).wrapping_add(1);
            i += 1;
        }
        list += 1;
    }
    ivar = 0 as ::core::ffi::c_int;
    while ivar < 2 as ::core::ffi::c_int * nvar {
        if *placed.offset(ivar as isize) != 0 {
            *work.offset(ivar as isize) = *dxyEdge.offset(ivar as isize)
                / *placed.offset(ivar as isize) as ::core::ffi::c_int as ::core::ffi::c_float;
        } else {
            *work.offset(ivar as isize) = -1.0f64 as ::core::ffi::c_float;
        }
        ivar += 1;
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn find_piece_shifts_fortran(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: *mut ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut ixpclist: *mut ::core::ffi::c_int,
    mut iypclist: *mut ::core::ffi::c_int,
    mut dxedge: *mut ::core::ffi::c_float,
    mut dyedge: *mut ::core::ffi::c_float,
    mut idir: *mut ::core::ffi::c_int,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: *mut ::core::ffi::c_int,
    mut dxyvar: *mut ::core::ffi::c_float,
    mut varStep: *mut ::core::ffi::c_int,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: *mut ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_float,
    mut fort: *mut ::core::ffi::c_int,
    mut leaveInd: *mut ::core::ffi::c_int,
    mut skipCrit: *mut ::core::ffi::c_int,
    mut robustCrit: *mut ::core::ffi::c_float,
    mut critMaxMove: *mut ::core::ffi::c_float,
    mut critMoveDiff: *mut ::core::ffi::c_float,
    mut maxIter: *mut ::core::ffi::c_int,
    mut numAvgForTest: *mut ::core::ffi::c_int,
    mut intervalForTest: *mut ::core::ffi::c_int,
    mut numIter: *mut ::core::ffi::c_int,
    mut wErrMean: *mut ::core::ffi::c_float,
    mut wErrMax: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    return find_piece_shifts(
        ivarpc,
        *nvar,
        indvar,
        ixpclist,
        iypclist,
        dxedge,
        dyedge,
        *idir,
        pieceLower,
        pieceUpper,
        ifskipEdge,
        *edgeStep,
        dxyvar,
        *varStep,
        edgeLower,
        edgeUpper,
        *pcStep,
        work,
        *fort,
        *leaveInd,
        *skipCrit,
        *robustCrit,
        *critMaxMove,
        *critMoveDiff,
        *maxIter,
        *numAvgForTest,
        *intervalForTest,
        numIter,
        wErrMean,
        wErrMax,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn find_piece_scalings(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut ixpclist: *mut ::core::ffi::c_int,
    mut iypclist: *mut ::core::ffi::c_int,
    mut ddenEdge: *mut ::core::ffi::c_float,
    mut idir: ::core::ffi::c_int,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: ::core::ffi::c_int,
    mut ddenVar: *mut ::core::ffi::c_float,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_float,
    mut fort: ::core::ffi::c_int,
    mut leaveInd: ::core::ffi::c_int,
    mut skipCrit: ::core::ffi::c_int,
    mut critMaxMove: ::core::ffi::c_float,
    mut critMoveDiff: ::core::ffi::c_float,
    mut maxIter: ::core::ffi::c_int,
    mut numAvgForTest: ::core::ffi::c_int,
    mut intervalForTest: ::core::ffi::c_int,
    mut numIter: *mut ::core::ffi::c_int,
    mut wErrMean: *mut ::core::ffi::c_float,
    mut wErrMax: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut iedge: ::core::ffi::c_int = 0;
    let mut ipc: ::core::ffi::c_int = 0;
    let mut numNeigh: ::core::ffi::c_int = 0;
    let mut isign: ::core::ffi::c_int = 0;
    let mut iter: ::core::ffi::c_int = 0;
    let mut nay: ::core::ffi::c_int = 0;
    let mut list: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut ivar: ::core::ffi::c_int = 0;
    let mut xyStep: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ixy: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut numEdge: [::core::ffi::c_int; 2] = [0; 2];
    let mut debug: ::core::ffi::c_int = 0;
    let mut xmovemax: ::core::ffi::c_float = 0.;
    let mut xsum: ::core::ffi::c_float = 0.;
    let mut ex: ::core::ffi::c_float = 0.;
    let mut wsum: ::core::ffi::c_float = 0.;
    let mut sumxmove: ::core::ffi::c_double = 0.;
    let mut dxsum: ::core::ffi::c_double = 0.;
    let mut errsum: ::core::ffi::c_double = 0.;
    let mut errmax: ::core::ffi::c_double = 0.;
    let mut medSum: ::core::ffi::c_double = 0.;
    let mut xmoveAvg: ::core::ffi::c_float = 0.;
    let mut xmoveLast: ::core::ffi::c_float = 0.;
    let mut ivarToList: *mut ::core::ffi::c_int = work as *mut ::core::ffi::c_int;
    let mut listToVar: *mut ::core::ffi::c_int = ivarToList.offset(nvar as isize);
    let mut neighInd: *mut ::core::ffi::c_int = listToVar.offset(nvar as isize);
    let mut neighList: *mut ::core::ffi::c_int = neighInd
        .offset(nvar as isize)
        .offset(1 as ::core::ffi::c_int as isize);
    let mut dtmpEdge: *mut ::core::ffi::c_float =
        neighList.offset((4 as ::core::ffi::c_int * nvar) as isize) as *mut ::core::ffi::c_float;
    let mut edgeDir: *mut ::core::ffi::c_uchar =
        dtmpEdge.offset((4 as ::core::ffi::c_int * nvar) as isize) as *mut ::core::ffi::c_uchar;
    let mut placed: *mut ::core::ffi::c_uchar =
        edgeDir.offset((4 as ::core::ffi::c_int * nvar) as isize);
    if edgeStep == 1 as ::core::ffi::c_int && pcStep == 1 as ::core::ffi::c_int {
        xyStep = 2 as ::core::ffi::c_int;
    } else if edgeStep > 1 as ::core::ffi::c_int && pcStep > 1 as ::core::ffi::c_int {
        xyStep = 1 as ::core::ffi::c_int;
    } else {
        return 1 as ::core::ffi::c_int;
    }
    if fort != 0 {
        fort = 1 as ::core::ffi::c_int;
    }
    initialize(
        ixpclist,
        iypclist,
        ivarpc,
        edgeLower,
        edgeUpper,
        pieceLower,
        pieceUpper,
        ifskipEdge,
        ddenEdge,
        ::core::ptr::null_mut::<::core::ffi::c_float>(),
        dtmpEdge,
        ddenVar,
        neighInd,
        ::core::ptr::null_mut::<::core::ffi::c_float>(),
        edgeDir,
        neighList,
        placed,
        ivarToList,
        listToVar,
        indvar,
        nvar,
        fort,
        idir,
        leaveInd,
        skipCrit,
        xyStep,
        edgeStep,
        pcStep,
    );
    numNeigh = *neighInd.offset(nvar as isize);
    xmoveLast = 1.0e10f32;
    xmoveAvg = 0.0f32;
    iter = 1 as ::core::ffi::c_int;
    while iter <= maxIter {
        sumxmove = 0.0f64;
        xmovemax = 0.0f32;
        dxsum = 0.0f64;
        errsum = 0.0f64;
        errmax = 0.0f64;
        list = 0 as ::core::ffi::c_int;
        while list < nvar {
            xsum = 0.0f32;
            wsum = 0.0f32;
            i = *neighInd.offset(list as isize);
            while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
                nay = *neighList.offset(i as isize);
                ex = *ddenVar.offset(nay as isize)
                    - *ddenVar.offset(list as isize)
                    - *dtmpEdge.offset(i as isize);
                xsum += ex;
                wsum = (wsum as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
                i += 1;
            }
            if wsum as ::core::ffi::c_double > 1.0e-6f64 {
                xsum /= wsum;
            }
            *ddenVar.offset(list as isize) += xsum;
            dxsum += *ddenVar.offset(list as isize) as ::core::ffi::c_double;
            sumxmove += fabs(xsum as ::core::ffi::c_double);
            xmovemax = (if xmovemax as ::core::ffi::c_double > fabs(xsum as ::core::ffi::c_double) {
                xmovemax as ::core::ffi::c_double
            } else {
                fabs(xsum as ::core::ffi::c_double)
            }) as ::core::ffi::c_float;
            list += 1;
        }
        ex = (dxsum / nvar as ::core::ffi::c_double) as ::core::ffi::c_float;
        list = 0 as ::core::ffi::c_int;
        while list < nvar {
            *ddenVar.offset(list as isize) -= ex;
            list += 1;
        }
        if xmovemax < critMaxMove {
            break;
        }
        if iter % intervalForTest >= intervalForTest - numAvgForTest {
            xmoveAvg = (xmoveAvg as ::core::ffi::c_double
                + sumxmove / nvar as ::core::ffi::c_double)
                as ::core::ffi::c_float;
        }
        if iter % intervalForTest == intervalForTest - 1 as ::core::ffi::c_int {
            xmoveAvg /= numAvgForTest as ::core::ffi::c_float;
            if xmoveLast - xmoveAvg < critMoveDiff {
                break;
            }
            xmoveLast = xmoveAvg;
            xmoveAvg = 0.0f32;
        }
        iter += 1;
    }
    errsum = 0.0f64;
    errmax = 0.0f64;
    nsum = 0 as ::core::ffi::c_int;
    list = 0 as ::core::ffi::c_int;
    while list < nvar {
        i = *neighInd.offset(list as isize);
        while i < *neighInd.offset((list + 1 as ::core::ffi::c_int) as isize) {
            nay = *neighList.offset(i as isize);
            ex = *ddenVar.offset(nay as isize)
                - *ddenVar.offset(list as isize)
                - *dtmpEdge.offset(i as isize);
            errsum += fabs(ex as ::core::ffi::c_double);
            errmax = if errmax > ex as ::core::ffi::c_double {
                errmax
            } else {
                ex as ::core::ffi::c_double
            };
            nsum += 1;
            i += 1;
        }
        list += 1;
    }
    *wErrMax = errmax as ::core::ffi::c_float;
    *wErrMean = (errsum
        / (if 1 as ::core::ffi::c_int > nsum {
            1 as ::core::ffi::c_int
        } else {
            nsum
        }) as ::core::ffi::c_double) as ::core::ffi::c_float;
    ivar = 0 as ::core::ffi::c_int;
    while ivar < nvar {
        *dtmpEdge.offset(ivar as isize) = *ddenVar.offset(ivar as isize);
        ivar += 1;
    }
    ivar = 0 as ::core::ffi::c_int;
    while ivar < nvar {
        *ddenVar.offset(ivar as isize) =
            *dtmpEdge.offset(*ivarToList.offset(ivar as isize) as isize);
        ivar += 1;
    }
    *numIter = iter;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn find_piece_scalings_fortran(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: *mut ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut ixpclist: *mut ::core::ffi::c_int,
    mut iypclist: *mut ::core::ffi::c_int,
    mut ddenedge: *mut ::core::ffi::c_float,
    mut idir: *mut ::core::ffi::c_int,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: *mut ::core::ffi::c_int,
    mut ddenvar: *mut ::core::ffi::c_float,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: *mut ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_float,
    mut fort: *mut ::core::ffi::c_int,
    mut leaveInd: *mut ::core::ffi::c_int,
    mut skipCrit: *mut ::core::ffi::c_int,
    mut critMaxMove: *mut ::core::ffi::c_float,
    mut critMoveDiff: *mut ::core::ffi::c_float,
    mut maxIter: *mut ::core::ffi::c_int,
    mut numAvgForTest: *mut ::core::ffi::c_int,
    mut intervalForTest: *mut ::core::ffi::c_int,
    mut numIter: *mut ::core::ffi::c_int,
    mut wErrMean: *mut ::core::ffi::c_float,
    mut wErrMax: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    return find_piece_scalings(
        ivarpc,
        *nvar,
        indvar,
        ixpclist,
        iypclist,
        ddenedge,
        *idir,
        pieceLower,
        pieceUpper,
        ifskipEdge,
        *edgeStep,
        ddenvar,
        edgeLower,
        edgeUpper,
        *pcStep,
        work,
        *fort,
        *leaveInd,
        *skipCrit,
        *critMaxMove,
        *critMoveDiff,
        *maxIter,
        *numAvgForTest,
        *intervalForTest,
        numIter,
        wErrMean,
        wErrMax,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pick_alternative_shifts(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut dxedge: *mut ::core::ffi::c_float,
    mut dyedge: *mut ::core::ffi::c_float,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: ::core::ffi::c_int,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: ::core::ffi::c_int,
    mut fort: ::core::ffi::c_int,
    mut altDxys: *mut ::core::ffi::c_float,
    mut numAlts: ::core::ffi::c_int,
    mut altIxy: ::core::ffi::c_int,
    mut errThresh: ::core::ffi::c_float,
    mut reduceFac: ::core::ffi::c_float,
    mut newThresh: ::core::ffi::c_float,
    mut fixedEdges: *mut ::core::ffi::c_int,
    mut numFixed: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut ixy: ::core::ffi::c_int = 0;
    let mut iyx: ::core::ffi::c_int = 0;
    let mut ipc: ::core::ffi::c_int = 0;
    let mut iedge: ::core::ffi::c_int = 0;
    let mut neigh: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ivar: ::core::ffi::c_int = 0;
    let mut varStart: ::core::ffi::c_int = 0;
    let mut varEnd: ::core::ffi::c_int = 0;
    let mut idir: ::core::ffi::c_int = 0;
    let mut prm1st: [::core::ffi::c_int; 2] = [0; 2];
    let mut prm2nd: [::core::ffi::c_int; 2] = [0; 2];
    let mut alt1st: [::core::ffi::c_int; 2] = [0; 2];
    let mut alt2nd: [::core::ffi::c_int; 2] = [0; 2];
    let mut ind1: ::core::ffi::c_int = 0;
    let mut ind2: ::core::ffi::c_int = 0;
    let mut ind3: ::core::ffi::c_int = 0;
    let mut ind4: ::core::ffi::c_int = 0;
    let mut minInd1: ::core::ffi::c_int = 0;
    let mut numFull: ::core::ffi::c_int = 0;
    let mut xyStep: ::core::ffi::c_int = 0;
    let mut aStep: ::core::ffi::c_int = 0;
    let mut minInd2: ::core::ffi::c_int = 0;
    let mut minInd3: ::core::ffi::c_int = 0;
    let mut minInd4: ::core::ffi::c_int = 0;
    let mut ftmp: ::core::ffi::c_float = 0.;
    let mut minErr: ::core::ffi::c_float = 0.;
    let mut delx: ::core::ffi::c_float = 0.;
    let mut dely: ::core::ffi::c_float = 0.;
    let mut err: ::core::ffi::c_float = 0.;
    let mut firstErr: ::core::ffi::c_float = 0.;
    let mut altStep: ::core::ffi::c_int = 2 as ::core::ffi::c_int * numAlts;
    if edgeStep == 1 as ::core::ffi::c_int && pcStep == 1 as ::core::ffi::c_int {
        xyStep = 2 as ::core::ffi::c_int;
    } else if edgeStep > 1 as ::core::ffi::c_int && pcStep > 1 as ::core::ffi::c_int {
        xyStep = 1 as ::core::ffi::c_int;
    } else {
        return 1 as ::core::ffi::c_int;
    }
    if fort != 0 {
        fort = 1 as ::core::ffi::c_int;
    }
    varStart = nvar / 2 as ::core::ffi::c_int;
    varEnd = 0 as ::core::ffi::c_int;
    idir = -(1 as ::core::ffi::c_int);
    while idir <= 1 as ::core::ffi::c_int {
        ivar = varStart;
        while idir * (ivar - varEnd) <= 0 as ::core::ffi::c_int {
            ipc = *ivarpc.offset(ivar as isize) - fort;
            numFull = 0 as ::core::ffi::c_int;
            ixy = 0 as ::core::ffi::c_int;
            while ixy < 2 as ::core::ffi::c_int {
                iyx = 1 as ::core::ffi::c_int - ixy;
                iedge = *edgeUpper.offset((xyStep * ipc + pcStep * ixy) as isize) - fort;
                ind = xyStep * iedge + edgeStep * ixy;
                if iedge >= 0 as ::core::ffi::c_int && *ifskipEdge.offset(ind as isize) == 0 {
                    neigh = *pieceUpper.offset(ind as isize) - fort;
                    if *indvar.offset(neigh as isize) >= 0 as ::core::ffi::c_int {
                        prm1st[ixy as usize] = ind;
                        alt1st[ixy as usize] = altStep * iedge + altIxy * ixy;
                        iedge = *edgeUpper.offset((xyStep * neigh + pcStep * iyx) as isize) - fort;
                        ind = xyStep * iedge + edgeStep * iyx;
                        if iedge >= 0 as ::core::ffi::c_int && *ifskipEdge.offset(ind as isize) == 0
                        {
                            neigh = *pieceUpper.offset(ind as isize) - fort;
                            if *indvar.offset(neigh as isize) >= 0 as ::core::ffi::c_int {
                                prm2nd[ixy as usize] = ind;
                                alt2nd[ixy as usize] = altStep * iedge + altIxy * iyx;
                                numFull += 1;
                            }
                        }
                    }
                }
                ixy += 1;
            }
            if !(numFull < 2 as ::core::ffi::c_int) {
                minErr = 1.0e10f32;
                firstErr = -1.0f64 as ::core::ffi::c_float;
                ind1 = -(1 as ::core::ffi::c_int);
                while ind1 < numAlts {
                    ind2 = -(1 as ::core::ffi::c_int);
                    while ind2 < numAlts {
                        ind3 = -(1 as ::core::ffi::c_int);
                        while ind3 < numAlts {
                            ind4 = -(1 as ::core::ffi::c_int);
                            while ind4 < numAlts {
                                if !((if ind1 < 0 as ::core::ffi::c_int {
                                    0 as ::core::ffi::c_int
                                } else {
                                    1 as ::core::ffi::c_int
                                }) + (if ind2 < 0 as ::core::ffi::c_int {
                                    0 as ::core::ffi::c_int
                                } else {
                                    1 as ::core::ffi::c_int
                                }) + (if ind3 < 0 as ::core::ffi::c_int {
                                    0 as ::core::ffi::c_int
                                } else {
                                    1 as ::core::ffi::c_int
                                }) + (if ind4 < 0 as ::core::ffi::c_int {
                                    0 as ::core::ffi::c_int
                                } else {
                                    1 as ::core::ffi::c_int
                                }) > 1 as ::core::ffi::c_int)
                                {
                                    if !(ind1 >= 0 as ::core::ffi::c_int
                                        && (*altDxys.offset(
                                            (alt1st[0 as ::core::ffi::c_int as usize]
                                                + 2 as ::core::ffi::c_int * ind1)
                                                as isize,
                                        )
                                            as ::core::ffi::c_double)
                                            < -1.0e20f64
                                        || ind2 >= 0 as ::core::ffi::c_int
                                            && (*altDxys.offset(
                                                (alt2nd[0 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind2)
                                                    as isize,
                                            )
                                                as ::core::ffi::c_double)
                                                < -1.0e20f64
                                        || ind3 >= 0 as ::core::ffi::c_int
                                            && (*altDxys.offset(
                                                (alt1st[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind3)
                                                    as isize,
                                            )
                                                as ::core::ffi::c_double)
                                                < -1.0e20f64
                                        || ind4 >= 0 as ::core::ffi::c_int
                                            && (*altDxys.offset(
                                                (alt2nd[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind4)
                                                    as isize,
                                            )
                                                as ::core::ffi::c_double)
                                                < -1.0e20f64)
                                    {
                                        delx = (if ind1 < 0 as ::core::ffi::c_int {
                                            *dxedge.offset(
                                                prm1st[0 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt1st[0 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind1)
                                                    as isize,
                                            )
                                        }) + (if ind2 < 0 as ::core::ffi::c_int {
                                            *dxedge.offset(
                                                prm2nd[0 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt2nd[0 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind2)
                                                    as isize,
                                            )
                                        }) - ((if ind3 < 0 as ::core::ffi::c_int {
                                            *dxedge.offset(
                                                prm1st[1 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt1st[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind3)
                                                    as isize,
                                            )
                                        }) + (if ind4 < 0 as ::core::ffi::c_int {
                                            *dxedge.offset(
                                                prm2nd[1 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt2nd[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind4)
                                                    as isize,
                                            )
                                        }));
                                        dely = (if ind1 < 0 as ::core::ffi::c_int {
                                            *dyedge.offset(
                                                prm1st[0 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt1st[0 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind1
                                                    + 1 as ::core::ffi::c_int)
                                                    as isize,
                                            )
                                        }) + (if ind2 < 0 as ::core::ffi::c_int {
                                            *dyedge.offset(
                                                prm2nd[0 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt2nd[0 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind2
                                                    + 1 as ::core::ffi::c_int)
                                                    as isize,
                                            )
                                        }) - ((if ind3 < 0 as ::core::ffi::c_int {
                                            *dyedge.offset(
                                                prm1st[1 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt1st[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind3
                                                    + 1 as ::core::ffi::c_int)
                                                    as isize,
                                            )
                                        }) + (if ind4 < 0 as ::core::ffi::c_int {
                                            *dyedge.offset(
                                                prm2nd[1 as ::core::ffi::c_int as usize] as isize,
                                            )
                                        } else {
                                            *altDxys.offset(
                                                (alt2nd[1 as ::core::ffi::c_int as usize]
                                                    + 2 as ::core::ffi::c_int * ind4
                                                    + 1 as ::core::ffi::c_int)
                                                    as isize,
                                            )
                                        }));
                                        err = sqrt(
                                            (delx * delx + dely * dely) as ::core::ffi::c_double,
                                        )
                                            as ::core::ffi::c_float;
                                        if firstErr
                                            < 0 as ::core::ffi::c_int as ::core::ffi::c_float
                                        {
                                            firstErr = err;
                                        }
                                        if err < minErr {
                                            minErr = err;
                                            minInd1 = ind1;
                                            minInd2 = ind2;
                                            minInd3 = ind3;
                                            minInd4 = ind4;
                                        }
                                    }
                                }
                                ind4 += 1;
                            }
                            ind3 += 1;
                        }
                        ind2 += 1;
                    }
                    ind1 += 1;
                }
                if firstErr > errThresh && minErr < firstErr * reduceFac && minErr <= newThresh {
                    if minInd1 >= 0 as ::core::ffi::c_int {
                        ftmp = *dxedge.offset(prm1st[0 as ::core::ffi::c_int as usize] as isize);
                        *dxedge.offset(prm1st[0 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt1st[0 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd1)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt1st[0 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd1)
                                as isize,
                        ) = ftmp;
                        ftmp = *dyedge.offset(prm1st[0 as ::core::ffi::c_int as usize] as isize);
                        *dyedge.offset(prm1st[0 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt1st[0 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd1
                                    + 1 as ::core::ffi::c_int)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt1st[0 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd1
                                + 1 as ::core::ffi::c_int) as isize,
                        ) = ftmp;
                        if !fixedEdges.is_null() {
                            let fresh6 = *numFixed;
                            *numFixed = *numFixed + 1;
                            *fixedEdges.offset(fresh6 as isize) =
                                prm1st[0 as ::core::ffi::c_int as usize];
                        }
                    }
                    if minInd2 >= 0 as ::core::ffi::c_int {
                        ftmp = *dxedge.offset(prm2nd[0 as ::core::ffi::c_int as usize] as isize);
                        *dxedge.offset(prm2nd[0 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt2nd[0 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd2)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt2nd[0 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd2)
                                as isize,
                        ) = ftmp;
                        ftmp = *dyedge.offset(prm2nd[0 as ::core::ffi::c_int as usize] as isize);
                        *dyedge.offset(prm2nd[0 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt2nd[0 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd2
                                    + 1 as ::core::ffi::c_int)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt2nd[0 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd2
                                + 1 as ::core::ffi::c_int) as isize,
                        ) = ftmp;
                        if !fixedEdges.is_null() {
                            let fresh7 = *numFixed;
                            *numFixed = *numFixed + 1;
                            *fixedEdges.offset(fresh7 as isize) =
                                prm2nd[0 as ::core::ffi::c_int as usize];
                        }
                    }
                    if minInd3 >= 0 as ::core::ffi::c_int {
                        ftmp = *dxedge.offset(prm1st[1 as ::core::ffi::c_int as usize] as isize);
                        *dxedge.offset(prm1st[1 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt1st[1 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd3)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt1st[1 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd3)
                                as isize,
                        ) = ftmp;
                        ftmp = *dyedge.offset(prm1st[1 as ::core::ffi::c_int as usize] as isize);
                        *dyedge.offset(prm1st[1 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt1st[1 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd3
                                    + 1 as ::core::ffi::c_int)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt1st[1 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd3
                                + 1 as ::core::ffi::c_int) as isize,
                        ) = ftmp;
                        if !fixedEdges.is_null() {
                            let fresh8 = *numFixed;
                            *numFixed = *numFixed + 1;
                            *fixedEdges.offset(fresh8 as isize) =
                                prm1st[1 as ::core::ffi::c_int as usize];
                        }
                    }
                    if minInd4 >= 0 as ::core::ffi::c_int {
                        ftmp = *dxedge.offset(prm2nd[1 as ::core::ffi::c_int as usize] as isize);
                        *dxedge.offset(prm2nd[1 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt2nd[1 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd4)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt2nd[1 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd4)
                                as isize,
                        ) = ftmp;
                        ftmp = *dyedge.offset(prm2nd[1 as ::core::ffi::c_int as usize] as isize);
                        *dyedge.offset(prm2nd[1 as ::core::ffi::c_int as usize] as isize) =
                            *altDxys.offset(
                                (alt2nd[1 as ::core::ffi::c_int as usize]
                                    + 2 as ::core::ffi::c_int * minInd4
                                    + 1 as ::core::ffi::c_int)
                                    as isize,
                            );
                        *altDxys.offset(
                            (alt2nd[1 as ::core::ffi::c_int as usize]
                                + 2 as ::core::ffi::c_int * minInd4
                                + 1 as ::core::ffi::c_int) as isize,
                        ) = ftmp;
                        if !fixedEdges.is_null() {
                            let fresh9 = *numFixed;
                            *numFixed = *numFixed + 1;
                            *fixedEdges.offset(fresh9 as isize) =
                                prm2nd[1 as ::core::ffi::c_int as usize];
                        }
                    }
                }
            }
            ivar += idir;
        }
        varStart = nvar / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
        varEnd = nvar - 1 as ::core::ffi::c_int;
        idir += 2 as ::core::ffi::c_int;
    }
    fflush(stdout);
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pick_alternative_shifts_fortran(
    mut ivarpc: *mut ::core::ffi::c_int,
    mut nvar: *mut ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut dxedge: *mut ::core::ffi::c_float,
    mut dyedge: *mut ::core::ffi::c_float,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut edgeStep: *mut ::core::ffi::c_int,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pcStep: *mut ::core::ffi::c_int,
    mut fort: *mut ::core::ffi::c_int,
    mut altDxys: *mut ::core::ffi::c_float,
    mut numAlts: *mut ::core::ffi::c_int,
    mut altIxy: *mut ::core::ffi::c_int,
    mut errThresh: *mut ::core::ffi::c_float,
    mut reduceFac: *mut ::core::ffi::c_float,
    mut newThresh: *mut ::core::ffi::c_float,
    mut fixedEdges: *mut ::core::ffi::c_int,
    mut numFixed: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return pick_alternative_shifts(
        ivarpc, *nvar, indvar, dxedge, dyedge, pieceLower, pieceUpper, ifskipEdge, *edgeStep,
        edgeLower, edgeUpper, *pcStep, *fort, altDxys, *numAlts, *altIxy, *errThresh, *reduceFac,
        *newThresh, fixedEdges, numFixed,
    );
}
unsafe extern "C" fn find_lowest_three(
    mut val: ::core::ffi::c_float,
    mut ind: ::core::ffi::c_int,
    mut lowest: *mut ::core::ffi::c_float,
    mut second: *mut ::core::ffi::c_float,
    mut third: *mut ::core::ffi::c_float,
) {
    if ind == 0 {
        *lowest = val;
    } else if val < *lowest {
        if ind > 1 as ::core::ffi::c_int {
            *third = *second;
        }
        *second = *lowest;
        *lowest = val;
        return;
    } else if ind <= 1 as ::core::ffi::c_int || val < *second {
        if ind > 1 as ::core::ffi::c_int {
            *third = *second;
        }
        *second = val;
    } else if ind <= 2 as ::core::ffi::c_int || val < *third {
        *third = val;
    }
}
unsafe extern "C" fn initialize(
    mut ixpclist: *mut ::core::ffi::c_int,
    mut iypclist: *mut ::core::ffi::c_int,
    mut ivarpc: *mut ::core::ffi::c_int,
    mut edgeLower: *mut ::core::ffi::c_int,
    mut edgeUpper: *mut ::core::ffi::c_int,
    mut pieceLower: *mut ::core::ffi::c_int,
    mut pieceUpper: *mut ::core::ffi::c_int,
    mut ifskipEdge: *mut ::core::ffi::c_int,
    mut dxedge: *mut ::core::ffi::c_float,
    mut dyedge: *mut ::core::ffi::c_float,
    mut dxyEdge: *mut ::core::ffi::c_float,
    mut dxyvar: *mut ::core::ffi::c_float,
    mut neighInd: *mut ::core::ffi::c_int,
    mut neighWgt: *mut ::core::ffi::c_float,
    mut edgeDir: *mut ::core::ffi::c_uchar,
    mut neighList: *mut ::core::ffi::c_int,
    mut placed: *mut ::core::ffi::c_uchar,
    mut ivarToList: *mut ::core::ffi::c_int,
    mut listToVar: *mut ::core::ffi::c_int,
    mut indvar: *mut ::core::ffi::c_int,
    mut nvar: ::core::ffi::c_int,
    mut fort: ::core::ffi::c_int,
    mut idir: ::core::ffi::c_int,
    mut leaveInd: ::core::ffi::c_int,
    mut skipCrit: ::core::ffi::c_int,
    mut xyStep: ::core::ffi::c_int,
    mut edgeStep: ::core::ffi::c_int,
    mut pcStep: ::core::ffi::c_int,
) {
    let mut minxpc: ::core::ffi::c_int = 0;
    let mut minypc: ::core::ffi::c_int = 0;
    let mut maxxpc: ::core::ffi::c_int = 0;
    let mut maxypc: ::core::ffi::c_int = 0;
    let mut imin: ::core::ffi::c_int = 0;
    let mut numOnList: ::core::ffi::c_int = 0;
    let mut listInd: ::core::ffi::c_int = 0;
    let mut iedge: ::core::ffi::c_int = 0;
    let mut ipc: ::core::ffi::c_int = 0;
    let mut numNeigh: ::core::ffi::c_int = 0;
    let mut isign: ::core::ffi::c_int = 0;
    let mut iter: ::core::ffi::c_int = 0;
    let mut nay: ::core::ffi::c_int = 0;
    let mut list: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut ivar: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ixy: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut lowup: ::core::ffi::c_int = 0;
    let mut neighpc: ::core::ffi::c_int = 0;
    let mut neighvar: ::core::ffi::c_int = 0;
    let mut distmin: ::core::ffi::c_float = 0.;
    let mut dist: ::core::ffi::c_float = 0.;
    let mut xmovemax: ::core::ffi::c_float = 0.;
    let mut ymovemax: ::core::ffi::c_float = 0.;
    let mut xsum: ::core::ffi::c_float = 0.;
    let mut ysum: ::core::ffi::c_float = 0.;
    let mut dx: ::core::ffi::c_float = 0.;
    let mut dy: ::core::ffi::c_float = 0.;
    let mut ex: ::core::ffi::c_float = 0.;
    let mut ey: ::core::ffi::c_float = 0.;
    let mut bigint: ::core::ffi::c_int = 100000000 as ::core::ffi::c_int;
    let mut dxyDim: ::core::ffi::c_int = if !dyedge.is_null() {
        2 as ::core::ffi::c_int
    } else {
        1 as ::core::ffi::c_int
    };
    ivar = 0 as ::core::ffi::c_int;
    while ivar < nvar {
        *placed.offset(ivar as isize) = 0 as ::core::ffi::c_uchar;
        *ivarToList.offset(ivar as isize) = -(1 as ::core::ffi::c_int);
        ivar += 1;
    }
    numNeigh = 0 as ::core::ffi::c_int;
    numOnList = 0 as ::core::ffi::c_int;
    loop {
        minxpc = bigint;
        minypc = bigint;
        maxxpc = -bigint;
        maxypc = -bigint;
        ivar = 0 as ::core::ffi::c_int;
        while ivar < nvar {
            if *placed.offset(ivar as isize) == 0 {
                minxpc =
                    if minxpc < *ixpclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize) {
                        minxpc
                    } else {
                        *ixpclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    };
                maxxpc =
                    if maxxpc > *ixpclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize) {
                        maxxpc
                    } else {
                        *ixpclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    };
                minypc =
                    if minypc < *iypclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize) {
                        minypc
                    } else {
                        *iypclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    };
                maxypc =
                    if maxypc > *iypclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize) {
                        maxypc
                    } else {
                        *iypclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    };
            }
            ivar += 1;
        }
        if minxpc == bigint {
            break;
        }
        distmin = 1.0e30f32;
        ivar = 0 as ::core::ffi::c_int;
        while ivar < nvar {
            if *placed.offset(ivar as isize) == 0 {
                dx = (*ixpclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    as ::core::ffi::c_double
                    - 0.5f64 * (maxxpc + minxpc) as ::core::ffi::c_double)
                    as ::core::ffi::c_float;
                dy = (*iypclist.offset((*ivarpc.offset(ivar as isize) - fort) as isize)
                    as ::core::ffi::c_double
                    - 0.5f64 * (maxypc + minypc) as ::core::ffi::c_double)
                    as ::core::ffi::c_float;
                dist = dx * dx + dy * dy;
                if dist < distmin {
                    distmin = dist;
                    imin = ivar;
                }
            }
            ivar += 1;
        }
        *ivarToList.offset(imin as isize) = numOnList;
        *listToVar.offset(numOnList as isize) = imin;
        let fresh3 = numOnList;
        numOnList = numOnList + 1;
        listInd = fresh3;
        while listInd < numOnList {
            xsum = 0.0f32;
            ysum = 0.0f32;
            nsum = 0 as ::core::ffi::c_int;
            ipc = *ivarpc.offset(*listToVar.offset(listInd as isize) as isize) - fort;
            *neighInd.offset(listInd as isize) = numNeigh;
            ixy = 0 as ::core::ffi::c_int;
            while ixy < 2 as ::core::ffi::c_int {
                lowup = 0 as ::core::ffi::c_int;
                while lowup < 2 as ::core::ffi::c_int {
                    isign = 2 as ::core::ffi::c_int * lowup - 1 as ::core::ffi::c_int;
                    if lowup != 0 {
                        iedge = *edgeUpper.offset((xyStep * ipc + pcStep * ixy) as isize) - fort;
                    } else {
                        iedge = *edgeLower.offset((xyStep * ipc + pcStep * ixy) as isize) - fort;
                    }
                    ind = xyStep * iedge + edgeStep * ixy;
                    if iedge >= 0 as ::core::ffi::c_int
                        && *ifskipEdge.offset(ind as isize) < skipCrit
                        && ind != leaveInd - fort
                    {
                        if lowup != 0 {
                            neighpc = *pieceUpper.offset(ind as isize) - fort;
                        } else {
                            neighpc = *pieceLower.offset(ind as isize) - fort;
                        }
                        neighvar = *indvar.offset(neighpc as isize) - fort;
                        if !(neighvar < 0 as ::core::ffi::c_int) {
                            if *ivarToList.offset(neighvar as isize) < 0 as ::core::ffi::c_int {
                                *listToVar.offset(numOnList as isize) = neighvar;
                                let fresh4 = numOnList;
                                numOnList = numOnList + 1;
                                *ivarToList.offset(neighvar as isize) = fresh4;
                            }
                            nay = *ivarToList.offset(neighvar as isize);
                            ex = (-idir * isign) as ::core::ffi::c_float
                                * *dxedge.offset(ind as isize);
                            *dxyEdge.offset((dxyDim * numNeigh) as isize) = ex;
                            if !dyedge.is_null() {
                                ey = (-idir * isign) as ::core::ffi::c_float
                                    * *dyedge.offset(ind as isize);
                                *dxyEdge.offset(
                                    (2 as ::core::ffi::c_int * numNeigh + 1 as ::core::ffi::c_int)
                                        as isize,
                                ) = ey;
                            }
                            if !neighWgt.is_null() {
                                *neighWgt.offset(numNeigh as isize) = 1.0f32;
                            }
                            *edgeDir.offset(numNeigh as isize) =
                                (lowup + 2 as ::core::ffi::c_int * ixy) as ::core::ffi::c_uchar;
                            let fresh5 = numNeigh;
                            numNeigh = numNeigh + 1;
                            *neighList.offset(fresh5 as isize) = nay;
                            if *placed.offset(neighvar as isize) != 0 {
                                xsum += *dxyvar.offset((dxyDim * nay) as isize) - ex;
                                if !dyedge.is_null() {
                                    ysum += *dxyvar.offset(
                                        (2 as ::core::ffi::c_int * nay + 1 as ::core::ffi::c_int)
                                            as isize,
                                    ) - ey;
                                }
                                nsum += 1;
                            }
                        }
                    }
                    lowup += 1;
                }
                ixy += 1;
            }
            if nsum != 0 {
                xsum /= nsum as ::core::ffi::c_float;
                ysum /= nsum as ::core::ffi::c_float;
            }
            *dxyvar.offset((dxyDim * listInd) as isize) = xsum;
            if !dyedge.is_null() {
                *dxyvar.offset(
                    (2 as ::core::ffi::c_int * listInd + 1 as ::core::ffi::c_int) as isize,
                ) = ysum;
            }
            *placed.offset(*listToVar.offset(listInd as isize) as isize) =
                1 as ::core::ffi::c_uchar;
            listInd += 1;
        }
    }
    *neighInd.offset(nvar as isize) = numNeigh;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_three_preserves_source_ordered_insertions() {
        unsafe {
            let mut low = 100.;
            let mut second = 100.;
            let mut third = 100.;
            find_lowest_three(4., 0, &mut low, &mut second, &mut third);
            find_lowest_three(2., 1, &mut low, &mut second, &mut third);
            find_lowest_three(3., 2, &mut low, &mut second, &mut third);
            assert_eq!((low, second, third), (2., 3., 4.));
        }
    }

    #[test]
    fn find_piece_shifts_solves_and_centers_a_two_piece_overlap() {
        unsafe {
            // Compact C layout: X edge 0 joins piece 0 to piece 1; all
            // other lower/upper edge slots are absent.
            let mut ivarpc = [0_i32, 1];
            let mut indvar = [0_i32, 1];
            let mut ixpclist = [0_i32, 1];
            let mut iypclist = [0_i32, 0];
            let mut piece_lower = [0_i32, 1];
            let mut edge_lower = [-1_i32, -1, 0, -1];
            let mut edge_upper = [0_i32, -1, -1, -1];
            let mut if_skip_edge = [0_i32, 0];
            let mut dxedge = [4.0_f32, 0.0];
            let mut dyedge = [0.0_f32, 0.0];
            let mut piece_upper = [1_i32, 0];
            let mut dxyvar = [0.0_f32; 4];
            let mut work = [0.0_f32; 64];
            let mut num_iter = 0_i32;
            let mut error_mean = 0.0_f32;
            let mut error_max = 0.0_f32;

            assert_eq!(
                find_piece_shifts(
                    ivarpc.as_mut_ptr(),
                    2,
                    indvar.as_mut_ptr(),
                    ixpclist.as_mut_ptr(),
                    iypclist.as_mut_ptr(),
                    dxedge.as_mut_ptr(),
                    dyedge.as_mut_ptr(),
                    1,
                    piece_lower.as_mut_ptr(),
                    piece_upper.as_mut_ptr(),
                    if_skip_edge.as_mut_ptr(),
                    1,
                    dxyvar.as_mut_ptr(),
                    1,
                    edge_lower.as_mut_ptr(),
                    edge_upper.as_mut_ptr(),
                    1,
                    work.as_mut_ptr(),
                    0,
                    -1,
                    1,
                    0.0,
                    1.0e-6,
                    1.0e-6,
                    50,
                    1,
                    1,
                    &mut num_iter,
                    &mut error_mean,
                    &mut error_max,
                ),
                0
            );
            assert!(num_iter > 0);
            assert!((dxyvar[0] - 2.0).abs() < 1.0e-5);
            assert!((dxyvar[2] + 2.0).abs() < 1.0e-5);
            assert!(dxyvar[1].abs() < 1.0e-5 && dxyvar[3].abs() < 1.0e-5);
            assert!(error_mean.abs() < 1.0e-5 && error_max.abs() < 1.0e-5);

            assert_eq!(
                find_piece_shifts(
                    ivarpc.as_mut_ptr(),
                    2,
                    indvar.as_mut_ptr(),
                    ixpclist.as_mut_ptr(),
                    iypclist.as_mut_ptr(),
                    dxedge.as_mut_ptr(),
                    dyedge.as_mut_ptr(),
                    1,
                    piece_lower.as_mut_ptr(),
                    piece_upper.as_mut_ptr(),
                    if_skip_edge.as_mut_ptr(),
                    1,
                    dxyvar.as_mut_ptr(),
                    2,
                    edge_lower.as_mut_ptr(),
                    edge_upper.as_mut_ptr(),
                    1,
                    work.as_mut_ptr(),
                    0,
                    -1,
                    1,
                    0.0,
                    1.0e-6,
                    1.0e-6,
                    1,
                    1,
                    1,
                    &mut num_iter,
                    &mut error_mean,
                    &mut error_max,
                ),
                1
            );
        }
    }

    #[test]
    fn find_piece_scalings_solves_and_centers_a_two_piece_overlap() {
        unsafe {
            let mut ivarpc = [0_i32, 1];
            let mut indvar = [0_i32, 1];
            let mut ixpclist = [0_i32, 1];
            let mut iypclist = [0_i32, 0];
            let mut piece_lower = [0_i32, 1];
            let mut edge_lower = [-1_i32, -1, 0, -1];
            let mut edge_upper = [0_i32, -1, -1, -1];
            let mut if_skip_edge = [0_i32, 0];
            let mut dden_edge = [0.2_f32, 0.0];
            let mut piece_upper = [1_i32, 0];
            let mut dden_var = [0.0_f32; 2];
            let mut work = [0.0_f32; 64];
            let mut num_iter = 0_i32;
            let mut error_mean = 0.0_f32;
            let mut error_max = 0.0_f32;

            assert_eq!(
                find_piece_scalings(
                    ivarpc.as_mut_ptr(),
                    2,
                    indvar.as_mut_ptr(),
                    ixpclist.as_mut_ptr(),
                    iypclist.as_mut_ptr(),
                    dden_edge.as_mut_ptr(),
                    1,
                    piece_lower.as_mut_ptr(),
                    piece_upper.as_mut_ptr(),
                    if_skip_edge.as_mut_ptr(),
                    1,
                    dden_var.as_mut_ptr(),
                    edge_lower.as_mut_ptr(),
                    edge_upper.as_mut_ptr(),
                    1,
                    work.as_mut_ptr(),
                    0,
                    -1,
                    1,
                    1.0e-6,
                    1.0e-6,
                    50,
                    1,
                    1,
                    &mut num_iter,
                    &mut error_mean,
                    &mut error_max,
                ),
                0
            );
            assert!(num_iter > 0);
            assert!((dden_var[0] - 0.1).abs() < 1.0e-5);
            assert!((dden_var[1] + 0.1).abs() < 1.0e-5);
            assert!(error_mean.abs() < 1.0e-5 && error_max.abs() < 1.0e-5);

            assert_eq!(
                find_piece_scalings(
                    ivarpc.as_mut_ptr(),
                    2,
                    indvar.as_mut_ptr(),
                    ixpclist.as_mut_ptr(),
                    iypclist.as_mut_ptr(),
                    dden_edge.as_mut_ptr(),
                    1,
                    piece_lower.as_mut_ptr(),
                    piece_upper.as_mut_ptr(),
                    if_skip_edge.as_mut_ptr(),
                    1,
                    dden_var.as_mut_ptr(),
                    edge_lower.as_mut_ptr(),
                    edge_upper.as_mut_ptr(),
                    2,
                    work.as_mut_ptr(),
                    0,
                    -1,
                    1,
                    1.0e-6,
                    1.0e-6,
                    1,
                    1,
                    1,
                    &mut num_iter,
                    &mut error_mean,
                    &mut error_max,
                ),
                1
            );
        }
    }

    #[test]
    fn pick_alternative_shifts_replaces_the_single_bad_cycle_edge() {
        unsafe {
            // 2 by 2 piece square.  Compact edge slots are X0, Y0, X1, Y1;
            // only X0 has an inconsistent displacement, and its one supplied
            // alternative closes the cycle exactly.
            let mut ivarpc = [0_i32, 1, 2, 3];
            let mut indvar = [0_i32, 1, 2, 3];
            let mut dxedge = [5.0_f32, 0.0, 1.0, 0.0];
            let mut dyedge = [0.0_f32; 4];
            let mut piece_lower = [0_i32, 0, 2, 1];
            let mut piece_upper = [1_i32, 2, 3, 3];
            let mut if_skip_edge = [0_i32; 4];
            let mut edge_lower = [-1_i32, -1, 0, -1, -1, 0, 1, 1];
            let mut edge_upper = [0_i32, 0, -1, 1, 1, -1, -1, -1];
            // X alternatives occupy 0..4 and Y alternatives begin at 4.
            let mut alternatives = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let mut fixed_edges = [0_i32; 4];
            let mut num_fixed = 0_i32;

            assert_eq!(
                pick_alternative_shifts(
                    ivarpc.as_mut_ptr(),
                    4,
                    indvar.as_mut_ptr(),
                    dxedge.as_mut_ptr(),
                    dyedge.as_mut_ptr(),
                    piece_lower.as_mut_ptr(),
                    piece_upper.as_mut_ptr(),
                    if_skip_edge.as_mut_ptr(),
                    1,
                    edge_lower.as_mut_ptr(),
                    edge_upper.as_mut_ptr(),
                    1,
                    0,
                    alternatives.as_mut_ptr(),
                    1,
                    4,
                    1.0,
                    0.5,
                    0.1,
                    fixed_edges.as_mut_ptr(),
                    &mut num_fixed,
                ),
                0
            );
            assert_eq!(num_fixed, 1);
            assert_eq!(fixed_edges[0], 0);
            assert_eq!(dxedge[0], 1.0);
            assert_eq!(alternatives[0], 5.0);
        }
    }
}
