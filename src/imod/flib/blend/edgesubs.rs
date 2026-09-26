//! Translation of `IMOD/flib/blend/edgesubs.f`: `findedgefunc`, `setgridchars`,
//! `localmean`, `sdintscan` — the edge-function routines shared by
//! `blendmont` and `finddistort`.
//!
//! None of these units `use blendvars`, so none takes a `BlendVars`.  Grid
//! arrays dimensioned `(ixgdim, iygdim)` are flat column-major slices:
//! `dxgrid(ix, iy)` is `dxgrid[(ix - 1) + ixgdim * (iy - 1)]`.
//!
//! The Fortran calls `montSdCalc`/`montBigSearch`, which Fortran's case
//! folding resolves to the `montsdcalc_`/`montbigsearch_` wrappers
//! (`sdsearch.c:208-214`, `:317-322`); those subtract 1 from the four box
//! limits.  The translation calls those wrappers (`montsdcalc`,
//! `montbigsearch`) with the Fortran's 1-based box limits.

use crate::imod::libcfshr::sdsearch::{montbigsearch, montsdcalc};

/// Original: `subroutine findedgefunc` (`edgesubs.f:22`).
///
/// `nxgrid`/`nygrid` are `&mut` because the no-overlap branch raises them to
/// at least 1 (`edgesubs.f:42-43`).
#[allow(clippy::too_many_arguments)]
pub fn findedgefunc(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ixgridstr: i32,
    iygridstr: i32,
    ixofset: i32,
    iyofset: i32,
    nxgrid: &mut i32,
    nygrid: &mut i32,
    intxgrid: i32,
    intygrid: i32,
    ixboxsiz: i32,
    iyboxsiz: i32,
    intscan: i32,
    dxgrid: &mut [f32],
    dygrid: &mut [f32],
    sdgrid: &mut [f32],
    ddengrid: &mut [f32],
    ixgdim: i32,
    iygdim: i32,
) {
    let g = |ix: i32, iy: i32| ((ix - 1) + ixgdim * (iy - 1)) as usize;
    // Locals the source leaves undefined until `localmean` sets them; they are
    // only read when `nsum > 1`, where `localmean` has written them.
    let (mut dxmean, mut dymean, mut ddenmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut nsum = 0_i32;
    let (mut sdmin, mut ddenmin) = (0.0_f32, 0.0_f32);
    let (mut idxmin, mut idymin) = (0_i32, 0_i32);
    let (mut dxgr, mut dygr): (f32, f32);
    let mut nsteps: i32;
    let mut limstep: i32;
    //
    // intercept case of no overlap and make a null grid
    //
    if *nxgrid <= 0 || *nygrid == 0 {
        *nxgrid = 1.max(*nxgrid);
        *nygrid = 1.max(*nygrid);
        for iy in 1..=*nygrid {
            for ix in 1..=*nxgrid {
                dxgrid[g(ix, iy)] = 0.;
                dygrid[g(ix, iy)] = 0.;
                sdgrid[g(ix, iy)] = -1.;
                ddengrid[g(ix, iy)] = 0.;
            }
        }
        return;
    }
    //
    // set up grid arrays
    //
    for iy in 1..=*nygrid {
        for ix in 1..=*nxgrid {
            sdgrid[g(ix, iy)] = -1 as f32;
        }
    }
    let idxbase = ixofset - ixgridstr;
    let idybase = iyofset - iygridstr;
    //
    // start from the inside out: so need the center location of grid
    //
    let ixgrdcen = *nxgrid / 2;
    let iygrdcen = *nygrid / 2;
    //
    // set up loops from center out
    // i.e. outer loop for long dimension from center to end then from
    // center to start
    let mut ixdir = 1_i32;
    let mut ixloopst: i32;
    let mut ixloopnd: i32;
    if *nxgrid > *nygrid {
        ixloopst = ixgrdcen + 1;
        ixloopnd = *nxgrid;
    } else {
        ixloopst = iygrdcen + 1;
        ixloopnd = *nygrid;
    }
    for _ixloop in 1..=2 {
        // `do iouter=ixloopst,ixloopnd,ixdir`: the trip count is fixed at
        // loop entry, max(0, (end - start + step) / step).
        let outer_trips = 0.max((ixloopnd - ixloopst + ixdir) / ixdir);
        let mut iouter = ixloopst;
        for _ in 0..outer_trips {
            //
            // then inner loop for short dimension from center to end then from
            // center to start
            let mut iydir = 1_i32;
            let mut iyloopst: i32;
            let mut iyloopnd: i32;
            if *nxgrid > *nygrid {
                iyloopst = iygrdcen + 1;
                iyloopnd = *nygrid;
            } else {
                iyloopst = ixgrdcen + 1;
                iyloopnd = *nxgrid;
            }
            for _iyloop in 1..=2 {
                let inner_trips = 0.max((iyloopnd - iyloopst + iydir) / iydir);
                let mut inner = iyloopst;
                for _ in 0..inner_trips {
                    let ixgrid: i32;
                    let iygrid: i32;
                    if *nxgrid > *nygrid {
                        ixgrid = iouter;
                        iygrid = inner;
                    } else {
                        ixgrid = inner;
                        iygrid = iouter;
                    }
                    //
                    // to get starting dx, dy, search for and average from neighbors
                    //
                    let ixbox0 = 1.max(ixgridstr + (ixgrid - 1) * intxgrid - ixboxsiz / 2);
                    let ixbox1 = (nx - 1).min(ixbox0 + ixboxsiz - 1);
                    let iybox0 = 1.max(iygridstr + (iygrid - 1) * intygrid - iyboxsiz / 2);
                    let iybox1 = (ny - 1).min(iybox0 + iyboxsiz - 1);
                    localmean(
                        dxgrid,
                        dygrid,
                        sdgrid,
                        ddengrid,
                        ixgdim,
                        iygdim,
                        *nxgrid,
                        *nygrid,
                        ixgrid,
                        iygrid,
                        &mut dxmean,
                        &mut dymean,
                        &mut ddenmean,
                        &mut nsum,
                    );
                    //
                    // if only one neighbor (or none), do complete scan
                    //
                    if nsum <= 1 {
                        let idx0 = -intscan;
                        let idx1 = intscan;
                        let idy0 = -intscan;
                        let idy1 = intscan;
                        sdintscan(
                            array,
                            brray,
                            nx,
                            ny,
                            ixbox0,
                            iybox0,
                            ixbox1,
                            iybox1,
                            idx0 + idxbase,
                            idy0 + idybase,
                            idx1 + idxbase,
                            idy1 + idybase,
                            &mut sdmin,
                            &mut ddenmin,
                            &mut idxmin,
                            &mut idymin,
                        );
                        dxgr = idxmin as f32;
                        dygr = idymin as f32;
                    } else {
                        //
                        // otherwise take starting point as average
                        //
                        nsteps = 4;
                        dxgr = dxmean + idxbase as f32;
                        dygr = dymean + idybase as f32;
                    }
                    //
                    // now do the real search
                    //
                    nsteps = 4;
                    limstep = 6;
                    // `montBigSearch` resolves to the `montbigsearch_` wrapper
                    // (`sdsearch.c:208`), which passes the box limits minus 1.
                    let k = g(ixgrid, iygrid);
                    montbigsearch(
                        array,
                        brray,
                        nx,
                        ny,
                        ixbox0,
                        iybox0,
                        ixbox1,
                        iybox1,
                        &mut dxgr,
                        &mut dygr,
                        &mut sdgrid[k],
                        &mut ddengrid[k],
                        nsteps,
                        limstep,
                    );
                    dxgrid[k] = dxgr - idxbase as f32;
                    dygrid[k] = dygr - idybase as f32;
                    inner += iydir;
                }
                iydir = -1;
                iyloopnd = 1;
                if *nxgrid > *nygrid {
                    iyloopst = iygrdcen;
                } else {
                    iyloopst = ixgrdcen;
                }
            }
            iouter += ixdir;
        }
        ixdir = -1;
        ixloopnd = 1;
        if *nxgrid > *nygrid {
            ixloopst = ixgrdcen;
        } else {
            ixloopst = iygrdcen;
        }
    }
}

/// Original: `subroutine setgridchars` (`edgesubs.f:183`).
///
/// `nxy`, `noverlap`, `igridstr`, `iofset` are indexed by x-y; `iboxsiz`,
/// `indent`, `intgrid` by short-long.
#[allow(clippy::too_many_arguments)]
pub fn setgridchars(
    nxy: &[i32],
    noverlap: &[i32],
    iboxsiz: &[i32],
    indent: &[i32],
    intgrid: &[i32],
    ixy: i32,
    ixdispl: i32,
    iydispl: i32,
    limit_lo: i32,
    limit_hi: i32,
    nxgrid: &mut i32,
    nygrid: &mut i32,
    igridstr: &mut [i32],
    iofset: &mut [i32],
) {
    let mut ngrid = [0_i32; 2];
    let mut idispl = [0_i32; 2];
    let mut len = [0_i32; 2];
    let mut jndent = [0_i32; 2];
    let mut limlen = [0_i32; 2];
    //
    // 12/98 change -i[xy]displ to +i[xy]displ
    if ixy == 1 {
        idispl[0] = nxy[0] - noverlap[0] + ixdispl;
        idispl[1] = iydispl;
    } else {
        idispl[0] = ixdispl;
        idispl[1] = nxy[1] - noverlap[1] + iydispl;
    }
    let iyx = 3 - ixy;
    let ixy_u = (ixy - 1) as usize;
    let iyx_u = (iyx - 1) as usize;
    //
    // get length of overlap in short direction: limit to standard length
    //
    len[ixy_u] = noverlap[ixy_u].min(nxy[ixy_u] - idispl[ixy_u]);
    limlen[ixy_u] = len[ixy_u];
    // length of overlap in long direction: reduce by displacement
    len[iyx_u] = nxy[iyx_u] - idispl[iyx_u].abs();
    limlen[iyx_u] = len[iyx_u];
    //
    // Get limited length for getting extent if limitHi is nonzero
    if limit_hi != 0 {
        limlen[iyx_u] = iboxsiz[iyx_u].max((limit_hi + 1 - limit_lo) - idispl[iyx_u].abs());
    }
    //
    // calculate extent usable within box, # gridpoints, indent to 1st point
    // but keep extent from getting negative
    //
    let mut isl = ixy_u; //index to short-long variables
    for i in 0..2 {
        let indent_use = indent[isl].min((limlen[i] - iboxsiz[isl]) / 2);
        let nextent = limlen[i] - iboxsiz[isl] - 2 * indent_use;
        ngrid[i] = 1 + nextent / intgrid[isl];
        jndent[i] = indent_use + (iboxsiz[isl] + nextent % intgrid[isl]) / 2;
        isl = iyx_u;
    }
    if limit_hi != 0 && limit_hi >= limit_lo {
        jndent[iyx_u] += limit_lo;
    }
    //
    // calculate coordinates of grid within each frame
    //
    for i in 0..2 {
        let lapcen = (nxy[i] - idispl[i]) / 2; //offset to center of overlap
        let ihafgrid = jndent[i] - len[i] / 2; //from center to start of grid
        iofset[i] = lapcen + ihafgrid; //offset in frame 2
        igridstr[i] = (nxy[i] - lapcen) + ihafgrid; //offset in frame 1
    }
    //
    *nxgrid = ngrid[0];
    *nygrid = ngrid[1];
}

/// Original: `subroutine localmean` (`edgesubs.f:249`).
///
/// `dxmean`, `dymean`, `ddenmean` are left untouched when `nsum` is 0, as the
/// source returns before assigning them.
#[allow(clippy::too_many_arguments)]
pub fn localmean(
    dxgrid: &[f32],
    dygrid: &[f32],
    sdgrid: &[f32],
    ddengrid: &[f32],
    ixgdim: i32,
    _iygdim: i32,
    nxgrid: i32,
    nygrid: i32,
    ix: i32,
    iy: i32,
    dxmean: &mut f32,
    dymean: &mut f32,
    ddenmean: &mut f32,
    nsum: &mut i32,
) {
    let g = |i: i32, j: i32| ((i - 1) + ixgdim * (j - 1)) as usize;
    *nsum = 0;
    let mut dxsum = 0.0_f32;
    let mut dysum = 0.0_f32;
    let mut densum = 0.0_f32;
    for i in ix - 1..=ix + 1 {
        for j in iy - 1..=iy + 1 {
            if (i != ix || j != iy) && i >= 1 && i <= nxgrid && j >= 1 && j <= nygrid {
                if sdgrid[g(i, j)] >= 0. {
                    dxsum += dxgrid[g(i, j)];
                    dysum += dygrid[g(i, j)];
                    densum += ddengrid[g(i, j)];
                    *nsum += 1;
                }
            }
        }
    }
    if *nsum == 0 {
        return;
    }
    *dxmean = dxsum / *nsum as f32;
    *dymean = dysum / *nsum as f32;
    *ddenmean = densum / *nsum as f32;
}

/// Original: `subroutine sdintscan` (`edgesubs.f:293`).
///
/// The source has no `implicit none`; by the implicit rules `sdmin`,
/// `ddenmin`, `sd`, `dden` are `real` and `idxmin`, `idymin`, `idx`, `idy`
/// `integer`, which is what the translation declares.  `idxmin`, `idymin`,
/// `ddenmin` are only written on an improvement, so they are `&mut`.
#[allow(clippy::too_many_arguments)]
pub fn sdintscan(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ixbox0: i32,
    iybox0: i32,
    ixbox1: i32,
    iybox1: i32,
    idx0: i32,
    idy0: i32,
    idx1: i32,
    idy1: i32,
    sdmin: &mut f32,
    ddenmin: &mut f32,
    idxmin: &mut i32,
    idymin: &mut i32,
) {
    // `sd` and `dden` are implicit `real` locals, uninitialised in the
    // Fortran.  `montsdcalc` always writes `sd` but writes `dden` only when it
    // compared a pixel, and a comparison of no pixels returns `sd = 9999.`,
    // which is below `sdmin = 1.e10` — so a box entirely outside `brray` on
    // the first displacement stores whatever `dden` held (stack residue in
    // native), and on a later one the previous displacement's value (the
    // same on both sides, since `dden` persists across iterations).  Fixed
    // in translation (`BUGS.md`): `dden` starts at 0, so no uninitialised
    // value is stored; `findedgefunc`, the only caller, does not use
    // `ddenmin` in any case (`edgesubs.f:121-127`).
    let mut sd = 0.0_f32;
    let mut dden = 0.0_f32;
    //
    *sdmin = 1.0e10;
    for idy in idy0..=idy1 {
        for idx in idx0..=idx1 {
            // `montSdCalc` resolves to the `montsdcalc_` wrapper
            // (`sdsearch.c:317`), which passes the box limits minus 1.
            montsdcalc(
                array, brray, nx, ny, ixbox0, iybox0, ixbox1, iybox1, idx as f32, idy as f32,
                &mut sd, &mut dden,
            );
            if sd < *sdmin {
                *sdmin = sd;
                *idxmin = idx;
                *idymin = idy;
                *ddenmin = dden;
            }
        }
    }
}
