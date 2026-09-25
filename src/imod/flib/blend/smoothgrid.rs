//! Translation of `IMOD/flib/blend/smoothgrid.f`: `smoothgrid`.
//!
//! No `use blendvars`, so no `BlendVars`.  Grid arrays dimensioned
//! `(ixgdim, iygdim)` are flat column-major slices, as in `edgesubs.rs`:
//! `dxgrid(ix, iy)` is `dxgrid[(ix - 1) + ixgdim * (iy - 1)]`.

use crate::imod::flib::blend::edgesubs::localmean;
use crate::imod::flib::subrs::statsubs::polyfit::polyfit;
use crate::imod::flib::subrs::statsubs::polyterm::poly_term;
use crate::imod::libcfshr::regression::multregress;

/// Original: `subroutine smoothgrid` (`smoothgrid.f:26`).
///
/// SMOOTHGRID smooths the 2 dimension arrays in DXGRID, DYGRID and
/// DDENGRID, based on values in D[XY]GRID and the array SDGRID.  IXGDIM,
/// IYGDIM are the dimensions of the arrays.  NXGRID, NYGRID are the
/// number of values in the X (1st) and Y (2nd) dimensions.  At each grid
/// position, it computes the differences between SDGRID and the local
/// average of SDGRID, and between the displacement specified by
/// D[XY]GRID and the local average of such displacements.  It computes
/// the mean and standard deviation of these two differences over all
/// grid positions.  Then, every position with a difference between
/// SDGRID and the local mean greater than SDCRIT standard deviations
/// more than the mean such difference, or with the difference between
/// the displacement and the local mean greater than DEVCRIT standard
/// deviations more than the mean such difference, will have their
/// DXGRID, DYGRID, and DDENGRID replaced by the local mean.  Next, it
/// fits a polynomial in X and Y to series of local parts of the DXGRID
/// and DYGRID arrays.  NORDER is the polynomial order (e.g. 2 means use
/// terms in X, Y, X**2, Y**2 and XY), NXFIT and NYFIT are the number of
/// X and Y positions to include in each fit, and N[XY]SKIP specifies the
/// number of X and Y positions to skip between adjacent polynomial fits.
/// Each point is closest to the center of some polynomial fitting area;
/// DXGRID, DYGRID, and DDENGRID for that point are replaced by the value
/// calculated from that polynomial for that position.
///
/// Translation notes:
/// - `dxmean`, `dymean`, `denmean` are uninitialised locals that
///   `localmean` leaves untouched when a position has no usable neighbour
///   (`edgesubs.f:284`), so they carry the previous position's values; the
///   first position's are stack residue in native and 0 here.
/// - `ix`/`iy` keep Fortran DO-variable semantics (one past the last value
///   on normal exit): the 1-D branch reads `ix` (`nxgrid == 1`) or `iy`
///   after an inner loop that may run zero times.
/// - `dxnew`/`dynew`/`denew` and the fixed-size `xr`, `bb`, `vect`, ... are
///   uninitialised automatic arrays; zero-filled here.  The fit loops write
///   every element that the final copy reads for any grid the source covers.
/// - `call polyfit` discards the function result, as the source does.
/// - `multRegress` resolves to the `multregress_` wrapper
///   (`regression.c:317`), which subtracts 1 from `wgtCol`.
#[allow(clippy::too_many_arguments)]
pub fn smoothgrid(
    dxgrid: &mut [f32],
    dygrid: &mut [f32],
    sdgrid: &[f32],
    ddengrid: &mut [f32],
    ixgdim: i32,
    iygdim: i32,
    nxgrid: i32,
    nygrid: i32,
    sdcrit: f32,
    devcrit: f32,
    nxfit: i32,
    nyfit: i32,
    norder: i32,
    nxskip: i32,
    nyskip: i32,
) {
    const IDIM: i32 = 100;
    const MSIZ: i32 = 50;
    let g = |i: i32, j: i32| ((i - 1) + ixgdim * (j - 1)) as usize;
    let n = |i: i32, j: i32| ((i - 1) + nxgrid * (j - 1)) as usize;
    let mut xr = vec![0.0_f32; (MSIZ * IDIM) as usize];
    let mut xm = [0.0_f32; MSIZ as usize];
    let mut sd = [0.0_f32; MSIZ as usize];
    let mut ssd = vec![0.0_f32; (MSIZ * MSIZ) as usize];
    let mut bb = [0.0_f32; (MSIZ * 3) as usize];
    let mut vect = [0.0_f32; MSIZ as usize];
    let mut c = [0.0_f32; 3];
    let mut b = [0.0_f32; MSIZ as usize];
    let mut b1 = [0.0_f32; MSIZ as usize];
    let mut b2 = [0.0_f32; MSIZ as usize];
    let mut b3 = [0.0_f32; MSIZ as usize];
    let nnew = if nxgrid > 0 && nygrid > 0 {
        (nxgrid * nygrid) as usize
    } else {
        0
    };
    let mut dxnew = vec![0.0_f32; nnew];
    let mut dynew = vec![0.0_f32; nnew];
    let mut denew = vec![0.0_f32; nnew];
    let mut sdmax: f32;
    let mut sdsum: f32;
    let mut sdsq: f32;
    let mut devsum: f32;
    let mut devsqsum: f32;
    let mut dxmean = 0.0_f32;
    let mut dymean = 0.0_f32;
    let mut denmean = 0.0_f32;
    let mut dev: f32;
    let mut devvary: f32;
    let mut bint = 0.0_f32;
    let mut ix: i32 = 0;
    let mut iy: i32 = 0;
    let mut nnear = 0_i32;
    let length: i32;
    let nfitpt: i32;
    let mut ihi: i32;
    let mut ilo: i32;
    let mut nfit: i32;
    let mut iord: i32;
    let nxftl: i32;
    let nyftl: i32;
    let ixsolvsub: i32;
    let ixsolvadd: i32;
    let ixsolvstr: i32;
    let ixsolvend: i32;
    let ixsolspan: i32;
    let nxsolve: i32;
    let ixfitsub: i32;
    let ixfitadd: i32;
    let iysolvsub: i32;
    let iysolvadd: i32;
    let iysolvstr: i32;
    let iysolvend: i32;
    let iysolspan: i32;
    let nysolve: i32;
    let iyfitsub: i32;
    let iyfitadd: i32;
    let nindep: i32;
    let idepen1: i32;
    let idepen2: i32;
    let idepen3: i32;
    let mut ixfitend: i32;
    let mut ixcen: i32;
    let mut ixfitstr: i32;
    let mut ixst: i32;
    let mut ixnd: i32;
    let mut iyfitend: i32;
    let mut iycen: i32;
    let mut iyfitstr: i32;
    let mut iyst: i32;
    let mut iynd: i32;
    let mut npnts: i32;
    let mut ind: i32;
    let nsum: i32;
    let mut xsum: f32;
    let mut ysum: f32;
    let mut dsum: f32;
    let mut devsq: f32;
    let devsd: f32;
    let devmean: f32;
    let sdmean: f32;
    let sdsd: f32;
    let mut sdvary: f32;
    //
    // intercept the case of no overlap
    //
    if sdgrid[g(1, 1)] < 0. {
        return;
    }
    //
    // take mean and sd of the sd's and of the deviations from local means
    sdmax = 0.;
    sdsum = 0.;
    sdsq = 0.;
    devsum = 0.;
    devsqsum = 0.;
    //
    ix = 1;
    while ix <= nxgrid {
        iy = 1;
        while iy <= nygrid {
            sdsum += sdgrid[g(ix, iy)];
            sdsq += sdgrid[g(ix, iy)] * sdgrid[g(ix, iy)];
            // `max(sdmax, sdgrid)`; the result is only read by a
            // commented-out write.
            sdmax = if sdgrid[g(ix, iy)] > sdmax {
                sdgrid[g(ix, iy)]
            } else {
                sdmax
            };
            localmean(
                dxgrid,
                dygrid,
                sdgrid,
                ddengrid,
                ixgdim,
                iygdim,
                nxgrid,
                nygrid,
                ix,
                iy,
                &mut dxmean,
                &mut dymean,
                &mut denmean,
                &mut nnear,
            );
            devsq = (dxmean - dxgrid[g(ix, iy)]) * (dxmean - dxgrid[g(ix, iy)])
                + (dymean - dygrid[g(ix, iy)]) * (dymean - dygrid[g(ix, iy)]);
            devsqsum += devsq;
            devsum += devsq.sqrt();
            iy += 1;
        }
        ix += 1;
    }
    let _ = sdmax;
    //
    nsum = nxgrid * nygrid;
    sdmean = sdsum / nsum as f32;
    sdsd = ((sdsq - nsum as f32 * (sdmean * sdmean)) / (nsum as f32 - 1.)).sqrt();
    devmean = devsum / nsum as f32;
    devsd = ((devsqsum - nsum as f32 * (devmean * devmean)) / (nsum as f32 - 1.)).sqrt();
    //
    ix = 1;
    while ix <= nxgrid {
        iy = 1;
        while iy <= nygrid {
            sdvary = (sdgrid[g(ix, iy)] - sdmean) / sdsd;
            localmean(
                dxgrid,
                dygrid,
                sdgrid,
                ddengrid,
                ixgdim,
                iygdim,
                nxgrid,
                nygrid,
                ix,
                iy,
                &mut dxmean,
                &mut dymean,
                &mut denmean,
                &mut nnear,
            );
            dev = ((dxmean - dxgrid[g(ix, iy)]) * (dxmean - dxgrid[g(ix, iy)])
                + (dymean - dygrid[g(ix, iy)]) * (dymean - dygrid[g(ix, iy)]))
            .sqrt();
            devvary = (dev - devmean) / devsd;
            if sdvary > sdcrit || devvary > devcrit {
                dxgrid[g(ix, iy)] = dxmean;
                dygrid[g(ix, iy)] = dymean;
                ddengrid[g(ix, iy)] = denmean;
            }
            iy += 1;
        }
        ix += 1;
    }

    if nxgrid == 1 || nygrid == 1 {
        length = if nxgrid > nygrid { nxgrid } else { nygrid };
        let mx = if nxfit > nyfit { nxfit } else { nyfit };
        nfitpt = if mx < length { mx } else { length };
        for ixy in 1..=length {
            ihi = if ixy + nfitpt / 2 < length {
                ixy + nfitpt / 2
            } else {
                length
            };
            ilo = if 1 > ihi + 1 - nfitpt {
                1
            } else {
                ihi + 1 - nfitpt
            };
            ihi = if length < ilo + nfitpt - 1 {
                length
            } else {
                ilo + nfitpt - 1
            };
            nfit = ihi + 1 - ilo;
            iord = norder;
            if nfit < 7 {
                iord = if norder < 2 { norder } else { 2 };
            }
            if nfit < 5 {
                iord = if norder < 1 { norder } else { 1 };
            }
            ind = 1;
            for i in ilo..=ihi {
                if nxgrid == 1 {
                    ix = 1;
                    iy = i;
                } else {
                    ix = i;
                    iy = 1;
                }
                b[(ind - 1) as usize] = (i - ixy) as f32;
                b1[(ind - 1) as usize] = dxgrid[g(ix, iy)];
                b2[(ind - 1) as usize] = dygrid[g(ix, iy)];
                b3[(ind - 1) as usize] = ddengrid[g(ix, iy)];
                ind += 1;
            }
            if nxgrid == 1 {
                iy = ixy;
            } else {
                ix = ixy;
            }
            polyfit(&b, &b1, nfit, iord, &mut vect, &mut bint);
            ind = ix + (iy - 1) * nxgrid;
            let _ = ind;
            dxnew[n(ix, iy)] = bint;
            polyfit(&b, &b2, nfit, iord, &mut vect, &mut bint);
            dynew[n(ix, iy)] = bint;
            polyfit(&b, &b3, nfit, iord, &mut vect, &mut bint);
            denew[n(ix, iy)] = bint;
        }
    } else {
        nxftl = if nxfit < nxgrid { nxfit } else { nxgrid };
        nyftl = if nyfit < nygrid { nyfit } else { nygrid };
        // now set up sub-grid on which to do regressions
        ixsolvsub = nxftl / 2; //to get starting, ending x to
        ixsolvadd = nxftl - 1 - ixsolvsub; //include in solution
        ixsolvstr = ixsolvsub + 1; //first position
        ixsolvend = nxgrid - ixsolvadd; //last position
        ixsolspan = ixsolvend + 1 - ixsolvstr; //span of solution points
        nxsolve = (ixsolspan + nxskip - 1) / nxskip; //# of positions to solve
        ixfitsub = nxskip / 2; //to get starting, ending x
        ixfitadd = nxskip - 1 - ixfitsub; //values to replace with fit
        let _ = ixfitsub;
        //
        iysolvsub = nyftl / 2; //to get starting, ending y to
        iysolvadd = nyftl - 1 - iysolvsub; //include in solution
        iysolvstr = iysolvsub + 1; //first position
        iysolvend = nygrid - iysolvadd; //last position
        iysolspan = iysolvend + 1 - iysolvstr; //span of solution points
        nysolve = (iysolspan + nyskip - 1) / nyskip; //# of positions to solve
        iyfitsub = nyskip / 2; //to get starting, ending y
        iyfitadd = nyskip - 1 - iyfitsub; //values to replace with fit
        let _ = iyfitsub;
        //
        nindep = norder * (norder + 3) / 2; //# of independent variables
        idepen1 = nindep + 1; //index of independent var 1
        idepen2 = nindep + 2; //index of independent var 2
        idepen3 = nindep + 3;
        let xri = |i: i32, j: i32| ((i - 1) + MSIZ * (j - 1)) as usize;
        ixfitend = 0;
        for ixsol in 1..=nxsolve {
            ixcen = if (ixsol - 1) * nxskip + ixsolvstr < ixsolvend {
                (ixsol - 1) * nxskip + ixsolvstr
            } else {
                ixsolvend
            }; //center position
            // take starting position to replace with fits 1 past last end
            ixfitstr = ixfitend + 1;
            // limit ending position by last solution point
            ixfitend = if ixcen + ixfitadd < ixsolvend - 1 {
                ixcen + ixfitadd
            } else {
                ixsolvend - 1
            };
            // but if this is last point, go to edge
            if ixsol == nxsolve {
                ixfitend = nxgrid;
            }
            ixst = ixcen - ixsolvsub;
            ixnd = ixcen + ixsolvadd;
            //
            iyfitend = 0;
            for iysol in 1..=nysolve {
                iycen = if (iysol - 1) * nyskip + iysolvstr < iysolvend {
                    (iysol - 1) * nyskip + iysolvstr
                } else {
                    iysolvend
                }; //center position
                // take starting position to replace with fits 1 past last end
                iyfitstr = iyfitend + 1;
                // limit ending position by last solution point
                iyfitend = if iycen + iyfitadd < iysolvend - 1 {
                    iycen + iyfitadd
                } else {
                    iysolvend - 1
                };
                // but if this is last point, go to edge
                if iysol == nysolve {
                    iyfitend = nygrid;
                }
                iyst = iycen - iysolvsub;
                iynd = iycen + iysolvadd;
                //
                npnts = 0;

                for ix in ixst..=ixnd {
                    for iy in iyst..=iynd {
                        npnts += 1;
                        poly_term(ix - ixcen, iy - iycen, norder, &mut xr[xri(1, npnts)..]);
                        xr[xri(idepen1, npnts)] = dxgrid[g(ix, iy)];
                        xr[xri(idepen2, npnts)] = dygrid[g(ix, iy)];
                        xr[xri(idepen3, npnts)] = ddengrid[g(ix, iy)];
                        if npnts > 1 {
                            if xr[xri(idepen1, npnts)] == xr[xri(idepen1, npnts - 1)] {
                                xr[xri(idepen1, npnts)] = dxgrid[g(ix, iy)] + 0.01_f32;
                            }
                            if xr[xri(idepen2, npnts)] == xr[xri(idepen2, npnts - 1)] {
                                xr[xri(idepen2, npnts)] = dygrid[g(ix, iy)] + 0.01_f32;
                            }
                            if xr[xri(idepen3, npnts)] == xr[xri(idepen3, npnts - 1)] {
                                xr[xri(idepen3, npnts)] = ddengrid[g(ix, iy)] + 0.001_f32;
                            }
                        }
                    }
                }
                //
                // Solve for dx, dy, den as function of terms and compute fitted values
                multregress(
                    &xr,
                    &MSIZ,
                    &1,
                    &nindep,
                    &npnts,
                    &3,
                    &0,
                    &mut bb,
                    &MSIZ,
                    Some(&mut c),
                    &mut xm,
                    &mut sd,
                    &mut ssd,
                );
                for ix in ixfitstr..=ixfitend {
                    for iy in iyfitstr..=iyfitend {
                        poly_term(ix - ixcen, iy - iycen, norder, &mut vect);
                        xsum = c[0];
                        ysum = c[1];
                        dsum = c[2];
                        for i in 1..=nindep {
                            xsum += bb[xri(i, 1)] * vect[(i - 1) as usize];
                            ysum += bb[xri(i, 2)] * vect[(i - 1) as usize];
                            dsum += bb[xri(i, 3)] * vect[(i - 1) as usize];
                        }
                        dxnew[n(ix, iy)] = xsum;
                        dynew[n(ix, iy)] = ysum;
                        denew[n(ix, iy)] = dsum;
                    }
                }
            }
        }
    }
    //
    // replace new values into dxgrid and dygrid arrays
    //
    for iy in 1..=nygrid {
        for ix in 1..=nxgrid {
            dxgrid[g(ix, iy)] = dxnew[n(ix, iy)];
            dygrid[g(ix, iy)] = dynew[n(ix, iy)];
            ddengrid[g(ix, iy)] = denew[n(ix, iy)];
        }
    }
    //
}
