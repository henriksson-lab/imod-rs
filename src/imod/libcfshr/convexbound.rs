//! Translation of `IMOD/libcfshr/convexbound.c`.

use super::robuststat::rs_sort_indexed_floats;
use super::simplestat::avg_sd;

/// Matches static `cenorder` (`IMOD/libcfshr/convexbound.c:204`).
fn cenorder(
    sx: &[f32],
    sy: &[f32],
    npnts: i32,
    inddist: &mut [i32],
    cendist: &mut [f32],
    xcen: &mut f32,
    ycen: &mut f32,
) {
    let mut xsum: f64;
    let mut ysum: f64;

    /* find centroid of the points */
    xsum = 0.;
    ysum = 0.;
    for i in 0..npnts as usize {
        xsum += sx[inddist[i] as usize] as f64;
        ysum += sy[inddist[i] as usize] as f64;
    }
    *xcen = (xsum / npnts as f64) as f32;
    *ycen = (ysum / npnts as f64) as f32;

    /* get distances from center */
    for i in 0..npnts as usize {
        let ind = inddist[i] as usize;
        cendist[ind] =
            (sx[ind] - *xcen) * (sx[ind] - *xcen) + (sy[ind] - *ycen) * (sy[ind] - *ycen);
    }

    /* order pointers by distance */
    rs_sort_indexed_floats(cendist, inddist, npnts);
}

/// Matches static `angorder` (`IMOD/libcfshr/convexbound.c:238`).
fn angorder(
    sx: &[f32],
    sy: &[f32],
    nptuse: i32,
    inddist: &[i32],
    xcen: f32,
    ycen: f32,
    iangtopt: &mut [i32],
    ipttoang: &mut [i32],
    angle: &mut [f32],
) {
    for i in 0..nptuse as usize {
        let ind = inddist[i] as usize;
        angle[ind] = ((sy[ind] - ycen) as f64).atan2((sx[ind] - xcen) as f64) as f32;
        iangtopt[i] = ind as i32;
    }
    rs_sort_indexed_floats(angle, iangtopt, nptuse);
    for i in 0..nptuse {
        ipttoang[iangtopt[i as usize] as usize] = i;
    }
}

/// Matches `convexBound` (`IMOD/libcfshr/convexbound.c:41`).  `npnts` is
/// `sx.len()` and `maxverts` is `bx.len()`.
///
/// Two deviations, both where the C has no defined behaviour to reproduce:
/// with `npnts == 0` the C divides by zero in `nextpt` (`% nptuse`) and dies
/// of SIGFPE, so this returns with `nvert` 0; and `indstart`/`xmax` are
/// uninitialised in the C until a point with `sy < 1.e30` is seen (never, if
/// every `sy` is NaN), so they start at 0 here.
pub fn convex_bound(
    sx: &[f32],
    syin: &[f32],
    fracomit: f32,
    pad: f32,
    bx: &mut [f32],
    by: &mut [f32],
    nvert: &mut i32,
    xcen: &mut f32,
    ycen: &mut f32,
) {
    let npnts = sx.len() as i32;
    let maxverts = bx.len().min(by.len()) as i32;
    let mut nptuse: i32;
    let mut itry: i32;
    let mut indstart: usize = 0;
    let (mut avgx, mut sdx, mut semx) = (0f32, 0f32, 0f32);
    let (mut avgy, mut sdy, mut semy) = (0f32, 0f32, 0f32);
    let mut ymin: f32;
    let mut xmax: f32 = 0.;
    let mut padfrac: f32;
    if npnts == 0 {
        *nvert = 0;
        return;
    }

    let n = npnts as usize;
    let mut cendist = vec![0f32; n];
    let mut angtmp = vec![0f32; n];
    let mut sy = vec![0f32; n];
    let mut inddist = vec![0i32; n];
    let mut iangtopt = vec![0i32; n];
    let mut ipttoang = vec![0i32; n];
    let mut vertex = vec![0u8; n];

    /* scale y to have same sd as x if fracomit is negative */
    sdx = 1.;
    sdy = 1.;
    if fracomit < 0. {
        avg_sd(sx, npnts, &mut avgx, &mut sdx, &mut semx);
        avg_sd(syin, npnts, &mut avgy, &mut sdy, &mut semy);
    }

    /* start with pointers in numerical order */
    for i in 0..n {
        inddist[i] = i as i32;
        sy[i] = sdx * syin[i] / sdy;
    }

    /* find distances from center and order by distance */
    cenorder(sx, &sy, npnts, &mut inddist, &mut cendist, xcen, ycen);
    nptuse = npnts;

    /* if omitting farthest points, find distances again */
    if fracomit != 0. {
        // `B3DMAX(npnts - B3DNINT(fabs((double)fracomit) * npnts), B3DMIN(3, npnts))`
        let omit = ((fracomit as f64).abs() * npnts as f64 + 0.5).floor() as i32;
        let floor = if 3 < npnts { 3 } else { npnts };
        nptuse = if npnts - omit > floor {
            npnts - omit
        } else {
            floor
        };
        if fracomit < 0. {
            for i in 0..n {
                sy[i] = syin[i];
            }
        }
        cenorder(sx, &sy, nptuse, &mut inddist, &mut cendist, xcen, ycen);
    }

    /* get pointers to points in order by angle and inverse pointers */
    angorder(
        sx,
        &sy,
        nptuse,
        &inddist,
        *xcen,
        *ycen,
        &mut iangtopt,
        &mut ipttoang,
        &mut angtmp,
    );

    // `#define nextpt(a) iangtopt[(ipttoang[a] + 1) % nptuse]`
    macro_rules! nextpt {
        ($a:expr) => {
            iangtopt[((ipttoang[$a] + 1) % nptuse) as usize] as usize
        };
    }

    /* find the rightmost lowest point; it must be a vertex */
    ymin = 1.0e30;
    for i in 0..nptuse as usize {
        let ind = inddist[i] as usize;
        if sy[ind] < ymin || (sy[ind] == ymin && sx[ind] > xmax) {
            indstart = ind;
            ymin = sy[ind];
            xmax = sx[ind];
        }
        vertex[ind] = 1;
    }

    /* start the scan at INDSTART and next 2 points */
    let mut ind1 = indstart;
    let mut ind2 = nextpt!(ind1);
    let mut ind3 = nextpt!(ind2);
    *nvert = nptuse;
    while ind2 != indstart && *nvert > 2 {
        /* test for left or right turn */
        if sx[ind1] * (sy[ind2] - sy[ind3]) - sx[ind2] * (sy[ind1] - sy[ind3])
            + sx[ind3] * (sy[ind1] - sy[ind2])
            > 0.
        {
            /* left turn; advance the scan */
            ind1 = ind2;
            ind2 = ind3;
            ind3 = nextpt!(ind3);
        } else {
            /* right turn; mark point 2 as non-vertex */
            vertex[ind2] = 0;
            *nvert -= 1;
            if ind1 == indstart {
                /* if still at start, advance points 2 and 3 */
                ind2 = ind3;
                ind3 = nextpt!(ind3);
            } else {
                /* otherwise drop points 1 and 2 back; make point 1 be the last
                point that is still eligible as a vertex */
                ind2 = ind1;
                loop {
                    itry = ipttoang[ind1] - 1;
                    if itry < 0 {
                        itry = nptuse - 1;
                    }
                    ind1 = iangtopt[itry as usize] as usize;
                    if vertex[ind1] != 0 {
                        break;
                    }
                }
            }
        }
    }

    /* put vertices into bx,by in order by angle relative to
     * centroid, with pad if called for */
    *nvert = 0;
    for i in 0..nptuse as usize {
        let ind = iangtopt[i] as usize;
        if vertex[ind] != 0 {
            if *nvert >= maxverts {
                unsafe {
                    libc::printf(
                        c"convexbound: The contour has too many vertices for the arrays\n".as_ptr(),
                    );
                }
                *nvert = -2;
                break;
            }
            padfrac = 0.;
            if pad > 0. && cendist[ind] > 1.0e-10 {
                padfrac = (pad as f64 / (cendist[ind] as f64).sqrt()) as f32;
            }
            bx[*nvert as usize] = sx[ind] + padfrac * (sx[ind] - *xcen);
            by[*nvert as usize] = sy[ind] + padfrac * (sy[ind] - *ycen);
            *nvert += 1;
        }
    }
}

/// Matches Fortran wrapper `convexbound` (`IMOD/libcfshr/convexbound.c:193`).
pub fn convexbound(
    sx: &[f32],
    sy_input: &[f32],
    number_points: usize,
    fraction_omit: f32,
    padding: f32,
    bx: &mut [f32],
    by: &mut [f32],
    number_vertices: &mut i32,
    x_center: &mut f32,
    y_center: &mut f32,
    maximum_vertices: usize,
) {
    convex_bound(
        &sx[..number_points],
        &sy_input[..number_points],
        fraction_omit,
        padding,
        &mut bx[..maximum_vertices],
        &mut by[..maximum_vertices],
        number_vertices,
        x_center,
        y_center,
    )
}
