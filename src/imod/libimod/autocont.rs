//! Translation of `IMOD/libimod/autocont.c` -- routines for autocontouring
//! used by `imodauto` and `3dmod/autox`.

use std::io::Write as _;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, b3d_omp_thread_num, c_format_bytes};
use crate::imod::libcfshr::filtxcorr::scaled_gaussian_kernel;
use crate::imod::libcfshr::islice::{
    Islice, MrcData, slice_byte_smooth, slice_init, slice_mat_filter, slice_scale_and_free,
};
use crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
use crate::imod::libimod::icont::{
    imod_contour_area, imod_contour_copy, imod_contour_default, imod_contour_get_bbox,
    imod_contour_new, imod_contour_reduce, imod_contour_shave, imod_contour_strip,
    imod_contours_new,
};
use crate::imod::libimod::imodel::{Icont, Iobj, Ipoint};
use crate::imod::libimod::iobj::{imod_object_add_contour, imod_object_new};
use crate::imod::libimod::ipoint::{
    imod_point_add, imod_point_append, imod_point_cont_distance, imod_point_inside_cont,
};

/// `AUTOX_BLANK` (`imodel.h:125`).
pub const AUTOX_BLANK: u8 = 0;
/// `AUTOX_FLOOD` (`imodel.h:126`).
pub const AUTOX_FLOOD: u8 = 1;
/// `AUTOX_PATCH` (`imodel.h:127`).
pub const AUTOX_PATCH: u8 = 1 << 1;
/// `AUTOX_FILL` (`imodel.h:128`).
pub const AUTOX_FILL: u8 = AUTOX_FLOOD | AUTOX_PATCH;
/// `AUTOX_CHECK` (`imodel.h:129`).
pub const AUTOX_CHECK: u8 = 1 << 5;
/// `MAX_AUTO_SLICE_THREADS` (`imodel.h:130`).
pub const MAX_AUTO_SLICE_THREADS: i32 = 8;

/// C `sStopProcessing` (`autocont.c:27`).
static S_STOP_PROCESSING: AtomicI32 = AtomicI32::new(0);

/// `KERNEL_MAXSIZE` (`autocont.c:29`).
const KERNEL_MAXSIZE: i32 = 7;

/// C `imodAutoPatch` (`autocont.c:38`).
///
/// Marks the area outside of the "flooded" region of pixels in `data`, ones
/// marked with `AUTOX_FLOOD`, with the patch flag, `AUTOX_PATCH`, then makes
/// all unmarked pixels be part of the flooded region, thus filling in the
/// interior of the flood.  The size of the data array is given in `xsize` and
/// `ysize`; `xlist` and `ylist` are temporary arrays with size given by
/// `listsize`, which should be at least 4 * (`xsize` + `ysize`).
pub fn imod_auto_patch(
    data: &mut [u8],
    xlist: &mut [i32],
    ylist: &mut [i32],
    listsize: i32,
    xsize: i32,
    ysize: i32,
) {
    let mut i: i32;
    let xysize: i32;
    let mut xmax: i32 = -1;
    let mut xmin: i32 = xsize;
    let mut ymax: i32 = -1;
    let mut ymin: i32 = ysize;

    /* get min and max of flooded area */
    for y in 0..ysize {
        for x in 0..xsize {
            if data[(x + y * xsize) as usize] & AUTOX_FLOOD != 0 {
                if x < xmin {
                    xmin = x;
                }
                if x > xmax {
                    xmax = x;
                }
                if y < ymin {
                    ymin = y;
                }
                if y > ymax {
                    ymax = y;
                }
            }
        }
    }

    /* Start a patch from every point along the four sides, because there
    may be isolated patches */
    for x in xmin..=xmax {
        auto_patch_fill_outside(
            data, xlist, ylist, listsize, xsize, xmin, xmax, ymin, ymax, x, ymin,
        );
        auto_patch_fill_outside(
            data, xlist, ylist, listsize, xsize, xmin, xmax, ymin, ymax, x, ymax,
        );
    }
    for y in ymin..=ymax {
        auto_patch_fill_outside(
            data, xlist, ylist, listsize, xsize, xmin, xmax, ymin, ymax, xmin, y,
        );
        auto_patch_fill_outside(
            data, xlist, ylist, listsize, xsize, xmin, xmax, ymin, ymax, xmax, y,
        );
    }

    xysize = xsize * ysize;

    /* Mark everything now not in a patch as in the flood */
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            i = x + y * xsize;
            if data[i as usize] & (AUTOX_FLOOD | AUTOX_PATCH) == 0 {
                data[i as usize] |= AUTOX_FLOOD;
            }
        }
    }

    /* Clear the patch flags */
    for i in 0..xysize {
        if data[i as usize] & AUTOX_PATCH != 0 {
            data[i as usize] &= !AUTOX_PATCH;
        }
    }
}

/// C `autoPatchFillOutside` (`autocont.c:94`).
///
/// To build a patch (add points to existing patch) from a single point.
#[allow(clippy::too_many_arguments)]
fn auto_patch_fill_outside(
    data: &mut [u8],
    xlist: &mut [i32],
    ylist: &mut [i32],
    listsize: i32,
    xsize: i32,
    xmin: i32,
    xmax: i32,
    ymin: i32,
    ymax: i32,
    x: i32,
    y: i32,
) {
    let mut ringnext: i32 = 0;
    let mut ringfree: i32 = 1;
    let mut pixind: i32;
    let neighflag: u8;

    /* Don't even start if this point is a patch or a flood */
    pixind = x + y * xsize;
    if data[pixind as usize] & (AUTOX_FLOOD | AUTOX_PATCH) != 0 {
        return;
    }

    /* initialize the ring buffer */
    xlist[0] = x;
    ylist[0] = y;
    data[pixind as usize] |= AUTOX_CHECK | AUTOX_PATCH;
    neighflag = AUTOX_FLOOD | AUTOX_CHECK | AUTOX_PATCH;

    while ringnext != ringfree {
        /* the next point on list got there by being neither patch nor
        flood, so it needs no checking or marking */
        let x = xlist[ringnext as usize];
        let y = ylist[ringnext as usize];
        pixind = x + y * xsize;

        /* add each of four neighbors on list if coordinate is legal
        and they are not already on list or in flood or patch.
        Mark each as on list and in patch */
        if x > xmin && data[(pixind - 1) as usize] & neighflag == 0 {
            xlist[ringfree as usize] = x - 1;
            ylist[ringfree as usize] = y;
            ringfree += 1;
            ringfree %= listsize;
            data[(pixind - 1) as usize] |= AUTOX_CHECK | AUTOX_PATCH;
        }
        if x < xmax && data[(pixind + 1) as usize] & neighflag == 0 {
            xlist[ringfree as usize] = x + 1;
            ylist[ringfree as usize] = y;
            ringfree += 1;
            ringfree %= listsize;
            data[(pixind + 1) as usize] |= AUTOX_CHECK | AUTOX_PATCH;
        }
        if y > ymin && data[(pixind - xsize) as usize] & neighflag == 0 {
            xlist[ringfree as usize] = x;
            ylist[ringfree as usize] = y - 1;
            ringfree += 1;
            ringfree %= listsize;
            data[(pixind - xsize) as usize] |= AUTOX_CHECK | AUTOX_PATCH;
        }
        if y < ymax && data[(pixind + xsize) as usize] & neighflag == 0 {
            xlist[ringfree as usize] = x;
            ylist[ringfree as usize] = y + 1;
            ringfree += 1;
            ringfree %= listsize;
            data[(pixind + xsize) as usize] |= AUTOX_CHECK | AUTOX_PATCH;
        }

        /* Take point off list, advance next pointer */
        data[pixind as usize] &= !AUTOX_CHECK;
        ringnext += 1;
        ringnext %= listsize;
    }
}

/// C `imodAutoShrink` (`autocont.c:165`).
///
/// Shrinks the area in `data` marked with the flag `AUTOX_FLOOD` by
/// eliminating every point with fewer than 7 neighbors in the flood.  `imax`
/// and `jmax` are the X and Y sizes of `data`.
pub fn imod_auto_shrink(data: &mut [u8], imax: i32, jmax: i32) {
    let mut k: i32;
    let mut x: i32;
    let mut y: i32;

    /* DNM: tried testing on fill flag before checking neighbors and it
    didn't work. */
    for j in 0..jmax {
        for i in 0..imax {
            if data[(i + j * imax) as usize] & AUTOX_FLOOD != 0 {
                k = 0;
                for n in -1..=1 {
                    y = n + j;
                    for m in -1..=1 {
                        x = m + i;
                        if (x >= 0)
                            && (y >= 0)
                            && (x < imax)
                            && (y < jmax)
                            && data[(x + y * imax) as usize] & AUTOX_FLOOD != 0
                        {
                            k += 1;
                        }
                    }
                }
                if k < 7 {
                    data[(i + j * imax) as usize] |= AUTOX_CHECK;
                }
            }
        }
    }

    /* DNM: clear check flag after use, not before */
    for j in 0..jmax {
        for i in 0..imax {
            if data[(i + j * imax) as usize] & AUTOX_CHECK != 0 {
                data[(i + j * imax) as usize] &= !(AUTOX_FLOOD | AUTOX_CHECK);
            }
        }
    }
}

/// C `imodAutoExpand` (`autocont.c:204`).
///
/// Expands the area in `data` marked with the flag `AUTOX_FLOOD` or
/// `AUTOX_PATCH` by adding all 8 neighbors around each marked by, marking it
/// with `AUTOX_FLOOD`.  `imax` and `jmax` are the X and Y sizes of `data`.
pub fn imod_auto_expand(data: &mut [u8], imax: i32, jmax: i32) {
    let mut x: i32;
    let mut y: i32;

    for j in 0..jmax {
        for i in 0..imax {
            if data[(i + j * imax) as usize] & AUTOX_FILL == 0 {
                continue;
            }

            for m in -1..=1 {
                y = j + m;
                if (y < 0) || (y >= jmax) {
                    continue;
                }
                for n in -1..=1 {
                    x = n + i;
                    if (x == i) && (y == j) {
                        continue;
                    }
                    if (x < 0) || (x >= imax) {
                        continue;
                    }
                    data[(x + y * imax) as usize] |= AUTOX_CHECK;
                }
            }
        }
    }

    /* DNM: clear check flag in this loop, not before use */
    for i in 0..imax * jmax {
        if data[i as usize] & AUTOX_CHECK != 0 {
            data[i as usize] |= AUTOX_FLOOD;
            data[i as usize] &= !AUTOX_CHECK;
        }
    }
}

/// C `imodAutoContoursFromSlice` (`autocont.c:288`).
///
/// Generates contours at given thresholds for one slice of byte data and
/// returns them in object `nobj`.  See `autocont.c:238-287` for the full
/// description of each argument.  Returns 1 for memory errors and -1 if
/// `imodAutoContourStop` was called.
///
/// Three deviations from the C parameter list, none of them behavioural:
///
/// * `linePtrs` is not a parameter.  The C caller builds it as
///   `&idata[j * nx]` for each row (`imodauto.c:517-518`), i.e. the rows of
///   the very `idata` this function filters in place, so it is rebuilt here
///   from `idata` after the filter rather than handed in.  The
///   `B3DCHOICE(exact < 0, linePtrs, NULL)` selection at `autocont.c:611` is
///   kept exactly.
/// * `Ilist *boundConts` of `int` is a `Vec<i32>`.
/// * The C's `nobjsize`/`onobjsize` are assigned and never read
///   (`autocont.c:468-532`); they are kept so the elimination loop reads as
///   the source writes it.
#[allow(clippy::too_many_arguments)]
pub fn imod_auto_contours_from_slice(
    ksigma: f32,
    highthresh: f64,
    lowthresh: f64,
    exact: i32,
    dim: i32,
    minsize: i32,
    maxsize: i32,
    followdiag: i32,
    inside: i32,
    shave: f64,
    tol: f64,
    delete_edge: i32,
    smoothflags: i32,
    bound_obj: Option<&Iobj>,
    nearest_bound: i32,
    nobj: &mut Iobj,
    bound_conts: &mut Vec<i32>,
    nx: i32,
    ny: i32,
    tdata: &mut [i32],
    idata: &mut [u8],
    fdata_in: &mut [u8],
    xlist_in: &mut [i32],
    ylist_in: &mut [i32],
    mean_in: f32,
    ksec: i32,
    listsize: i32,
    num_threads: i32,
) -> i32 {
    let mut mean = mean_in;
    let mut num_threads = num_threads;
    let mut nco: i32;
    let mut cz: i32;
    let mut ncont: i32 = 0;
    let mut incont: i32;
    let mut thrd: i32;
    let mut add_error: i32 = 0;
    let mut nump: i32;
    let mut i: i32;
    let mut j: i32;
    let mut ind: i32;
    let mut cont_label: i32;
    let mut tsum: f64;
    let t1: i32;
    let t2: i32;
    let nxy: i32 = nx * ny;
    let mut area: f32;
    let mut thresh_used: f32;
    let mut nobjsize: i32;
    let onobjsize: i32;
    let mut pmin = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let mut pmax = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let mut pim = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let mut nedge: i32;
    let mut critedge: i32;
    let mut diagonal: i32;
    let mut reverse: i32;
    let mut kerdim: i32 = 0;

    S_STOP_PROCESSING.store(0, Ordering::Relaxed);
    nobj.red = 0.;
    nobj.green = 1.;
    nobj.blue = 0.;

    /* `ACCUM_MIN(numThreads, MAX_AUTO_SLICE_THREADS)` (`autocont.c:327`). */
    if MAX_AUTO_SLICE_THREADS < num_threads {
        num_threads = MAX_AUTO_SLICE_THREADS;
    }
    /* `autocont.c:328-332` partitions the caller's `fdata`, `xlist` and
    `ylist` into one block per thread; the offsets are recomputed at each use
    below instead of being held as pointers. */

    /* Get list of boundary contours for this section */
    if let Some(bound_obj) = bound_obj {
        if find_boundary_conts(ksec, bound_obj, nearest_bound, bound_conts) != 0 {
            return 1;
        }
    }

    /* Filter first if sigma entered */
    if ksigma >= 0. {
        let mut slice = Islice {
            data: MrcData::default(),
            xsize: 0,
            ysize: 0,
            mode: 0,
            csize: 0,
            dsize: 0,
            min: 0.,
            max: 0.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        /* The C slice borrows `idata` itself (`autocont.c:340`) and the
        filters write back through it; here the slice owns its bytes and they
        are copied back into `idata` afterwards. */
        slice_init(
            &mut slice,
            nx,
            ny,
            MRC_MODE_BYTE,
            MrcData::B(idata[..nxy as usize].to_vec()),
        );
        slice.min = 0.;
        slice.max = 0.;
        if ksigma > 0. {
            let mut kernel = [0f32; (KERNEL_MAXSIZE * KERNEL_MAXSIZE) as usize];
            scaled_gaussian_kernel(&mut kernel, &mut kerdim, KERNEL_MAXSIZE, ksigma);
            if let Some(mut sout) = slice_mat_filter(&slice, &kernel, kerdim) {
                slice_scale_and_free(&mut sout, &mut slice);
            }
        } else {
            slice_byte_smooth(&mut slice);
        }
        idata[..nxy as usize].copy_from_slice(&slice.data.b()[..nxy as usize]);
    }

    /* The filter is done; `linePtrs` (`imodauto.c:517-518`) is the rows of
    `idata`. */
    let idata: &[u8] = &*idata;
    let mut line_ptrs: Vec<&[u8]> = Vec::with_capacity(ny as usize);
    for jj in 0..ny {
        line_ptrs.push(&idata[(jj * nx) as usize..((jj + 1) * nx) as usize]);
    }

    /* Get per-section mean for dim = 3 */
    if dim == 3 {
        tsum = 0.;
        for i in 0..nxy {
            tsum += idata[i as usize] as f64;
        }
        mean = (tsum / nxy as f64) as f32;
    }

    /* Use the thresholds literally for dim = 1 */
    if dim == 1 {
        t1 = lowthresh as i32;
        t2 = highthresh as i32;
    } else {
        /* `(int) mean * lowthresh`: the cast binds to `mean`, so this is an
        `int` times a `double`, truncated back to `int`. */
        t1 = (mean as i32 as f64 * lowthresh) as i32;
        t2 = (mean as i32 as f64 * highthresh) as i32;
    }

    /* To match imod auto, increment t1 and test for >= that, or <=
    low threshold.  But also set t1 to -1 if it's 0, to enforce an
    exclusion of 0's.  Also, if doing exact, these values are already set up */
    let mut t1 = t1;
    let mut t2 = t2;
    if exact < 0 {
        t2 += 1;
        if t1 == 0 {
            t1 = -1;
        }
    }
    let t1 = t1;
    let t2 = t2;

    cont_label = 1;
    /* init tdata and fill with out of bounds data */
    for i in 0..nxy {
        tdata[i as usize] = 0;
    }
    for i in 0..nxy {
        if (idata[i as usize] as i32) > t1
            && (idata[i as usize] as i32) < t2
            && idata[i as usize] as i32 != exact
        {
            tdata[i as usize] = -1;
        }
    }

    /* If there are boundary contours, check each point not marked out yet and mark
    the ones that are not inside any contour */
    /* `incont` is an uninitialised local in the C (`autocont.c:299`); it is
    only read at `:506` after this loop has written it, except when this loop
    is skipped or never reaches a pixel, where the C reads stack residue. */
    incont = 0;
    if !bound_conts.is_empty() {
        pim.z = ksec as f32;
        for j in 0..ny {
            for i in 0..nx {
                if tdata[(i + j * nx) as usize] != 0 {
                    continue;
                }
                pim.x = (i as f64 + 0.5) as f32;
                pim.y = (j as f64 + 0.5) as f32;
                incont = 0;
                ind = 0;
                while ind < bound_conts.len() as i32 && incont == 0 {
                    nump = bound_conts[ind as usize];
                    incont = imod_point_inside_cont(
                        &bound_obj.expect("boundConts is non-empty").cont[nump as usize],
                        &pim,
                    );
                    ind += 1;
                }
                if incont == 0 {
                    tdata[(i + j * nx) as usize] = -1;
                }
            }
        }
    }

    /* Loop on each pixel that hasn't been marked somehow, and fill a patch from that
    point */
    for j in 0..ny {
        for i in 0..nx {
            if S_STOP_PROCESSING.load(Ordering::Relaxed) != 0 {
                return -1;
            }
            if tdata[(i + j * nx) as usize] != 0 {
                continue;
            }
            if followdiag <= 0 {
                diagonal = 0;
            } else if followdiag >= 3 {
                diagonal = 1;
            } else if followdiag == 1 {
                diagonal = ((exact < 0 && idata[(i + j * nx) as usize] as i32 >= t2)
                    || idata[(i + j * nx) as usize] as i32 == exact)
                    as i32;
            } else {
                diagonal = ((exact < 0 && idata[(i + j * nx) as usize] as i32 <= t1)
                    || idata[(i + j * nx) as usize] as i32 == exact)
                    as i32;
            }

            if imoda_object_bfill_2d(
                idata,
                tdata,
                &mut xlist_in[..listsize as usize],
                &mut ylist_in[..listsize as usize],
                nx,
                ny,
                i,
                j,
                t1,
                t2,
                exact,
                diagonal,
                cont_label,
                listsize,
            ) > 1
                || minsize < 2
            {
                cont_label += 1;
            } else {
                /* If single pixel, and minsize > 1, just eliminate
                this pixel */
                tdata[(i + j * nx) as usize] = -1;
            }
        }
    }

    // Return if nothing found
    if cont_label <= 1 {
        return 0;
    }

    /* sort the points into contours */
    let Some(mut obj) = imod_object_new() else {
        return 1;
    };
    let Some(conts) = imod_contours_new(cont_label - 1) else {
        return 1;
    };
    obj.cont = conts;

    for j in 0..ny {
        for i in 0..nx {
            let co = tdata[(i + j * nx) as usize];
            if co > 0 {
                let pt = Ipoint {
                    x: i as f32,
                    y: j as f32,
                    z: ksec as f32,
                };
                if imod_point_append(&mut obj.cont[(co - 1) as usize], pt) == 0 {
                    return 1;
                }
            }
        }
    }

    nobjsize = nobj.cont.len() as i32;
    onobjsize = nobjsize;
    let _ = onobjsize;
    /* eliminate contours with # of points outside the bounds */
    for co in 0..obj.cont.len() {
        if S_STOP_PROCESSING.load(Ordering::Relaxed) != 0 {
            return -1;
        }

        nedge = 0;
        critedge = delete_edge;
        /* If doing inside, set up to eliminate any contour touching
        an edge if it is the wrong polarity */
        if inside != 0 {
            i = obj.cont[co].pts[0].x as i32;
            j = obj.cont[co].pts[0].y as i32;
            if (followdiag == 1
                && ((exact < 0 && idata[(i + j * nx) as usize] as i32 <= t1)
                    || (exact >= 0 && idata[(i + j * nx) as usize] as i32 != exact)))
                || (followdiag == 2 && idata[(i + j * nx) as usize] as i32 >= t2)
            {
                critedge = 1;
            }
        }
        if critedge != 0 {
            /* count the edges that the contour touches */
            imod_contour_get_bbox(Some(&obj.cont[co]), &mut pmin, &mut pmax);
            if pmin.x == 0. {
                nedge += 1;
            }
            if pmax.x == (nx - 1) as f32 {
                nedge += 1;
            }
            if pmin.y == 0. {
                nedge += 1;
            }
            if pmax.y == (ny - 1) as f32 {
                nedge += 1;
            }

            /* If there are boundary conts, find distance of each point to each contour and
            set edge flag if it is ever close */
            // What in the world is incont doing there?
            if !bound_conts.is_empty() {
                ind = 0;
                while ind < bound_conts.len() as i32 && incont == 0 {
                    nump = bound_conts[ind as usize];
                    i = 0;
                    while i < obj.cont[co].pts.len() as i32 {
                        pim.x = (obj.cont[co].pts[i as usize].x as f64 + 0.5) as f32;
                        pim.y = (obj.cont[co].pts[i as usize].y as f64 + 0.5) as f32;
                        let mut closest: i32 = 0;
                        if imod_point_cont_distance(
                            &bound_obj.expect("boundConts is non-empty").cont[nump as usize],
                            &pim,
                            0,
                            0,
                            &mut closest,
                        ) < 0.8
                        {
                            nedge += 1;
                            break;
                        }
                        i += 1;
                    }
                    if i < obj.cont[co].pts.len() as i32 {
                        break;
                    }
                    ind += 1;
                }
            }
        }
        if (obj.cont[co].pts.len() as i32) < minsize
            || (obj.cont[co].pts.len() as i32 > maxsize && maxsize > 0)
            || (critedge != 0 && nedge >= critedge)
        {
            if !obj.cont[co].pts.is_empty() {
                obj.cont[co].pts.clear();
            }
            continue;
        }

        nobjsize += 1;
    }
    let _ = nobjsize;

    add_error = 0;

    let mut thrd_obj: Vec<Iobj> = Vec::new();
    for _ in 0..num_threads {
        let Some(one) = imod_object_new() else {
            return 1;
        };
        thrd_obj.push(one);
    }

    for co in 0..obj.cont.len() {
        thrd = b3d_omp_thread_num();

        if obj.cont[co].pts.is_empty()
            || add_error != 0
            || S_STOP_PROCESSING.load(Ordering::Relaxed) != 0
        {
            continue;
        }

        cz = obj.cont[co].pts[0].z as i32;

        let fdata_base = (thrd * nx * ny) as usize;
        let xylist_base = (thrd * listsize) as usize;

        /* Clear fdata array and mark pixels in this contour as FLOOD */
        for i in 0..nxy {
            fdata_in[fdata_base + i as usize] = 0;
        }

        i = 0;
        j = 0;
        for cpt in 0..obj.cont[co].pts.len() {
            i = obj.cont[co].pts[cpt].x as i32;
            j = obj.cont[co].pts[cpt].y as i32;
            fdata_in[fdata_base + (i + j * nx) as usize] = AUTOX_FLOOD;
        }

        /* `i` and `j` are the last point of the contour here, which is what
        the C's own loop leaves behind (`autocont.c:567-582`). */
        if followdiag <= 0 {
            diagonal = 0;
        } else if followdiag >= 3 {
            diagonal = 1;
        } else if followdiag == 1 {
            diagonal = ((exact < 0 && idata[(i + j * nx) as usize] as i32 >= t2)
                || idata[(i + j * nx) as usize] as i32 == exact) as i32;
        } else {
            diagonal = ((exact < 0 && idata[(i + j * nx) as usize] as i32 <= t1)
                || idata[(i + j * nx) as usize] as i32 == exact) as i32;
        }

        imod_auto_patch(
            &mut fdata_in[fdata_base..fdata_base + nxy as usize],
            &mut xlist_in[xylist_base..xylist_base + listsize as usize],
            &mut ylist_in[xylist_base..xylist_base + listsize as usize],
            listsize,
            nx,
            ny,
        );

        /* Set the reverse flag and set threshold based on which one is passed */
        /* These won't be used for exact work */
        reverse = if idata[(i + j * nx) as usize] as i32 <= t1 {
            1
        } else {
            0
        };
        thresh_used = -1.;
        if idata[(i + j * nx) as usize] as i32 <= t1 {
            thresh_used = (t1 as f64 + 0.5) as f32;
        }
        if idata[(i + j * nx) as usize] as i32 >= t2 {
            thresh_used = (t2 as f64 - 0.5) as f32;
        }

        /* If we do an expand, shrink, or smooth, run the patch again and set
        the threshold to be found by the routine.
        Probably should forbid this for exact */
        j = if smoothflags > 3 { smoothflags >> 2 } else { 1 };
        for _i in 0..j {
            if smoothflags & 2 != 0 {
                imod_auto_expand(&mut fdata_in[fdata_base..fdata_base + nxy as usize], nx, ny);
            }
            if smoothflags % 2 != 0 {
                imod_auto_shrink(&mut fdata_in[fdata_base..fdata_base + nxy as usize], nx, ny);
            }
        }
        if smoothflags & 3 != 0 {
            imod_auto_patch(
                &mut fdata_in[fdata_base..fdata_base + nxy as usize],
                &mut xlist_in[xylist_base..xylist_base + listsize as usize],
                &mut ylist_in[xylist_base..xylist_base + listsize as usize],
                listsize,
                nx,
                ny,
            );
            thresh_used = -1.;
        }

        let mut newconts = imod_contours_from_image_points(
            &mut fdata_in[fdata_base..fdata_base + nxy as usize],
            if exact < 0 { Some(&line_ptrs) } else { None },
            nx,
            ny,
            cz,
            AUTOX_FLOOD,
            diagonal,
            thresh_used,
            reverse,
            &mut ncont,
        );
        for i in 0..ncont {
            /* Just check the area and eliminate again */
            area = imod_contour_area(Some(&newconts[i as usize]));
            if area < minsize as f32 || (maxsize > 0 && area > maxsize as f32) {
                continue;
            }

            imod_contour_strip(&mut newconts[i as usize]);
            if tol != 0.0 {
                imod_contour_reduce(Some(&mut newconts[i as usize]), tol as f32);
            }
            if shave != 0.0 {
                imod_contour_shave(&mut newconts[i as usize], shave);
            }
            let Some(tmpcont) = imod_contour_new() else {
                add_error = 1;
                break;
            };
            if imod_object_add_contour(&mut thrd_obj[thrd as usize], tmpcont) < 0 {
                add_error = 1;
                break;
            }
            let last = thrd_obj[thrd as usize].cont.len() - 1;
            let source = newconts[i as usize].clone();
            imod_contour_copy(&source, &mut thrd_obj[thrd as usize].cont[last]);
        }
    }

    drop(obj);
    if S_STOP_PROCESSING.load(Ordering::Relaxed) != 0 {
        return -1;
    }

    if add_error == 0 {
        // Count up the new contours and alllocate/reallocate contour array in one shot
        nco = nobj.cont.len() as i32;
        for thrd in 0..num_threads {
            nco += thrd_obj[thrd as usize].cont.len() as i32;
        }
        nobj.cont
            .reserve((nco as usize).saturating_sub(nobj.cont.len()));

        // Assign contour array, copy over the contours one by one
        for thrd in 0..num_threads {
            for co in 0..thrd_obj[thrd as usize].cont.len() {
                let source = thrd_obj[thrd as usize].cont[co].clone();
                nobj.cont.push(source);
            }
        }
    }

    add_error
}

/// C `imodAutoContourStop` (`autocont.c:684`).
///
/// Makes `imod_auto_contours_from_slice` stop processing and return -1.
pub fn imod_auto_contour_stop() {
    S_STOP_PROCESSING.store(1, Ordering::Relaxed);
}

/// `RIGHT_EDGE` (`autocont.c:689`).
const RIGHT_EDGE: u8 = 16;
/// `TOP_EDGE` (`autocont.c:690`).
const TOP_EDGE: u8 = RIGHT_EDGE << 1;
/// `LEFT_EDGE` (`autocont.c:691`).
const LEFT_EDGE: u8 = RIGHT_EDGE << 2;
/// `BOTTOM_EDGE` (`autocont.c:692`).
const BOTTOM_EDGE: u8 = RIGHT_EDGE << 3;
/// `ANY_EDGE` (`autocont.c:693`).
const ANY_EDGE: u8 = RIGHT_EDGE | TOP_EDGE | LEFT_EDGE | BOTTOM_EDGE;

/// C `imodContoursFromImagePoints` (`autocont.c:723`).
///
/// Forms contours around marked points in an array.  `data` is an array of
/// flags marking the image points.  `imdata` is the corresponding actual image
/// array as one slice per row, which is used to compute interpolated positions
/// for the edges between marked and unmarked points; if it is `None`, no
/// interpolation is done and contours will follow horizontal, vertical, and 45
/// degree diagonal lines.  The number of contours created is returned in
/// `ncont`.
#[allow(clippy::too_many_arguments)]
pub fn imod_contours_from_image_points(
    data: &mut [u8],
    imdata: Option<&Vec<&[u8]>>,
    xsize: i32,
    ysize: i32,
    z: i32,
    testmask: u8,
    diagonal: i32,
    threshold_in: f32,
    reverse: i32,
    ncont: &mut i32,
) -> Vec<Icont> {
    let mut threshold = threshold_in;
    let mut point = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let mut contarr: Vec<Icont> = Vec::new();
    let mut itst: i32;
    let mut jtst: i32;
    let mut side: u8;
    let xs: i32;
    let ys: i32;
    let edgemask: [u8; 4] = [RIGHT_EDGE, TOP_EDGE, LEFT_EDGE, BOTTOM_EDGE];
    let nextx: [i32; 4] = [0, -1, 0, 1];
    let nexty: [i32; 4] = [1, 0, -1, 0];
    let cornerx: [i32; 4] = [1, -1, -1, 1];
    let cornery: [i32; 4] = [1, 1, -1, -1];
    let otherx: [i32; 4] = [1, 0, -1, 0];
    let othery: [i32; 4] = [0, 1, 0, -1];
    let mut frac: f32;
    let mut found: i32;
    let mut iedge: usize;
    let mut ixst: i32;
    let mut iyst: i32;
    let mut iedgest: usize;
    let mut nsum: i32;
    let mut nayx: i32;
    let mut nayy: i32;
    let polarity: i32;
    let mut diff: i32 = 0;
    let mut edge_sum: f64;

    xs = xsize - 1;
    ys = ysize - 1;
    point.z = z as f32;
    *ncont = 0;
    nsum = 0;
    edge_sum = 0.;
    polarity = if reverse != 0 { -1 } else { 1 };
    if imdata.is_none() {
        threshold = -1.;
    }

    /* Go through all points including the edges of the image area, and
    mark all the edges of defined area.  Compute a  */

    for j in 0..ysize {
        for i in 0..xsize {
            if data[(i + j * xsize) as usize] & testmask != 0 {
                /* Mark a side if on edge of image, or if next pixel over
                is not in the set */
                side = 0;
                if i == xs {
                    side |= RIGHT_EDGE;
                } else if data[((i + 1) + j * xsize) as usize] & testmask == 0 {
                    side |= RIGHT_EDGE;
                    if threshold < 0. {
                        if let Some(imdata) = imdata {
                            edge_sum += (imdata[j as usize][i as usize] as i32
                                + imdata[j as usize][(i + 1) as usize] as i32)
                                as f64;
                            nsum += 1;
                        }
                    }
                }
                if j == ys {
                    side |= TOP_EDGE;
                } else if data[(i + (j + 1) * xsize) as usize] & testmask == 0 {
                    side |= TOP_EDGE;
                    if threshold < 0. {
                        if let Some(imdata) = imdata {
                            edge_sum += (imdata[j as usize][i as usize] as i32
                                + imdata[(j + 1) as usize][i as usize] as i32)
                                as f64;
                            nsum += 1;
                        }
                    }
                }
                if i == 0 {
                    side |= LEFT_EDGE;
                } else if data[((i - 1) + j * xsize) as usize] & testmask == 0 {
                    side |= LEFT_EDGE;
                    if threshold < 0. {
                        if let Some(imdata) = imdata {
                            edge_sum += (imdata[j as usize][i as usize] as i32
                                + imdata[j as usize][(i - 1) as usize] as i32)
                                as f64;
                            nsum += 1;
                        }
                    }
                }
                if j == 0 {
                    side |= BOTTOM_EDGE;
                } else if data[(i + (j - 1) * xsize) as usize] & testmask == 0 {
                    side |= BOTTOM_EDGE;
                    if threshold < 0. {
                        if let Some(imdata) = imdata {
                            edge_sum += (imdata[j as usize][i as usize] as i32
                                + imdata[(j - 1) as usize][i as usize] as i32)
                                as f64;
                            nsum += 1;
                        }
                    }
                }
                data[(i + j * xsize) as usize] |= side;
            }
        }
    }

    if nsum != 0 {
        threshold = (0.5 * edge_sum / nsum as f64) as f32;
    }

    found = 1;
    while found != 0 {
        found = 0;
        'search: for jsearch in 0..ysize {
            for isearch in 0..xsize {
                if data[(isearch + jsearch * xsize) as usize] & ANY_EDGE == 0 {
                    continue;
                }
                let mut i = isearch;
                let mut j = jsearch;

                /* find lowest edge */
                iedge = 0;
                while iedge < 4 {
                    if data[(i + j * xsize) as usize] & edgemask[iedge] != 0 {
                        break;
                    }
                    iedge += 1;
                }

                /* Start a new contour */
                *ncont += 1;
                contarr.push(Icont::default());
                let contidx = (*ncont - 1) as usize;
                imod_contour_default(&mut contarr[contidx]);

                /* keep track of starting place and stop when reach
                it again */
                iedgest = iedge;
                ixst = i;
                iyst = j;
                while contarr[contidx].pts.is_empty() || i != ixst || j != iyst || iedge != iedgest
                {
                    if diagonal == 0 {
                        /* If no diagonals, look for next edge first
                        around corner on same pixel */
                        if data[(i + j * xsize) as usize] & edgemask[(iedge + 1) % 4] != 0 {
                            iedge = (iedge + 1) % 4;
                        } else if data[((i + nextx[iedge]) + (j + nexty[iedge]) * xsize) as usize]
                            & edgemask[iedge]
                            != 0
                        {
                            /* same edge, next pixel */
                            i += nextx[iedge];
                            j += nexty[iedge];
                        } else {
                            /* pixel on an inside corner - it's got to
                            be, but put in check for testing */
                            i += cornerx[iedge];
                            j += cornery[iedge];
                            iedge = (iedge + 3) % 4;
                            if data[(i + j * xsize) as usize] & edgemask[iedge] == 0 {
                                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                    "no edge around corner at i %d, j %d, edge %d\n",
                                    &[
                                        CArg::Int(i as i64),
                                        CArg::Int(j as i64),
                                        CArg::Int(iedge as i64),
                                    ],
                                ));
                            }
                        }
                    } else {
                        /* If diagonals, look for next edge first on pixel
                        around inside corner if it's legal */
                        itst = i + cornerx[iedge];
                        jtst = j + cornery[iedge];
                        if itst >= 0
                            && itst < xsize
                            && jtst >= 0
                            && jtst < ysize
                            && (data[(itst + jtst * xsize) as usize] & edgemask[(iedge + 3) % 4]
                                != 0)
                        {
                            i = itst;
                            j = jtst;
                            iedge = (iedge + 3) % 4;
                        } else {
                            itst = i + nextx[iedge];
                            jtst = j + nexty[iedge];
                            if itst >= 0
                                && itst < xsize
                                && jtst >= 0
                                && jtst < ysize
                                && data[((i + nextx[iedge]) + (j + nexty[iedge]) * xsize) as usize]
                                    & edgemask[iedge]
                                    != 0
                            {
                                /* then same edge, next pixel */
                                i += nextx[iedge];
                                j += nexty[iedge];
                            } else {
                                /* go around corner on this pixel - the
                                edge has to be there, but put in check
                                for testing */
                                iedge = (iedge + 1) % 4;
                                if data[(i + j * xsize) as usize] & edgemask[iedge] == 0 {
                                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                        "no edge around corner at i %d, j %d, edge %d\n",
                                        &[
                                            CArg::Int(i as i64),
                                            CArg::Int(j as i64),
                                            CArg::Int(iedge as i64),
                                        ],
                                    ));
                                }
                            }
                        }
                    }

                    frac = 0.45;
                    nayx = i + otherx[iedge];
                    nayy = j + othery[iedge];
                    if threshold > 0. && nayx >= 0 && nayx < xsize && nayy >= 0 && nayy < ysize {
                        let imdata = imdata.expect("threshold > 0 implies imdata");
                        diff = imdata[nayy as usize][nayx as usize] as i32
                            - imdata[j as usize][i as usize] as i32;
                        if polarity * diff < 0 {
                            frac = (threshold - imdata[j as usize][i as usize] as i32 as f32)
                                / diff as f32;
                            /* `B3DMIN`/`B3DMAX` against a double literal: the
                            comparison and the result are in double. */
                            frac = (if 0.99 < frac as f64 {
                                0.99
                            } else {
                                frac as f64
                            }) as f32;
                            frac = (if 0.01 > frac as f64 {
                                0.01
                            } else {
                                frac as f64
                            }) as f32;
                        }
                    }
                    point.x = ((i as f64 + 0.5) + (frac * otherx[iedge] as f32) as f64) as f32;
                    point.y = ((j as f64 + 0.5) + (frac * othery[iedge] as f32) as f64) as f32;

                    /* add the point and clear the edge */
                    let psize = contarr[contidx].pts.len() as i32;
                    imod_point_add(&mut contarr[contidx], Some(point), psize);
                    data[(i + j * xsize) as usize] &= !edgemask[iedge];
                }
                found = 1;
                break 'search;
            }
        }
    }
    contarr
}

/// C `imoda_object_bfill_2d` (`autocont.c:936`).
///
/// Marks one patch in `data` with the value `cont_label` starting at point
/// `x`, `y`.  It adds contiguous points below or above threshold depending on
/// whether the value in `idata` is below or above threshold.
#[allow(clippy::too_many_arguments)]
fn imoda_object_bfill_2d(
    idata: &[u8],
    data: &mut [i32],
    xlist: &mut [i32],
    ylist: &mut [i32],
    xsize: i32,
    ysize: i32,
    x: i32,
    y: i32,
    t1: i32,
    t2: i32,
    exact: i32,
    diagonal: i32,
    cont_label: i32,
    listsize: i32,
) -> i32 {
    /* `threshold` and `direction` are uninitialised in the C for the exact
    branch, where neither is read (`autocont.c:941-957`). */
    let threshold: i32;
    let mut ringnext: i32 = 0;
    let mut ringfree: i32 = 1;
    let mut pixind: i32;
    let direction: i32;
    let mut nadded: i32 = 0;
    let mut test_exact: i32 = 0;

    if exact >= 0 {
        test_exact = if idata[(x + y * xsize) as usize] as i32 == exact {
            1
        } else {
            0
        };
        threshold = 0;
        direction = 0;
    } else if idata[(x + y * xsize) as usize] as i32 <= t1 {
        threshold = t1;
        direction = -1;
    } else {
        threshold = t2;
        direction = 1;
    }

    /* initialize the ring buffer */
    xlist[0] = x;
    ylist[0] = y;
    data[(x + y * xsize) as usize] = -2;

    while ringnext != ringfree {
        /* check next point on list */
        let x = xlist[ringnext as usize];
        let y = ylist[ringnext as usize];
        pixind = x + y * xsize;
        if (exact < 0 && direction * (idata[pixind as usize] as i32 - threshold) >= 0)
            || (exact >= 0
                && ((test_exact != 0 && idata[pixind as usize] as i32 == exact)
                    || (test_exact == 0
                        && (idata[pixind as usize] as i32 <= t1
                            || idata[pixind as usize] as i32 >= t2))))
        {
            /* If point passes test, mark as flood */
            data[pixind as usize] = cont_label;
            nadded += 1;

            /* add each of four neighbors on list if coordinate is legal
            and they are not already on list or in flood */
            if x > 0 && data[(pixind - 1) as usize] == 0 {
                xlist[ringfree as usize] = x - 1;
                ylist[ringfree as usize] = y;
                ringfree += 1;
                ringfree %= listsize;
                data[(pixind - 1) as usize] = -2;
            }
            if x < xsize - 1 && data[(pixind + 1) as usize] == 0 {
                xlist[ringfree as usize] = x + 1;
                ylist[ringfree as usize] = y;
                ringfree += 1;
                ringfree %= listsize;
                data[(pixind + 1) as usize] = -2;
            }
            if y > 0 && data[(pixind - xsize) as usize] == 0 {
                xlist[ringfree as usize] = x;
                ylist[ringfree as usize] = y - 1;
                ringfree += 1;
                ringfree %= listsize;
                data[(pixind - xsize) as usize] = -2;
            }
            if y < ysize - 1 && data[(pixind + xsize) as usize] == 0 {
                xlist[ringfree as usize] = x;
                ylist[ringfree as usize] = y + 1;
                ringfree += 1;
                ringfree %= listsize;
                data[(pixind + xsize) as usize] = -2;
            }
            if diagonal != 0 {
                if x > 0 && y > 0 && data[(pixind - 1 - xsize) as usize] == 0 {
                    xlist[ringfree as usize] = x - 1;
                    ylist[ringfree as usize] = y - 1;
                    ringfree += 1;
                    ringfree %= listsize;
                    data[(pixind - 1 - xsize) as usize] = -2;
                }
                if x < xsize - 1 && y > 0 && data[(pixind + 1 - xsize) as usize] == 0 {
                    xlist[ringfree as usize] = x + 1;
                    ylist[ringfree as usize] = y - 1;
                    ringfree += 1;
                    ringfree %= listsize;
                    data[(pixind + 1 - xsize) as usize] = -2;
                }
                if x > 0 && y < ysize - 1 && data[(pixind - 1 + xsize) as usize] == 0 {
                    xlist[ringfree as usize] = x - 1;
                    ylist[ringfree as usize] = y + 1;
                    ringfree += 1;
                    ringfree %= listsize;
                    data[(pixind - 1 + xsize) as usize] = -2;
                }
                if x < xsize - 1 && y < ysize - 1 && data[(pixind + 1 + xsize) as usize] == 0 {
                    xlist[ringfree as usize] = x + 1;
                    ylist[ringfree as usize] = y + 1;
                    ringfree += 1;
                    ringfree %= listsize;
                    data[(pixind + 1 + xsize) as usize] = -2;
                }
            }
        }

        /* Take point off list, advance next pointer */
        if data[pixind as usize] == -2 {
            data[pixind as usize] = 0;
        }
        ringnext += 1;
        ringnext %= listsize;
    }

    nadded
}

/// C `findBoundaryConts` (`autocont.c:1046`).
///
/// Find the boundary contours for a given Z value.  Returns the contour
/// numbers in the list of ints.  Returns only contours at the given Z value if
/// `nearest_bound` is 0, otherwise returns contours at the nearest Z value.
fn find_boundary_conts(
    z: i32,
    bound_obj: &Iobj,
    nearest_bound: i32,
    cont_list: &mut Vec<i32>,
) -> i32 {
    let mut min_diff: i32;
    let mut diff: i32;
    /* `zmin` is uninitialised in the C and is only read after the first loop
    has set it, unless every contour has fewer than 3 points. */
    let mut zmin: i32 = 0;
    let mut zcont: i32;
    cont_list.clear();
    min_diff = 100000000;
    for co in 0..bound_obj.cont.len() {
        if bound_obj.cont[co].pts.len() < 3 {
            continue;
        }
        zcont = (bound_obj.cont[co].pts[0].z as f64 + 0.5).floor() as i32;
        diff = zcont - z;
        if diff.abs() < min_diff {
            min_diff = diff.abs();
            zmin = zcont;
        }
    }
    if nearest_bound == 0 && min_diff > 0 {
        return 0;
    }
    for co in 0..bound_obj.cont.len() {
        if bound_obj.cont[co].pts.len() < 3 {
            continue;
        }
        zcont = (bound_obj.cont[co].pts[0].z as f64 + 0.5).floor() as i32;
        if zmin == zcont {
            cont_list.push(co as i32);
        }
    }
    0
}
