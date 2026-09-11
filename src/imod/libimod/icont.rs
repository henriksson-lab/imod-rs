//! Translation of `IMOD/libimod/icont.c` together with its paired header
//! `IMOD/include/icont.h` -- "Library of contour handling routines."
//!
//! Coverage note: every function of `icont.c` is translated here except the
//! removed/uncompiled `imodContourTracer`.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imat::{
    B3D_X, B3D_Z, imod_mat_delete, imod_mat_new, imod_mat_rot, imod_mat_transform,
    imod_mat_transform2d,
};
use crate::imod::libimod::imodel::{ICONT_OPEN, ICONT_WILD, Icont, Iobj, Ipoint};
use crate::imod::libimod::ipoint::{
    imod_point_add, imod_point_append, imod_point_cross, imod_point_delete, imod_point_distance,
    imod_point_inside_cont, imod_point_set_size, imod_point2d_angle, imod_point3d_scale_distance,
    imodel_point_dist,
};
use crate::imod::libimod::istore::{
    Istore, StoreUnion, istore_add_one_index_item, istore_break_contour, istore_clean_ends,
    istore_copy_non_index, istore_delete_point, istore_extract_changes, istore_invert,
    istore_point_is_gap, istore_retain_point,
};
use std::borrow::Cow;

/// Original: `ICONT_STIPPLED` (`icont.h:23`).
pub const ICONT_STIPPLED: u32 = 1 << 5;
/// Original: `ICONT_CURSOR_LIKE` (`icont.h:24`).
pub const ICONT_CURSOR_LIKE: u32 = 1 << 6;
/// Original: `ICONT_DRAW_ALLZ` (`icont.h:25`).
pub const ICONT_DRAW_ALLZ: u32 = 1 << 7;
/// Original: `ICONT_MMODEL_ONLY` (`icont.h:26`).
pub const ICONT_MMODEL_ONLY: u32 = 1 << 8;
/// Original: `ICONT_NOCONNECT` (`icont.h:27`).
pub const ICONT_NOCONNECT: u32 = 1 << 9;
/// Original: `ICONT_SCANLINE` (`icont.h:28`).
pub const ICONT_SCANLINE: u32 = 1 << 17;
/// Original: `ICONT_CONNECT_TOP` (`icont.h:29`).
pub const ICONT_CONNECT_TOP: u32 = 1 << 18;
/// Original: `ICONT_CONNECT_BOTTOM` (`icont.h:30`).
pub const ICONT_CONNECT_BOTTOM: u32 = 1 << 19;
/// Original: `ICONT_CONNECT_INVERT` (`icont.h:31`).
pub const ICONT_CONNECT_INVERT: u32 = 1 << 20;
/// Original: `ICONT_TEMPUSE2` (`icont.h:32`).
pub const ICONT_TEMPUSE2: u32 = 1 << 30;
/// Original: `ICONT_TEMPUSE` (`icont.h:33`).
pub const ICONT_TEMPUSE: u32 = 1 << 31;
/// Original: `ICONT_TEMPLEVEL_MASK` (`icont.h:34`).
pub const ICONT_TEMPLEVEL_MASK: u32 = 0xF;
/// Original: `ICONT_TEMPLEVEL_SHIFT` (`icont.h:35`).
pub const ICONT_TEMPLEVEL_SHIFT: u32 = 26;

/// Original: `IMOD_CONTOUR_CLOCKWISE` (`icont.h:38`).
pub const IMOD_CONTOUR_CLOCKWISE: i32 = -1;
/// Original: `IMOD_CONTOUR_COUNTER_CLOCKWISE` (`icont.h:39`).
pub const IMOD_CONTOUR_COUNTER_CLOCKWISE: i32 = 1;

/// Original: `ICONT_FIND_NOSORT` (`icont.h:41`).
pub const ICONT_FIND_NOSORT: i32 = 0;
/// Original: `ICONT_FIND_SORTX` (`icont.h:42`).
pub const ICONT_FIND_SORTX: i32 = 1;
/// Original: `ICONT_FIND_SORTY` (`icont.h:43`).
pub const ICONT_FIND_SORTY: i32 = 2;
/// Original: `ICONT_FIND_SORTXY` (`icont.h:44`).
pub const ICONT_FIND_SORTXY: i32 = 3;

/// Original: `Nesting` / `struct Nest_struct` (`icont.h:49`).
///
/// The source's `int *inside` / `int *outside` pointer-plus-count pairs become
/// `Vec`s here; `ninside` and `noutside` are kept as they are read back by
/// `imodContourNestLevels` and by the callers of `imodContourCheckNesting`.
#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct Nesting {
    /// contour number
    pub co: i32,
    /// Level in from outside-most
    pub level: i32,
    /// Number inside
    pub ninside: i32,
    /// Numbers of contours inside
    pub inside: Vec<i32>,
    /// Number outside
    pub noutside: i32,
    /// Numbers of contours outside
    pub outside: Vec<i32>,
    /// Scan contour of object interior, for odd levels
    pub inscan: Option<Icont>,
}

/// Original: `imodContourBad` (`icont.h:45`, macro).
pub fn imod_contour_bad(c: Option<&Icont>, p: i32) -> i32 {
    match c {
        Some(c) => {
            if (c.pts.len() as i32) < p {
                1
            } else {
                0
            }
        }
        None => 1,
    }
}

/// Original: `imodContourIsOpen` (`icont.h:46`, macro).
pub fn imod_contour_is_open(c: &Icont) -> u32 {
    c.flags & ICONT_OPEN
}

/// Original: `imodContourNew` (`icont.c:26`).
///
/// Creates one new contour and initializes it to default values.
pub fn imod_contour_new() -> Option<Icont> {
    let mut cont = Icont::default();
    imod_contour_default(&mut cont);
    Some(cont)
}

/// Original: `imodContoursNew` (`icont.c:43`).
///
/// Creates an array of `size` contours initialized to default values.
pub fn imod_contours_new(size: i32) -> Option<Vec<Icont>> {
    if size <= 0 {
        return None;
    }
    let mut cont = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let mut one = Icont::default();
        imod_contour_default(&mut one);
        cont.push(one);
    }
    Some(cont)
}

/// Original: `imodContourDefault` (`icont.c:64`).
///
/// Initializes `cont` to default values for an empty contour.
pub fn imod_contour_default(cont: &mut Icont) {
    cont.pts.clear();
    cont.flags = 0;
    cont.time = 0;
    cont.surf = 0;
    cont.sizes.clear();
    cont.store.clear();
    cont.temp_val = 0.;
}

/// Original: `imodContourCopy` (`icont.c:82`).
///
/// Copies the contour structure in `from` to `to`.
pub fn imod_contour_copy(from: &Icont, to: &mut Icont) -> i32 {
    *to = from.clone();
    0
}

/// Original: `imodContourDup` (`icont.c:97`).
///
/// Creates a new contour containing the same point, size, and label data as
/// `cont`.
pub fn imod_contour_dup(cont: &Icont) -> Option<Icont> {
    Some(cont.clone())
}

/// Original: `imodContourDelete` (`icont.c:133`).
pub fn imod_contour_delete(cont: &mut Icont) -> i32 {
    imod_contour_clear(cont);
    0
}

/// Original: `imodContourClear` (`icont.c:148`).
///
/// Frees all data in contour `cont`.
pub fn imod_contour_clear(cont: &mut Icont) -> i32 {
    imod_contour_clear_points(cont);
    cont.sizes.clear();
    cont.flags = 0;
    cont.time = 0;
    /* `icont.c:157-159`: delete the label and set it to NULL. */
    crate::imod::libimod::ilabel::imod_label_delete(cont.label.take());
    cont.store.clear();
    0
}

/// Original: `imodContourClearPoints` (`icont.c:169`).
pub fn imod_contour_clear_points(cont: &mut Icont) -> i32 {
    cont.pts.clear();
    0
}

/// Original: `imodContoursDelete` (`icont.c:185`).
pub fn imod_contours_delete(cont: &mut Vec<Icont>, size: i32) -> i32 {
    for co in 0..size as usize {
        imod_contour_clear(&mut cont[co]);
    }
    cont.clear();
    0
}

/// Original: `imodContoursDeleteToEnd` (`icont.c:203`).
///
/// Deletes contours from object `obj`, retaining `keep`.
pub fn imod_contours_delete_to_end(obj: &mut Iobj, keep: i32) -> i32 {
    if (obj.cont.len() as i32) < keep {
        return -1;
    }
    if obj.cont.len() as i32 == keep {
        return 0;
    }
    for co in keep as usize..obj.cont.len() {
        imod_contour_clear(&mut obj.cont[co]);
    }
    obj.cont.truncate(keep as usize);
    0
}

/// Original: `imodel_contour_newsurf` (`icont.c:238`).
///
/// Assigns a new surface number to the contour at index `co` in `obj` and
/// adjusts the maximum surface number for `obj`.  The source takes the contour
/// by pointer; it is addressed by index here because it lives inside `obj`.
pub fn imodel_contour_newsurf(obj: &mut Iobj, co: usize) -> i32 {
    if obj.cont.is_empty() {
        return -1;
    }
    obj.cont[co].surf = imodel_unused_surface(Some(obj));
    if obj.surfsize < obj.cont[co].surf {
        obj.surfsize = obj.cont[co].surf;
    }
    0
}

/// Original: `imodel_unused_surface` (`icont.c:256`).
///
/// Finds the first unused surface number in object `obj`.
pub fn imodel_unused_surface(obj: Option<&Iobj>) -> i32 {
    let Some(obj) = obj else {
        return 0;
    };
    if obj.cont.is_empty() {
        return 0;
    }

    /* find maximum surface number */
    let mut max = obj.cont[0].surf;
    for co in 1..obj.cont.len() {
        if max < obj.cont[co].surf {
            max = obj.cont[co].surf;
        }
    }

    /* Count number of contours at each surface */
    let mut bins: Vec<i32> = vec![0; (max + 1) as usize];
    for co in 0..obj.cont.len() {
        if obj.cont[co].surf >= 0 {
            bins[obj.cont[co].surf as usize] += 1;
        }
    }

    /* find first empty bin */
    let mut co = 0i32;
    while co <= max {
        if bins[co as usize] == 0 {
            break;
        }
        co += 1;
    }
    co
}

/// Original: `imodel_contour_check_wild` (`icont.c:297`).
///
/// Sets the wild flag in contour `cont` if Z is not the same for all points
/// when rounded to the nearest integer.
pub fn imodel_contour_check_wild(cont: Option<&mut Icont>) {
    let Some(cont) = cont else {
        return;
    };
    let mut cz = 0i32;
    if !cont.pts.is_empty() {
        cz = (cont.pts[0].z as f64 + 0.5).floor() as i32;
    }
    cont.flags &= !ICONT_WILD;
    for pt in 1..cont.pts.len() {
        if (cont.pts[pt].z as f64 + 0.5).floor() as i32 != cz {
            cont.flags |= ICONT_WILD;
            break;
        }
    }
}

/****************************************************************************/
/* INFORMATION FUNCTIONS                                                    */

/// Original: `imodContourArea` (`icont.c:324`).
///
/// Returns area of contour `cont` in square pixels.  The contour need not be
/// coplanar.
pub fn imod_contour_area(cont: Option<&Icont>) -> f32 {
    let Some(cont) = cont else {
        return 0.;
    };
    if cont.pts.len() < 3 {
        return 0.;
    }

    let mut n = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };

    for i in 0..cont.pts.len() {
        let next = if i == cont.pts.len() - 1 { 0 } else { i + 1 };
        n.x += (cont.pts[i].y * cont.pts[next].z) - (cont.pts[i].z * cont.pts[next].y);
        n.y += (cont.pts[i].z * cont.pts[next].x) - (cont.pts[i].x * cont.pts[next].z);
        n.z += (cont.pts[i].x * cont.pts[next].y) - (cont.pts[i].y * cont.pts[next].x);
    }
    ((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() as f32 * 0.5f32
}

/// Original: `imodel_contour_area` (`icont.c:357`).
///
/// Returns area of contour `icont` in the X/Y plane in square pixels, measured
/// by converting to a scan contour and summing the length of the scan lines.
pub fn imodel_contour_area(icont: Option<&Icont>) -> i32 {
    let Some(icont) = icont else {
        return 0;
    };
    if icont.pts.len() < 3 {
        return 0;
    }

    let Some(cont) = imodel_contour_scan(Some(icont)) else {
        return 0;
    };

    let mut pix: i32 = 0;
    let mut i: usize = 0;
    let mut scanline = 0;
    while i + 1 < cont.pts.len() {
        let bgnpt = i;
        while cont.pts[i].y == cont.pts[i + 1].y {
            i += 1;
            if i == cont.pts.len() {
                i -= 1;
                break;
            }
            if i + 1 >= cont.pts.len() {
                break;
            }
        }
        let endpt = i;

        /* check for odd amount of scans, shouldn't happen! */
        if (endpt - bgnpt) % 2 == 0 {
            // continue
        } else {
            let mut j = bgnpt;
            while j < endpt {
                let xmin = cont.pts[j].x as i32;
                let xmax = cont.pts[j + 1].x as i32;
                if xmin >= cont.surf {
                    pix += xmax - xmin;
                }
                j += 1;
                j += 1;
            }
        }
        i += 1;
        scanline += 1;
    }
    let _ = scanline;
    pix
}

/// Original: `imodContourLength` (`icont.c:410`).
///
/// Returns length of contour `cont` in pixels, including the distance from
/// ending to starting point if `closed` is non-zero.
pub fn imod_contour_length(cont: Option<&Icont>, closed: i32) -> f32 {
    let mut dist = 0.0f64;
    let Some(cont) = cont else {
        return -1.;
    };
    if cont.pts.len() < 2 {
        return dist as f32;
    }
    for pt in 1..cont.pts.len() {
        dist += imod_point_distance(&cont.pts[pt], &cont.pts[pt - 1]) as f64;
    }
    if closed != 0 {
        dist += imod_point_distance(&cont.pts[0], &cont.pts[cont.pts.len() - 1]) as f64;
    }
    dist as f32
}

/// Original: `imodel_contour_length` (`icont.c:434`).
///
/// Returns length of contour `cont`, excluding a connection between last and
/// first points.
pub fn imodel_contour_length(cont: Option<&Icont>) -> f64 {
    let mut dist = 0.0f64;
    let Some(cont) = cont else {
        return -1.;
    };
    if cont.pts.len() < 2 {
        return dist;
    }
    for pt in 0..cont.pts.len() - 1 {
        dist += imodel_point_dist(&cont.pts[pt], &cont.pts[pt + 1]);
    }
    dist
}

/// Original: `makeOrReturnScanContour` (`icont.c:453`, static).
///
/// If contour is already a scan, return it; otherwise get scan contour.
fn make_or_return_scan_contour(cont: Option<&Icont>) -> Option<Cow<'_, Icont>> {
    let cont = cont?;
    if cont.flags & ICONT_SCANLINE != 0 {
        return Some(Cow::Borrowed(cont));
    }
    if cont.pts.is_empty() {
        return None;
    }
    imodel_contour_scan(Some(cont)).map(Cow::Owned)
}

/// Original: `cleanupScanContour` (`icont.c:463`, static).
///
/// Delete scan contour if one was made from `cont`.
fn cleanup_scan_contour(cont: &Icont, scont: Cow<'_, Icont>) {
    if cont.flags & ICONT_SCANLINE == 0 {
        drop(scont);
    }
}

/// Original: `imodContourMoment` (`icont.c:477`).
///
/// Returns a moment of the contour `cont` relative to the origin, of order `a`
/// in X and `b` in Y.
pub fn imod_contour_moment(cont: Option<&Icont>, a: i32, b: i32) -> f64 {
    let Some(scont) = make_or_return_scan_contour(cont) else {
        return 0.0;
    };
    let mut moment = 0.0f64;

    let mut pt = 0usize;
    while pt < scont.pts.len() {
        let bline = (scont.pts[pt].x as f64 + 0.5).floor() as i32;
        let eline = (scont.pts[pt + 1].x as f64 + 0.5).floor() as i32;
        let powia = (scont.pts[pt].y as f64).powf(a as f64);
        if b == 0 {
            for _j in bline..eline {
                moment += powia;
            }
        } else if b == 1 {
            for j in bline..eline {
                moment += powia * (j as f64 + 0.5);
            }
        } else {
            for j in bline..eline {
                moment += powia * (j as f64 + 0.5).powf(b as f64);
            }
        }
        pt += 2;
    }

    if let Some(cont) = cont {
        cleanup_scan_contour(cont, scont);
    }
    moment
}

/// Original: `imodContourCenterOfMass` (`icont.c:515`).
///
/// Computes the center of mass of contour `cont` and returns the result in
/// `rpt`.
pub fn imod_contour_center_of_mass(cont: Option<&mut Icont>, rpt: &mut Ipoint) -> i32 {
    let mut weights = 0.0f64;

    rpt.x = 0.0;
    rpt.y = 0.0;
    rpt.z = 0.0;
    let Some(cont) = cont else {
        return 0;
    };
    if cont.pts.is_empty() {
        return 0;
    }

    let save_flags = cont.flags;
    if cont.flags & ICONT_TEMPUSE != 0 {
        cont.flags |= !ICONT_TEMPUSE;
    }
    let err = imodel_contour_centroid(Some(cont), rpt, &mut weights);
    cont.flags = save_flags;
    if err != 0 {
        return err;
    }

    if weights.abs() > 1.0e-20 {
        rpt.x = (rpt.x as f64 / weights) as f32;
        rpt.y = (rpt.y as f64 / weights) as f32;
        rpt.z = (rpt.z as f64 / weights) as f32;
    }
    0
}

/// Original: `imodel_contour_centroid` (`icont.c:551`).
///
/// Computes components of the centroid of contour `icont`, returning the sum of
/// pixel locations in `rcp` and the pixel sum or length in `rtw`.
pub fn imodel_contour_centroid(icont: Option<&Icont>, rcp: &mut Ipoint, rtw: &mut f64) -> i32 {
    let min_weight: f32 = 0.01;
    let scale = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };

    let Some(icont) = icont else {
        return -1;
    };
    if icont.pts.is_empty() {
        return -1;
    }

    if icont.pts.len() == 1 {
        rcp.x = icont.pts[0].x;
        rcp.y = icont.pts[0].y;
        rcp.z = icont.pts[0].z;
        *rtw = 1.;
        return 0;
    }

    rcp.x = 0.0;
    rcp.y = 0.0;
    *rtw = 0.;

    /* Handle contour in open contour object */
    if icont.flags & ICONT_TEMPUSE != 0 {
        rcp.z = 0.;
        for i in 1..icont.pts.len() {
            let weight = imod_point3d_scale_distance(&icont.pts[i - 1], &icont.pts[i], &scale);
            rcp.x += weight * (icont.pts[i - 1].x + icont.pts[i].x) * 0.5f32;
            rcp.y += weight * (icont.pts[i - 1].y + icont.pts[i].y) * 0.5f32;
            rcp.z += weight * (icont.pts[i - 1].z + icont.pts[i].z) * 0.5f32;
            *rtw += weight as f64;
        }
        return 0;
    }

    /* Use scan contour for closed contour object */
    rcp.z = icont.pts[0].z;
    let Some(cont) = make_or_return_scan_contour(Some(icont)) else {
        return -1;
    };

    let mut pix: i32 = 0;
    let mut i: usize = 0;
    let mut scanline: i32 = 0;
    while i + 1 < cont.pts.len() {
        let y = cont.pts[i].y as i32;
        let bgnpt = i;
        /* find all scans for this y line. */
        while i < cont.pts.len() - 1 && cont.pts[i].y == cont.pts[i + 1].y {
            i += 1;
        }
        let endpt = i;

        /* check for odd amount of scans, shouldn't happen! */
        if (endpt - bgnpt) % 2 == 0 {
            unsafe {
                libc::printf(
                    c" (Error scan line %d,%d)\n".as_ptr(),
                    scanline as std::ffi::c_int,
                    y as std::ffi::c_int,
                );
            }
        } else {
            /* add all points in each scan. */
            let mut j = bgnpt;
            while j < endpt {
                let xmin = cont.pts[j].x as i32;
                let xmax = cont.pts[j + 1].x as i32;
                let xval = (xmin + xmax) as f32 * 0.5f32;
                let yval = cont.pts[j].y;
                let weight = if min_weight > (xmax - xmin) as f32 {
                    min_weight
                } else {
                    (xmax - xmin) as f32
                };
                rcp.x += xval * weight;
                rcp.y += yval * weight;
                *rtw += weight as f64;

                if xmin >= cont.surf {
                    pix += xmax - xmin;
                }
                j += 1;
                j += 1;
            }
        }
        i += 1;
        scanline += 1;
    }
    let _ = pix;
    cleanup_scan_contour(icont, cont);
    rcp.z = (rcp.z as f64 * *rtw) as f32;
    0
}

/// Original: `imodContourCenterMoment` (`icont.c:642`).
///
/// Returns the moment of contour `cont` of order `a` in X and order `b` in Y,
/// relative to the point `org`.
pub fn imod_contour_center_moment(cont: Option<&Icont>, org: &Ipoint, a: i32, b: i32) -> f64 {
    let Some(scont) = make_or_return_scan_contour(cont) else {
        return 0.0;
    };
    let mut moment = 0.0f64;

    let mut pt = 0usize;
    while pt < scont.pts.len() {
        let powia = ((scont.pts[pt].y - org.y) as f64).powf(a as f64);
        let bline = ((scont.pts[pt].x - org.x) as f64 + 0.5).floor() as i32;
        let eline = ((scont.pts[pt + 1].x - org.x) as f64 + 0.5).floor() as i32;
        for j in bline..eline {
            moment += powia * (j as f64 + 0.5).powf(b as f64);
        }
        pt += 2;
    }

    if let Some(cont) = cont {
        cleanup_scan_contour(cont, scont);
    }
    moment
}

/// Original: `imodContourEquivEllipse` (`icont.c:674`).
///
/// Computes an ellipse whose moments are the same as the area inside contour
/// `cont`.  Returns 1 for error.
pub fn imod_contour_equiv_ellipse(
    cont: Option<&Icont>,
    center: &mut Ipoint,
    long_axis: &mut f32,
    short_axis: &mut f32,
    angle: &mut f32,
) -> i32 {
    let pi4inv: f64 = 4. / 3.1415927;
    let mut scale: f32 = 1.;

    let Some(cont) = cont else {
        return 1;
    };
    if cont.pts.len() < 3 {
        return 1;
    }

    let mut big_cont_owned: Option<Icont> = None;
    if cont.flags & ICONT_SCANLINE == 0 {
        let Some(mut big) = imod_contour_dup(cont) else {
            return 1;
        };
        scale = 4.;
        for pt in 0..big.pts.len() {
            big.pts[pt].x *= scale;
            big.pts[pt].y *= scale;
        }
        big_cont_owned = Some(big);
    }
    let big_cont: &Icont = big_cont_owned.as_ref().unwrap_or(cont);
    let Some(scont) = make_or_return_scan_contour(Some(big_cont)) else {
        return 1;
    };

    let mut scont_owned = scont.into_owned();
    let err = imod_contour_center_of_mass(Some(&mut scont_owned), center);
    let (m11, m20, m02);
    if err == 0 {
        m11 = imod_contour_center_moment(Some(&scont_owned), center, 1, 1);
        m20 = imod_contour_center_moment(Some(&scont_owned), center, 2, 0);
        m02 = imod_contour_center_moment(Some(&scont_owned), center, 0, 2);
    } else {
        m11 = 0.;
        m20 = 0.;
        m02 = 0.;
    }
    if err != 0 {
        return 1;
    }

    if m11 == 0. || m20 == 0. || m02 == 0. {
        return 1;
    }
    let discr = (4. * m11 * m11 + (m02 - m20) * (m02 - m20)).sqrt();
    let lambda1 = (m02 + m20 + discr) / 2.;
    let lambda2 = (m02 + m20 - discr) / 2.;
    *angle = (m11.atan2(lambda1 - m20) / 0.01745329252) as f32;
    if *angle < 0. {
        *angle += 180.;
    }

    *long_axis = ((((pi4inv * pi4inv / lambda2).ln() + 3. * lambda1.ln()) / 8.).exp()) as f32;
    *short_axis = (pi4inv * lambda2 / *long_axis as f64).powf(0.3333333) as f32;
    center.x /= scale;
    center.y /= scale;
    *long_axis /= scale;
    *short_axis /= scale;
    center.z = 0.;
    for pt in 0..cont.pts.len() {
        center.z += cont.pts[pt].z / cont.pts.len() as f32;
    }
    0
}

/// Original: `imodContourCircularity` (`icont.c:743`).
///
/// Returns the circularity of a closed contour, based on perimeter squared to
/// area.  A perfect circle = 1.0, a square = 1.27.  Returns 1000. for error.
pub fn imod_contour_circularity(cont: Option<&Icont>) -> f64 {
    let c = imodel_contour_length(cont);
    let a = imodel_contour_area(cont) as f64;
    if a == 0.0 {
        return 1000.0;
    }
    (c * c) / (12.56637062 * a)
}

/// Original: `imodContourLongAxis` (`icont.c:759`).
///
/// Measures the angle at which the contour is most elongated, to the precision
/// set by `precision`, in degrees.  Returns the angle to the long axis in
/// radians, the ratio of long to short axis in `aspect`, and the length of the
/// long axis in `longaxis`.
pub fn imod_contour_long_axis(
    cont: Option<&Icont>,
    precision: f32,
    aspect: &mut f32,
    longaxis: &mut f32,
) -> f64 {
    let dtor: f64 = 0.017453293;
    let ntrial = (90. / precision) as i32;

    *aspect = 1.;
    *longaxis = 0.;
    if imod_contour_bad(cont, 1) != 0 {
        return 0.0;
    }
    let cont = cont.unwrap();

    if cont.pts.len() == 2 {
        let center = Ipoint {
            x: cont.pts[1].x - cont.pts[0].x,
            y: cont.pts[1].y - cont.pts[0].y,
            z: 0.,
        };
        *aspect = 1.0e6;
        *longaxis = ((center.x * center.x + center.y * center.y) as f64).sqrt() as f32;
        return imod_point2d_angle(&center);
    }

    /* try every angle at the given precision from 0 to 90 degrees */
    let mut minratio: f32 = 1.0e30;
    let mut minangle: f64 = 0.;
    let mut minlong: f32 = 0.;
    for itry in 0..ntrial {
        let angle = itry as f64 * precision as f64 * dtor;
        let cosa = angle.cos() as f32;
        let sina = angle.sin() as f32;
        let mut xmin: f32 = 1.0e30;
        let mut ymin: f32 = 1.0e30;
        let mut xmax: f32 = -1.0e30;
        let mut ymax: f32 = -1.0e30;

        /* rotate points, find min and max */
        for pt in 0..cont.pts.len() {
            let xrot = cont.pts[pt].x * cosa - cont.pts[pt].y * sina;
            let yrot = cont.pts[pt].x * sina + cont.pts[pt].y * cosa;
            if xrot > xmax {
                xmax = xrot;
            }
            if xrot < xmin {
                xmin = xrot;
            }
            if yrot > ymax {
                ymax = yrot;
            }
            if yrot < ymin {
                ymin = yrot;
            }
        }
        let xran = xmax - xmin;
        let yran = ymax - ymin;
        let mut ratio: f32 = 1.0e6;

        /* if the ratio is a new minimum and height is less than width,
        this angle is a good as is; otherwise it's off by 90 degrees */
        if yran < xran {
            if xran > 1.0e-6 * yran {
                ratio = yran / xran;
            }
            if ratio < minratio {
                minratio = ratio;
                minangle = -angle;
                minlong = xran;
            }
        } else {
            if yran > 1.0e-6 * xran {
                ratio = xran / yran;
            }
            if ratio < minratio {
                minratio = ratio;
                minangle = 90. * dtor - angle;
                minlong = yran;
            }
        }
    }
    *aspect = 1.0e6;
    if minratio > 1.0e-6 {
        *aspect = 1. / minratio;
    }
    *longaxis = minlong;
    minangle
}

/// Original: `imodContourFitPlane` (`icont.c:844`).
///
/// Fits a plane to the points in contour `cont`, with coordinates scaled by
/// `scale`.  The normal to the plane is returned in `norm` and the constant
/// term in `dval`; the rotations about the Y axis then the X axis that will
/// make the plane be flat are returned in `beta` and `alpha`.  Returns 1 if
/// there are too few points in the contour or if the points are too close to
/// colinear.
pub fn imod_contour_fit_plane(
    cont: &Icont,
    scale: &Ipoint,
    norm: &mut Ipoint,
    dval: &mut f32,
    alpha: &mut f64,
    beta: &mut f64,
) -> i32 {
    let mut a = [[0.0f64; 3]; 3];
    let small: f64 = 1.0e-4;
    /* RADIANS_PER_DEGREE (`b3dutil.h:68`) */
    let pi: f64 = 180. * 0.01745329252;
    let pts = &cont.pts;
    let mut v1 = Ipoint::default();
    let mut v2 = Ipoint::default();
    let mut pt_norm = Ipoint::default();
    let mut pt_back = Ipoint::default();
    let mut yp: f32;

    let n = cont.pts.len() as i32;
    if n < 3 {
        return 1;
    }

    // Find the normal to 3 point triangle with cross-product
    let i1 = (n / 3) as usize;
    let i2 = ((2 * n) / 3) as usize;
    v1.x = pts[i1].x - pts[0].x;
    v1.y = pts[i1].y - pts[0].y;
    v1.z = pts[i1].z - pts[0].z;
    v2.x = pts[i2].x - pts[0].x;
    v2.y = pts[i2].y - pts[0].y;
    v2.z = pts[i2].z - pts[0].z;
    imod_point_cross(&v1, &v2, &mut pt_norm);

    // Get rotation around Z that bring normal to Y direction
    let gamma: f64;
    if (pt_norm.x as f64) < 1.0e-10 * pt_norm.z as f64
        && (pt_norm.y as f64) < 1.0e-10 * pt_norm.z as f64
    {
        gamma = 0.;
    } else {
        gamma = pi / 2. - (pt_norm.y as f64).atan2(pt_norm.x as f64);
    }
    yp = (pt_norm.x as f64 * gamma.sin() + pt_norm.y as f64 * gamma.cos()) as f32;

    // Rotate normal to Y and get rotation that rotates it up to Z axis
    let alfa = pi / 2. - (pt_norm.z as f64).atan2(yp as f64);
    let mut cosa = alfa.cos();
    let sina = alfa.sin();
    let cosg = gamma.cos();
    let sing = gamma.sin();

    /* Form sums of all kinds */
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    let mut sxsq = 0.0f64;
    let mut sysq = 0.0f64;
    let mut szsq = 0.0f64;
    let mut sxy = 0.0f64;
    let mut sxz = 0.0f64;
    let mut syz = 0.0f64;
    for i in 0..n as usize {
        let xs = pts[i].x * scale.x;
        let ys = pts[i].y * scale.y;
        let zs = pts[i].z * scale.z;
        let x = (xs as f64 * cosg - ys as f64 * sing) as f32;
        yp = (xs as f64 * sing + ys as f64 * cosg) as f32;
        let y = (yp as f64 * cosa - zs as f64 * sina) as f32;
        let z = (yp as f64 * sina + zs as f64 * cosa) as f32;
        sx += x as f64;
        sy += y as f64;
        sz += z as f64;
        sxsq += (x * x) as f64;
        sysq += (y * y) as f64;
        szsq += (z * z) as f64;
        sxy += (x * y) as f64;
        sxz += (x * z) as f64;
        syz += (y * z) as f64;
    }

    /* Compute the terms of the 3 equations in the normal components, which are:
    -ai0*cos(a)*sin(b) + a11*sin(a) + ai2*cos(a)*cos(b) = 0   for i = 0,1,2
    */
    let xm = sx / n as f64;
    let ym = sy / n as f64;
    let zm = sz / n as f64;
    a[0][0] = sxsq - xm * sx;
    a[0][1] = sxy - ym * sx;
    a[0][2] = sxz - zm * sx;
    a[1][0] = sxy - xm * sy;
    a[1][1] = sysq - ym * sy;
    a[1][2] = syz - zm * sy;
    a[2][0] = sxz - xm * sz;
    a[2][1] = syz - ym * sz;
    a[2][2] = szsq - zm * sz;

    /* Combine the first and second equation, or first and third if that fails */
    for i in 1..3 {
        let mut sinfac = a[0][1] * a[i][0] - a[i][1] * a[0][0];
        let mut cosfac = a[0][1] * a[i][2] - a[i][1] * a[0][2];

        /* If both factors are too small, the equations are redundant due to
        colinear points on the two dimensions, so try other pair or quit */
        if sinfac.abs() < small && cosfac.abs() < small {
            if i == 1 {
                continue;
            }
            return 1;
        }

        /* get angle between +/- 90 */
        if sinfac < 0. {
            sinfac = -sinfac;
            cosfac = -cosfac;
        }
        *beta = cosfac.atan2(sinfac);
        break;
    }

    let sinb = (*beta).sin();
    let cosb = (*beta).cos();
    for i in 0..3 {
        let mut sinfac = a[i][1];
        let mut cosfac = a[i][0] * sinb - a[i][2] * cosb;

        /* If both factors are too small, try the next equation. */
        if sinfac.abs() < small && cosfac.abs() < small {
            if i < 2 {
                continue;
            }
            return 1;
        }

        /* again get angle between +/- 90 */
        if sinfac < 0. {
            sinfac = -sinfac;
            cosfac = -cosfac;
        }
        *alpha = cosfac.atan2(sinfac);
        break;
    }

    cosa = (*alpha).cos();

    /* Rotate the normal back to rotated coordinates */
    pt_norm.x = (-cosa * sinb) as f32;
    pt_norm.y = (*alpha).sin() as f32;
    pt_norm.z = (cosa * cosb) as f32;

    // Back-rotate to native coordinates and make Z positive
    let Some(mut mat) = imod_mat_new(3) else {
        return 2;
    };
    if imod_mat_rot(&mut mat, -alfa / 0.01745329252, B3D_X) != 0
        || imod_mat_rot(&mut mat, -gamma / 0.01745329252, B3D_Z) != 0
    {
        imod_mat_delete(&mut mat);
        return 2;
    }
    imod_mat_transform(&mat, &pt_norm, norm);

    if norm.z < 0. {
        norm.x = -norm.x;
        norm.y = -norm.y;
        norm.z = -norm.z;
    }

    // Back-rotate mean for getting the D value
    pt_norm.x = xm as f32;
    pt_norm.y = ym as f32;
    pt_norm.z = zm as f32;
    imod_mat_transform(&mat, &pt_norm, &mut pt_back);
    imod_mat_delete(&mut mat);

    *dval = -(pt_back.x * norm.x + pt_back.y * norm.y + pt_back.z * norm.z);

    /* Find the angles again */
    *beta = 0.;
    *alpha = 0.;
    if (norm.x as f64).abs() > small || (norm.z as f64).abs() > small {
        *beta = -(norm.x as f64).atan2(norm.z as f64);
    }
    let zrot = norm.z as f64 * (*beta).cos() - norm.x as f64 * (*beta).sin();
    if zrot.abs() > small || (norm.y as f64).abs() > small {
        *alpha = -(zrot.atan2(norm.y as f64) - 1.570796327);
    }
    0
}

/// Original: `imodContourGetBBox` (`icont.c:1024`).
///
/// Calculates the full 3D bounding box of contour `cont`.
pub fn imod_contour_get_bbox(cont: Option<&Icont>, ll: &mut Ipoint, ur: &mut Ipoint) -> i32 {
    let Some(cont) = cont else {
        return -1;
    };
    if cont.pts.is_empty() {
        return -1;
    }

    ll.x = cont.pts[0].x;
    ur.x = cont.pts[0].x;
    ll.y = cont.pts[0].y;
    ur.y = cont.pts[0].y;
    ll.z = cont.pts[0].z;
    ur.z = cont.pts[0].z;

    for pt in 0..cont.pts.len() {
        if cont.pts[pt].x < ll.x {
            ll.x = cont.pts[pt].x;
        }
        if cont.pts[pt].y < ll.y {
            ll.y = cont.pts[pt].y;
        }
        if cont.pts[pt].z < ll.z {
            ll.z = cont.pts[pt].z;
        }
        if cont.pts[pt].x > ur.x {
            ur.x = cont.pts[pt].x;
        }
        if cont.pts[pt].y > ur.y {
            ur.y = cont.pts[pt].y;
        }
        if cont.pts[pt].z > ur.z {
            ur.z = cont.pts[pt].z;
        }
    }
    0
}

/// Original: `imodContourZValue` (`icont.c:1060`).
pub fn imod_contour_z_value(cont: Option<&Icont>) -> i32 {
    match cont {
        None => -1,
        Some(c) if c.pts.is_empty() => -1,
        Some(_) => (imod_contour_float_z_value(cont) as f64 + 0.5).floor() as i32,
    }
}

/// Original: `imodContourFloatZValue` (`icont.c:1070`).
pub fn imod_contour_float_z_value(cont: Option<&Icont>) -> f32 {
    let mut z = 0.0f32;
    let Some(cont) = cont else {
        return -1.0e20;
    };
    if cont.pts.is_empty() {
        return -1.0e20;
    }
    for p in 0..cont.pts.len() {
        z += cont.pts[p].z;
    }
    z / cont.pts.len() as f32
}

/// Original: `imodContZDirection` (`icont.c:1090`).
///
/// Determines whether contour `cont` is clockwise or counter-clockwise.
pub fn imod_cont_z_direction(cont: Option<&Icont>) -> i32 {
    let mut a = 0.0f64;
    let Some(cont) = cont else {
        return 0;
    };
    if cont.pts.len() < 3 {
        return 0;
    }

    let mpt = cont.pts.len();
    for pt in 0..mpt {
        let mut nextp = pt + 1;
        if nextp == mpt {
            nextp = 0;
        }
        a += ((cont.pts[pt].y + cont.pts[nextp].y) * (cont.pts[nextp].x - cont.pts[pt].x)) as f64;
    }
    if a < 0. {
        return IMOD_CONTOUR_COUNTER_CLOCKWISE;
    }
    if a > 0. {
        return IMOD_CONTOUR_CLOCKWISE;
    }
    0
}

/// Original: `imodel_contour_on` (`icont.c:1119`).
///
/// Returns non-zero if (`x`, `y`) is in point list of contour `cont`, or 0 if
/// it is not.  (unused 4/22/05)
pub fn imodel_contour_on(cont: Option<&Icont>, x: i32, y: i32) -> i32 {
    let Some(cont) = cont else {
        return 0;
    };
    let mut retval;

    for i in 0..cont.pts.len() as i32 {
        let mut prev = i - 1;
        let mut next = i + 1;
        if prev < 0 {
            prev = cont.pts.len() as i32 - 1;
        }
        if next == cont.pts.len() as i32 {
            next = 0;
        }

        if (cont.pts[i as usize].x == x as f32) && (cont.pts[i as usize].y == y as f32) {
            retval = 10;

            if cont.pts[i as usize].y == cont.pts[next as usize].y {
                retval = 2;
            } else {
                while cont.pts[i as usize].y == cont.pts[prev as usize].y {
                    prev -= 1;
                    if prev < 0 {
                        prev = cont.pts.len() as i32 - 1;
                    }
                }
            }

            if (cont.pts[i as usize].y < cont.pts[prev as usize].y)
                && (cont.pts[i as usize].y < cont.pts[next as usize].y)
            {
                retval = 1;
            }

            if (cont.pts[i as usize].y > cont.pts[prev as usize].y)
                && (cont.pts[i as usize].y > cont.pts[next as usize].y)
            {
                retval = 1;
            }

            return retval;
        }
    }
    0
}

/// Original: `imodContourNearest` (`icont.c:1169`).
///
/// Returns the index of the point in contour `cont` that is nearest in 3D to
/// point `pnt`, or -1 for an error.
pub fn imod_contour_nearest(cont: Option<&Icont>, pnt: &Ipoint) -> i32 {
    let Some(cont) = cont else {
        return -1;
    };
    let mut index = 0i32;
    if cont.pts.is_empty() {
        return -1;
    }

    let mut dx = cont.pts[0].x - pnt.x;
    let mut dy = cont.pts[0].y - pnt.y;
    let mut dz = cont.pts[0].z - pnt.z;

    let mut dist = (dx * dx) + (dy * dy) + (dz * dz);

    for i in 1..cont.pts.len() {
        dx = cont.pts[i].x - pnt.x;
        dy = cont.pts[i].y - pnt.y;
        dz = cont.pts[i].z - pnt.z;

        let tdist = (dx * dx) + (dy * dy) + (dz * dz);

        if tdist < dist {
            dist = tdist;
            index = i as i32;
        }
    }
    index
}

/// Original: `imodel_contour_nearest` (`icont.c:1203`).
///
/// Returns the index of the point in contour `cont` that is nearest in X and Y
/// to (`x`, `y`), or -1 for an error.  Unused 3/29/05.
pub fn imodel_contour_nearest(cont: Option<&Icont>, x: i32, y: i32) -> i32 {
    let Some(cont) = cont else {
        return -1;
    };
    let mut index = 0i32;
    if cont.pts.is_empty() {
        return -1;
    }
    let mut dist = (((cont.pts[0].x - x as f32) * (cont.pts[0].x - x as f32))
        + ((cont.pts[0].y - y as f32) * (cont.pts[0].y - y as f32))) as f64;

    for i in 1..cont.pts.len() {
        let tdist = (((cont.pts[i].x - x as f32) * (cont.pts[i].x - x as f32))
            + ((cont.pts[i].y - y as f32) * (cont.pts[i].y - y as f32))) as f64;
        if tdist < dist {
            dist = tdist;
            index = i as i32;
        }
    }
    index
}

/// Original: `imodContourInsideCont` (`icont.c:1229`).
///
/// Returns 1 if contour `inner` is completely inside or touching contour
/// `outer`, otherwise returns 0.
pub fn imod_contour_inside_cont(inner: &Icont, outer: &Icont) -> i32 {
    for pt in 0..inner.pts.len() {
        if imod_point_inside_cont(outer, &inner.pts[pt]) == 0 {
            return 0;
        }
    }
    1
}

/****************************************************************************/
/* CONTOUR-MODIFYING FUNCTIONS                                              */

/// Original: `imodContourJoin` (`icont.c:1263`).
///
/// Joins two contours `c1` and `c2` by adding connecting lines between them
/// and arranging points from the two contours into a single path.  See the
/// source comment for the meaning of `st1`, `st2`, `fill` and `counterdir`.
///
/// Deviation note: the source's `Istore item;` for the end-of-contour gap is a
/// stack variable whose `value` member is never assigned, so the gap item
/// carries stack garbage there; a zeroed `value` is used here (CLAUDE.md,
/// "Uninitialised memory").
pub fn imod_contour_join(
    c1: Option<&mut Icont>,
    c2: Option<&mut Icont>,
    mut st1: i32,
    mut st2: i32,
    fill: i32,
    counterdir: i32,
) -> Option<Icont> {
    let mut nstore: Vec<Istore> = Vec::new();
    let mut add_pt = [0i32; 4];

    let (c1, c2) = match (c1, c2) {
        (None, Some(c2)) => return imod_contour_dup(c2),
        (Some(c1), None) => return imod_contour_dup(c1),
        (None, None) => return None,
        (Some(a), Some(b)) => (a, b),
    };

    /* If one of the contours is open, make sure it is the first one so
    the opening is preserved */
    let (c1, c2) = if (c2.flags & ICONT_OPEN) != 0 && (c1.flags & ICONT_OPEN) == 0 {
        let pt = st1;
        st1 = st2;
        st2 = pt;
        (c2, c1)
    } else {
        (c1, c2)
    };

    /* If second contour is open, insert a gap at its end */
    if (c2.flags & ICONT_OPEN) != 0 && istore_point_is_gap(&c2.store, c2.pts.len() as i32 - 1) == 0
    {
        let item = Istore {
            /* GEN_STORE_GAP (`istore.h:41`), GEN_STORE_ONEPOINT (`istore.h:32`) */
            type_: 4,
            flags: 1 << 7,
            index: StoreUnion {
                i: c2.pts.len() as i32 - 1,
            },
            value: StoreUnion::default(),
        };
        istore_add_one_index_item(&mut c2.store, item);
    }

    let dir1 = imod_cont_z_direction(Some(c1));
    let dir2 = imod_cont_z_direction(Some(c2));
    if st1 >= 0 && st2 >= 0 {
        /* adjust points if they will be inverted */
        if dir1 != IMOD_CONTOUR_CLOCKWISE {
            st1 = c1.pts.len() as i32 - 1 - st1;
        }
        if (counterdir == 0 && dir2 != IMOD_CONTOUR_CLOCKWISE)
            || (counterdir != 0 && dir2 == IMOD_CONTOUR_CLOCKWISE)
        {
            st2 = c2.pts.len() as i32 - 1 - st2;
        }
    }

    /* DNM: changed from COUNTER to CLOCKWISE when inverted sense of flags */
    if dir1 != IMOD_CONTOUR_CLOCKWISE {
        imod_contour_make_direction(c1, IMOD_CONTOUR_CLOCKWISE);
    }
    if counterdir == 0 && dir2 != IMOD_CONTOUR_CLOCKWISE {
        imod_contour_make_direction(c2, IMOD_CONTOUR_CLOCKWISE);
    }
    if counterdir != 0 && dir2 == IMOD_CONTOUR_CLOCKWISE {
        imod_contour_make_direction(c2, IMOD_CONTOUR_COUNTER_CLOCKWISE);
    }

    if st1 < 0 || st2 < 0 {
        /* find closest points */
        let mut mdist = imodel_point_dist(&c1.pts[0], &c2.pts[0]);
        st1 = 0;
        st2 = 0;
        for pt in 0..c1.pts.len() {
            for pt2 in 0..c2.pts.len() {
                let dist = imodel_point_dist(&c1.pts[pt], &c2.pts[pt2]);
                if dist < mdist {
                    st1 = pt as i32;
                    st2 = pt2 as i32;
                    mdist = dist;
                }
            }
        }
    }

    /* set up new contour and fill point. */
    let mut cont = imod_contour_new()?;
    let mut point = Ipoint {
        x: (c1.pts[st1 as usize].x + c2.pts[st2 as usize].x) * 0.5f32,
        y: (c1.pts[st1 as usize].y + c2.pts[st2 as usize].y) * 0.5f32,
        z: (c1.pts[st1 as usize].z + c2.pts[st2 as usize].z) * 0.5f32,
    };
    if fill == 1 {
        point.z += 0.75f32;
    }
    if fill == -1 {
        point.z -= 0.75f32;
    }

    /* Take care of joining storage lists first, but keep new list separate
    to avoid having it get renumbered.  Propagate the properties at st1 and
    st2 into the fill point */
    if !c1.store.is_empty() || !c2.store.is_empty() {
        let lst2 = st1 + 1 + if fill != 0 { 1 } else { 0 };
        let lst3 = lst2 + c2.pts.len() as i32 - st2;
        let lst4 = st1 + c2.pts.len() as i32 + 2 + if fill != 0 { 2 } else { 0 };
        let c1size = c1.pts.len() as i32;
        let c2size = c2.pts.len() as i32;
        if istore_extract_changes(&c1.store, &mut nstore, 0, st1, 0, c1size) != 0
            || (fill != 0
                && istore_extract_changes(&c1.store, &mut nstore, st1, st1, st1 + 1, c1size) != 0)
            || istore_extract_changes(&c2.store, &mut nstore, st2, c2size - 1, lst2, c2size) != 0
            || istore_extract_changes(&c2.store, &mut nstore, 0, st2, lst3, c2size) != 0
            || (fill != 0
                && istore_extract_changes(&c2.store, &mut nstore, st2, st2, lst4 - 1, c2size) != 0)
            || istore_extract_changes(&c1.store, &mut nstore, st1, c1size - 1, lst4, c1size) != 0
            || istore_copy_non_index(&c1.store, &mut nstore) != 0
            || istore_copy_non_index(&c2.store, &mut nstore) != 0
        {
            return None;
        }
    }

    /* add points to new contour, backing off from drawing an overlapping
    connector line unless the fill flag is set */
    for pt in 0..st1 as usize {
        imod_point_append(&mut cont, c1.pts[pt]);
    }
    backoff_and_add_point(&mut cont, c1, st1, -1, fill);
    add_pt[0] = cont.pts.len() as i32 - 1;
    if fill == 1 || fill == -1 {
        imod_point_append(&mut cont, point);
    }
    backoff_and_add_point(&mut cont, c2, st2, 1, fill);
    add_pt[1] = cont.pts.len() as i32 - 1;
    for pt in (st2 + 1) as usize..c2.pts.len() {
        imod_point_append(&mut cont, c2.pts[pt]);
    }
    for pt in 0..st2 as usize {
        imod_point_append(&mut cont, c2.pts[pt]);
    }
    backoff_and_add_point(&mut cont, c2, st2, -1, fill);
    add_pt[2] = cont.pts.len() as i32 - 1;
    if fill == 1 || fill == -1 {
        imod_point_append(&mut cont, point);
    }
    backoff_and_add_point(&mut cont, c1, st1, 1, fill);
    add_pt[3] = cont.pts.len() as i32 - 1;
    for pt in (st1 + 1) as usize..c1.pts.len() {
        imod_point_append(&mut cont, c1.pts[pt]);
    }

    let mut pt2 = 0i32;
    if !c1.sizes.is_empty() {
        for pt in 0..=st1 as usize {
            imod_point_set_size(&mut cont, pt2, c1.sizes[pt]);
            pt2 += 1;
        }
    } else {
        pt2 += st1 + 1;
    }

    if fill == 1 || fill == -1 {
        pt2 += 1;
    }

    if !c2.sizes.is_empty() {
        for pt in st2 as usize..c2.pts.len() {
            imod_point_set_size(&mut cont, pt2, c2.sizes[pt]);
            pt2 += 1;
        }
        for pt in 0..=st2 as usize {
            imod_point_set_size(&mut cont, pt2, c2.sizes[pt]);
            pt2 += 1;
        }
    } else {
        pt2 += c2.pts.len() as i32 + 1;
    }

    if fill == 1 || fill == -1 {
        pt2 += 1;
    }

    if !c1.sizes.is_empty() {
        for pt in st1 as usize..c1.pts.len() {
            imod_point_set_size(&mut cont, pt2, c1.sizes[pt]);
            pt2 += 1;
        }
    }

    // Fill value > 10 indicates how many pixels of separation; back off each
    // side of junction by half that after handling all the point properties
    // and sizes
    if fill > 10 {
        let mdist = (fill - 10) as f64 / 2.;
        let mut lst2 = 3i32;
        while lst2 >= 0 {
            if cont.pts.len() < 8 {
                lst2 -= 1;
                continue;
            }
            point.x = cont.pts[add_pt[lst2 as usize] as usize].x;
            point.y = cont.pts[add_pt[lst2 as usize] as usize].y;
            let dir1 = if (lst2 % 2) != 0 { 1 } else { -1 };

            // Find first point past the limit if possible
            let mut lst3 = add_pt[lst2 as usize];
            let mut lst4 = (lst3 + cont.pts.len() as i32 + dir1) % cont.pts.len() as i32;
            let mut pt2 = 0i32;
            while (imod_point_distance(&cont.pts[lst4 as usize], &point) as f64) < mdist
                && pt2 < cont.pts.len() as i32 / 5
            {
                lst3 = lst4;
                lst4 = (lst3 + cont.pts.len() as i32 + dir1) % cont.pts.len() as i32;
                pt2 += 1;
            }

            // If there is actually a point past limit, move previous point to
            // the limit
            let dist = imod_point_distance(&cont.pts[lst4 as usize], &point) as f64;
            if dist > mdist {
                let last_dist = imod_point_distance(&cont.pts[lst3 as usize], &point) as f64;
                let frac = (mdist - last_dist) / (dist - last_dist);
                cont.pts[lst3 as usize].x = (cont.pts[lst3 as usize].x as f64
                    + frac * (cont.pts[lst4 as usize].x - cont.pts[lst3 as usize].x) as f64)
                    as f32;
                cont.pts[lst3 as usize].y = (cont.pts[lst3 as usize].y as f64
                    + frac * (cont.pts[lst4 as usize].y - cont.pts[lst3 as usize].y) as f64)
                    as f32;
            } else {
                // Otherwise set up to delete one more point
                pt2 += 1;
            }

            // Delete pt2 points
            lst3 = add_pt[lst2 as usize];
            for _pt in 0..pt2 {
                imod_point_delete(&mut cont, lst3);
                lst3 = (lst3 + cont.pts.len() as i32 + dir1) % cont.pts.len() as i32;
            }
            lst2 -= 1;
        }
    }

    /* transfer open flag from c1 */
    if c1.flags & ICONT_OPEN != 0 {
        cont.flags |= ICONT_OPEN;
    }

    istore_clean_ends(&mut nstore);
    cont.store = nstore;
    Some(cont)
}

/// Original: `backoffAndAddPoint` (`icont.c:1479`, static).
///
/// Adds an endpoint of the connector when joining contours, which is at point
/// `join_pt` in contour `cfrom`, and is being added to `cont`.  If `keep_pos`
/// is not non-zero, it backs the point off by up to 1 pixel or halfway to the
/// previous/next point depending on the sign of `direc`.
fn backoff_and_add_point(cont: &mut Icont, cfrom: &Icont, join_pt: i32, direc: i32, keep_pos: i32) {
    let mut point = Ipoint::default();
    let dir_fac = if direc >= 0 {
        1
    } else {
        cfrom.pts.len() as i32 - 1
    };
    let last_pt = ((join_pt + dir_fac) % cfrom.pts.len() as i32) as usize;
    let seglen = imod_point_distance(&cfrom.pts[join_pt as usize], &cfrom.pts[last_pt]);
    if keep_pos != 0 || cfrom.pts.len() < 3 || seglen < 1.0e-3 {
        imod_point_append(cont, cfrom.pts[join_pt as usize]);
        return;
    }
    point.z = cfrom.pts[join_pt as usize].z;
    /* B3DMIN(0.5, 1. / seglen) is evaluated in double and stored in a float */
    let frac = (if 0.5f64 < 1. / seglen as f64 {
        0.5f64
    } else {
        1. / seglen as f64
    }) as f32;
    point.x = cfrom.pts[join_pt as usize].x
        + frac * (cfrom.pts[last_pt].x - cfrom.pts[join_pt as usize].x);
    point.y = cfrom.pts[join_pt as usize].y
        + frac * (cfrom.pts[last_pt].y - cfrom.pts[join_pt as usize].y);
    imod_point_append(cont, point);
}

/// Original: `imodContourSplice` (`icont.c:1501`).
///
/// Returns a contour with points 0 to `p1` from contour `c1` and points from
/// `p2` to the end from contour `c2`.
pub fn imod_contour_splice(
    c1: Option<&Icont>,
    c2: Option<&Icont>,
    p1: i32,
    p2: i32,
) -> Option<Icont> {
    let mut nstore: Vec<Istore> = Vec::new();

    let (Some(c1), Some(c2)) = (c1, c2) else {
        return None;
    };

    if (p1 < 0) || (p2 < 0) || (p1 >= c1.pts.len() as i32) || (p2 >= c2.pts.len() as i32) {
        return None;
    }

    let mut nc = imod_contour_new()?;

    /* Take care of any store items first */
    /* Extract changes from each and copy any non-index items */
    if !c1.store.is_empty() || !c2.store.is_empty() {
        if istore_extract_changes(&c1.store, &mut nstore, 0, p1, 0, c1.pts.len() as i32) != 0
            || istore_extract_changes(
                &c2.store,
                &mut nstore,
                p2,
                c2.pts.len() as i32 - 1,
                p1 + 1,
                c2.pts.len() as i32,
            ) != 0
            || istore_copy_non_index(&c1.store, &mut nstore) != 0
            || istore_copy_non_index(&c2.store, &mut nstore) != 0
        {
            return None;
        }
    }

    for i in 0..=p1 as usize {
        imod_point_append(&mut nc, c1.pts[i]);
    }
    for i in p2 as usize..c2.pts.len() {
        imod_point_append(&mut nc, c2.pts[i]);
    }
    if !c1.sizes.is_empty() {
        for i in 0..=p1 as usize {
            imod_point_set_size(&mut nc, i as i32, c1.sizes[i]);
        }
    }
    if !c2.sizes.is_empty() {
        for i in p2 as usize..c2.pts.len() {
            imod_point_set_size(&mut nc, p1 + 1 + i as i32 - p2, c2.sizes[i]);
        }
    }

    istore_clean_ends(&mut nstore);
    nc.store = nstore;
    Some(nc)
}

/// Original: `imodContourScanAdd` (`icont.c:1551`).
///
/// Adds together two scan contours `c1` and `c2`, returning all of their
/// segments in order and combining segments where they overlap.
///
/// Deviation note: the source `malloc`s the point array and the segment-merge
/// branch sets only X and Y, so a merged point's Z is uninitialised heap
/// there; a zeroed array is used here (CLAUDE.md, "Uninitialised memory").
pub fn imod_contour_scan_add(c1: Option<&Icont>, c2: Option<&Icont>) -> Option<Icont> {
    let (Some(c1), Some(c2)) = (c1, c2) else {
        return None;
    };
    if (c1.flags & ICONT_SCANLINE) == 0
        || (c2.flags & ICONT_SCANLINE) == 0
        || c1.pts.is_empty()
        || c2.pts.is_empty()
    {
        return None;
    }
    let mut cont = imod_contour_new()?;
    let mut psize = 0usize;
    cont.pts = vec![Ipoint::default(); c1.pts.len() + c2.pts.len()];
    let mut pt1 = 0i32;
    let mut pt2 = 0i32;
    while pt1 < c1.pts.len() as i32 - 1 || pt2 < c2.pts.len() as i32 - 1 {
        let mut y1 = i32::MAX;
        let mut y2 = i32::MAX;
        if pt1 < c1.pts.len() as i32 - 1 {
            y1 = c1.pts[pt1 as usize].y as i32;
        }
        if pt2 < c2.pts.len() as i32 - 1 {
            y2 = c2.pts[pt2 as usize].y as i32;
        }
        /* Compare the current pair of segments to see which one to add next */
        if y1 < y2 || (y1 == y2 && c1.pts[(pt1 + 1) as usize].x < c2.pts[pt2 as usize].x) {
            cont.pts[psize] = c1.pts[pt1 as usize];
            psize += 1;
            pt1 += 1;
            cont.pts[psize] = c1.pts[pt1 as usize];
            psize += 1;
            pt1 += 1;
        } else if y2 < y1 || (y1 == y2 && c2.pts[(pt2 + 1) as usize].x < c1.pts[pt1 as usize].x) {
            cont.pts[psize] = c2.pts[pt2 as usize];
            psize += 1;
            pt2 += 1;
            cont.pts[psize] = c2.pts[pt2 as usize];
            psize += 1;
            pt2 += 1;
        } else {
            /* Combine the two segments */
            cont.pts[psize].x = if c1.pts[pt1 as usize].x < c2.pts[pt2 as usize].x {
                c1.pts[pt1 as usize].x
            } else {
                c2.pts[pt2 as usize].x
            };
            cont.pts[psize].y = y1 as f32;
            psize += 1;
            cont.pts[psize].x = if c1.pts[(pt1 + 1) as usize].x > c2.pts[(pt2 + 1) as usize].x {
                c1.pts[(pt1 + 1) as usize].x
            } else {
                c2.pts[(pt2 + 1) as usize].x
            };
            cont.pts[psize].y = y1 as f32;
            psize += 1;
            pt1 += 2;
            pt2 += 2;
        }
    }
    cont.pts.truncate(psize);
    cont.flags |= ICONT_SCANLINE;
    Some(cont)
}

/// Original: `imodContourBreak` (`icont.c:1602`).
///
/// Returns a contour containing points from `p1` to `p2`, inclusive, from
/// contour `cont`, and removes those points from `cont`.  If `p2` is < 0 all
/// points from `p1` to the end are transferred to the new contour.
pub fn imod_contour_break(cont: &mut Icont, p1: i32, mut p2: i32) -> Option<Icont> {
    /* check for bogus input data. */
    if p2 < 0 {
        p2 = cont.pts.len() as i32 - 1;
    }

    if cont.pts.is_empty()
        || p1 < 0
        || p1 >= cont.pts.len() as i32
        || p2 >= cont.pts.len() as i32
        || p2 < p1
    {
        return None;
    }

    /* Get contour and points, copy properties */
    let mut nc = imod_contour_new()?;
    nc.flags = cont.flags;
    nc.time = cont.time;
    nc.surf = cont.surf;
    nc.pts = vec![Ipoint::default(); (p2 + 1 - p1) as usize];

    /* Copy points then sizes */
    for i in p1..=p2 {
        nc.pts[(i - p1) as usize] = cont.pts[i as usize];
    }

    if !cont.sizes.is_empty() {
        nc.sizes = vec![0.0f32; nc.pts.len()];
        for i in p1..=p2 {
            nc.sizes[(i - p1) as usize] = cont.sizes[i as usize];
        }
    }

    if istore_break_contour(cont, &mut nc, p1, p2) != 0 {
        return None;
    }

    /* Copy down remaining points and reset size of old contour */
    let mut ni = p1 as usize;
    for i in (p2 + 1) as usize..cont.pts.len() {
        cont.pts[ni] = cont.pts[i];
        if !cont.sizes.is_empty() {
            cont.sizes[ni] = cont.sizes[i];
        }
        ni += 1;
    }
    cont.pts.truncate(ni);
    if !cont.sizes.is_empty() {
        cont.sizes.truncate(ni);
    }

    if ni == 0 {
        cont.pts.clear();
        cont.sizes.clear();
    }

    /* Do not resize arrays since we are now committed to error-free return */

    Some(nc)
}

/// Original: `imodel_contour_sortx` (`icont.c:1672`).
///
/// Sorts points in contour `cont` from `bgnpt` through `endpt` by X.
pub fn imodel_contour_sortx(cont: &mut Icont, bgnpt: i32, endpt: i32) -> i32 {
    if bgnpt < 0 {
        return -1;
    }
    if endpt > cont.pts.len() as i32 - 1 {
        return -1;
    }

    for i in bgnpt..=endpt - 1 {
        let mut sindex = i;
        for j in i + 1..=endpt {
            if cont.pts[sindex as usize].x > cont.pts[j as usize].x {
                sindex = j;
            }
        }
        cont.pts.swap(i as usize, sindex as usize);
        if !cont.sizes.is_empty() {
            cont.sizes.swap(i as usize, sindex as usize);
        }
    }
    0
}

/// Original: `imodel_contour_sorty` (`icont.c:1715`).
pub fn imodel_contour_sorty(cont: &mut Icont, bgnpt: i32, endpt: i32) -> i32 {
    if bgnpt < 0 {
        return -1;
    }
    if endpt > cont.pts.len() as i32 - 1 {
        return -1;
    }
    for i in bgnpt..=endpt - 1 {
        let mut sindex = i;
        for j in i + 1..=endpt {
            if cont.pts[sindex as usize].y > cont.pts[j as usize].y {
                sindex = j;
            }
        }
        cont.pts.swap(i as usize, sindex as usize);
        if !cont.sizes.is_empty() {
            cont.sizes.swap(i as usize, sindex as usize);
        }
    }
    0
}

/// Original: `imodel_contour_sortz` (`icont.c:1758`).
pub fn imodel_contour_sortz(cont: &mut Icont, bgnpt: i32, endpt: i32) -> i32 {
    if bgnpt < 0 {
        return -1;
    }
    if endpt > cont.pts.len() as i32 - 1 {
        return -1;
    }
    for i in bgnpt..=endpt - 1 {
        let mut sindex = i;
        for j in i + 1..=endpt {
            if cont.pts[sindex as usize].z > cont.pts[j as usize].z {
                sindex = j;
            }
        }
        cont.pts.swap(i as usize, sindex as usize);
        if !cont.sizes.is_empty() {
            cont.sizes.swap(i as usize, sindex as usize);
        }
    }
    0
}

/// Original: `imodContourSort3D` (`icont.c:1804`).
///
/// Sorts points in contour `cont` by proximity in 3D with coordinates scaled
/// by `scale`.  Returns -1 for error.
pub fn imod_contour_sort3d(cont: Option<&mut Icont>, scale: &Ipoint) -> i32 {
    let Some(cont) = cont else {
        return -1;
    };

    imod_contour_unique(cont);

    for _pass in 0..2 {
        let mut i = 0i32;
        while i < cont.pts.len() as i32 - 1 {
            /* For each point, find the closest point farther along in the
            contour */
            let mut sindex = i + 1;
            let mut sdist = imod_point3d_scale_distance(
                &cont.pts[i as usize],
                &cont.pts[(i + 1) as usize],
                scale,
            ) as f64;
            for j in (i + 2)..cont.pts.len() as i32 {
                let distance = imod_point3d_scale_distance(
                    &cont.pts[i as usize],
                    &cont.pts[j as usize],
                    scale,
                ) as f64;
                if sdist > distance {
                    sdist = distance;
                    sindex = j;
                }
            }

            /* Swap that point in for the next point */
            let point = cont.pts[(i + 1) as usize];
            cont.pts[(i + 1) as usize] = cont.pts[sindex as usize];
            cont.pts[sindex as usize] = point;
            if !cont.sizes.is_empty() {
                let size = cont.sizes[(i + 1) as usize];
                cont.sizes[(i + 1) as usize] = cont.sizes[sindex as usize];
                cont.sizes[sindex as usize] = size;
            }
            i += 1;
        }

        /* Invert the contour after each pass */
        imodel_contour_invert(cont);
    }
    0
}

/// Original: `imodel_contour_sort` (`icont.c:1856`).
///
/// Sorts points in contour `cont` by proximity in the X/Y plane, by calling
/// `imodContourSort3D` with a scale of 1,1,0.
pub fn imodel_contour_sort(cont: Option<&mut Icont>) -> i32 {
    let scale = Ipoint {
        x: 1.,
        y: 1.,
        z: 0.,
    };
    imod_contour_sort3d(cont, &scale)
}

/// Original: `imodel_contour_invert` (`icont.c:1866`).
///
/// Inverts the order of points in contour `cont`.
pub fn imodel_contour_invert(cont: &mut Icont) -> i32 {
    if cont.pts.is_empty() {
        return -1;
    }

    let pmo = cont.pts.len() - 1;
    for i in 0..(cont.pts.len() + 1) / 2 {
        cont.pts.swap(i, pmo - i);
        if !cont.sizes.is_empty() {
            cont.sizes.swap(i, pmo - i);
        }
    }

    /* Invert the storage items */
    let psize = cont.pts.len() as i32;
    istore_invert(&mut cont.store, psize);
    0
}

/// Original: `imodContourReduce` (`icont.c:1908`).
///
/// Reduces the points in contour `cont` by selecting a minimal subset of the
/// points.  Each of the original points will be within a tolerance `tol` of
/// the segments defined by the remaining points.
pub fn imod_contour_reduce(cont: Option<&mut Icont>, tol: f32) {
    let Some(cont) = cont else {
        return;
    };
    if cont.pts.len() < 3 {
        return;
    }

    /*      moving from right to left, look at possible segments from a given
    point going to right.  A segment is possible if all intervening
    points are within TOL of the segment.  From the given point, find
    the possible segment that involves the fewest segments to get to
    the right end.  Keep track of the endpoint of the best segment in
    NEXTPT and the total number of segments in MINSEG  */

    let npts = cont.pts.len() as i32;
    let mut minseg: Vec<i32> = vec![0; npts as usize];
    let mut nextpt: Vec<i32> = vec![0; npts as usize];
    minseg[(npts - 1) as usize] = 0;
    minseg[(npts - 2) as usize] = 1;
    nextpt[(npts - 1) as usize] = npts;
    nextpt[(npts - 2) as usize] = npts - 1;
    let tolsq = tol * tol;

    /*      Stop looking at longer segments after finding NOUTLIM segments that
    are not possible */

    let noutlim = 3;
    let mut left = npts - 3;
    while left >= 0 {
        /*      set left edge of segment */

        minseg[left as usize] = minseg[(left + 1) as usize] + 1;
        nextpt[left as usize] = left + 1;
        let mut irt = left + 2;
        let x1 = cont.pts[left as usize].x;
        let y1 = cont.pts[left as usize].y;
        let mut nout = 0;
        while irt < npts && nout < noutlim {
            if 1 + minseg[irt as usize] < minseg[left as usize] {
                /*    look at a segment only if it will give a better path: set
                right edge */

                let mut ifout = 0;
                let x2 = cont.pts[irt as usize].x;
                let y2 = cont.pts[irt as usize].y;
                let dx = x2 - x1;
                let dy = y2 - y1;
                let denom = dx * dx + dy * dy;
                let mut its = left + 1;
                while its < irt && ifout == 0 {
                    /*     check distance of points from segment until one falls out */

                    let dx0 = cont.pts[its as usize].x - x1;
                    let dy0 = cont.pts[its as usize].y - y1;

                    let distsq;
                    if denom < 0.00001 {
                        distsq = dx0 * dx0 + dy0 * dy0;
                    } else {
                        let tmin = (dx0 * dx + dy0 * dy) / denom;
                        distsq = (tmin * dx - dx0) * (tmin * dx - dx0)
                            + (tmin * dy - dy0) * (tmin * dy - dy0);
                    }
                    if distsq >= tolsq {
                        ifout = 1;
                    }
                    if istore_retain_point(&cont.store, its) != 0 {
                        ifout = 1;
                    }
                    its += 1;
                }

                /* if its an OK segment, set it up as new minimum */

                if ifout == 0 {
                    minseg[left as usize] = 1 + minseg[irt as usize];
                    nextpt[left as usize] = irt;
                } else {
                    nout += 1;
                }
            }
            irt += 1;
        }
        left -= 1;
    }

    /*      when we get to the left edge, the minimal path is available by
    following the chain of NEXTPT values */
    if minseg[0] + 1 != npts {
        let mut ipo = 0i32;
        if !cont.store.is_empty() {
            /* If there is a store it must be maintained by deleting points
            npo is number of points already retained and index of points to
            delete; ipo is the index of a retained point in original contour */
            let mut npo = 1i32;
            while nextpt[ipo as usize] < npts {
                /* Delete any points between current and next point */
                let mut its = ipo + 1;
                while its < nextpt[ipo as usize] {
                    imod_point_delete(cont, npo);
                    its += 1;
                }
                ipo = nextpt[ipo as usize];
                npo += 1;
            }
        } else {
            /* Otherwise just repack the retained points into the contour and
            adjust size */
            let mut npo = 0i32;
            while nextpt[ipo as usize] < npts {
                ipo = nextpt[ipo as usize];
                npo += 1;
                cont.pts[npo as usize] = cont.pts[ipo as usize];
            }
            cont.pts.truncate((npo + 1) as usize);
            if !cont.sizes.is_empty() {
                cont.sizes.truncate((npo + 1) as usize);
            }
        }
    }
}

/// Original: `imodContourShave` (`icont.c:2041`).
///
/// Removes points from contour `cont` whose distance from both the previous
/// and the next point is less than `dist`.  Returns -1 for error.
pub fn imod_contour_shave(cont: &mut Icont, dist: f64) -> i32 {
    if cont.pts.len() < 3 {
        return -1;
    }

    let mut i = 1i32;
    while i < cont.pts.len() as i32 - 1 {
        let pdist = imodel_point_dist(&cont.pts[(i - 1) as usize], &cont.pts[i as usize]);
        let ndist = imodel_point_dist(&cont.pts[(i + 1) as usize], &cont.pts[i as usize]);
        if (pdist < dist) && (ndist < dist) && istore_retain_point(&cont.store, i) == 0 {
            imod_point_delete(cont, i);
            i -= 1;
        }

        /* DNM 7/4/05: removed code for negative dist that deleted point if it
        was farther than -dist from both points! */
        i += 1;
    }
    0
}

/// Original: `imodContourUnique` (`icont.c:2068`).
///
/// Removes adjacent duplicate points from contour `cont`.
pub fn imod_contour_unique(cont: &mut Icont) -> i32 {
    let mut i = 0usize;
    while i < cont.pts.len() && cont.pts.len() > 1 {
        let j = (i + 1) % cont.pts.len();
        if cont.pts[i].x == cont.pts[j].x
            && cont.pts[i].y == cont.pts[j].y
            && cont.pts[i].z == cont.pts[j].z
        {
            imod_point_delete(cont, j as i32);
        } else {
            i += 1;
        }
    }
    0
}

/// Original: `imodContourStrip` (`icont.c:2095`).
///
/// Removes points that are not needed in contour `cont`; specifically, if
/// three sequential points are colinear it removes the middle one.
pub fn imod_contour_strip(cont: &mut Icont) -> i32 {
    let mut pt = 0i32;
    while pt < cont.pts.len() as i32 - 2 {
        let npt = pt + 1;
        let nnpt = pt + 2;
        let mut is = 0;
        let mut nis = 0;
        if cont.pts[npt as usize].x == cont.pts[pt as usize].x {
            is = 1;
        }
        if cont.pts[npt as usize].x == cont.pts[nnpt as usize].x {
            nis = 1;
        }
        if is != 0 && nis != 0 {
            imod_point_delete(cont, npt);
            pt -= 1;
            pt += 1;
            continue;
        }
        if is != 0 {
            pt += 1;
            continue;
        }
        if nis != 0 {
            pt += 1;
            continue;
        }

        let slope = ((cont.pts[npt as usize].y - cont.pts[pt as usize].y)
            / (cont.pts[npt as usize].x - cont.pts[pt as usize].x)) as f64;
        let nslope = ((cont.pts[nnpt as usize].y - cont.pts[npt as usize].y)
            / (cont.pts[nnpt as usize].x - cont.pts[npt as usize].x)) as f64;

        if slope == nslope {
            imod_point_delete(cont, npt);
            pt -= 1;
        }
        pt += 1;
    }
    0
}

/// Original: `imodel_contour_whole` (`icont.c:2133`).
///
/// Rounds off X/Y point coordinates of contour `cont` to the nearest integer.
pub fn imodel_contour_whole(cont: &mut Icont) {
    for i in 0..cont.pts.len() {
        let x = (cont.pts[i].x as f64 + 0.5).floor() as i32;
        let y = (cont.pts[i].y as f64 + 0.5).floor() as i32;
        cont.pts[i].x = x as f32;
        cont.pts[i].y = y as f32;
    }
}

/// Original: `imodContourFlatten` (`icont.c:2150`).
///
/// Sets the Z values of all points to the mean Z value rounded to nearest int.
pub fn imod_contour_flatten(cont: &mut Icont) {
    let cz = imod_contour_z_value(Some(cont));
    for i in 0..cont.pts.len() {
        cont.pts[i].z = cz as f32;
    }
}

/// Original: `imodel_contour_double` (`icont.c:2160`).
///
/// Doubles number of points in contour by interpolation.  Unused 4/22/05.
pub fn imodel_contour_double(cont: &Icont) -> Option<Icont> {
    let mut fcont = imod_contour_new()?;
    let mut point = Ipoint::default();
    let mut index = 0i32;

    let mut pt = 0i32;
    while pt < cont.pts.len() as i32 - 1 {
        imod_point_add(&mut fcont, Some(cont.pts[pt as usize]), index);
        index += 1;
        point.x = (cont.pts[pt as usize].x + cont.pts[(pt + 1) as usize].x) / 2.;
        point.y = (cont.pts[pt as usize].y + cont.pts[(pt + 1) as usize].y) / 2.;
        point.z = (cont.pts[pt as usize].z + cont.pts[(pt + 1) as usize].z) / 2.;
        imod_point_add(&mut fcont, Some(point), index);
        index += 1;
        pt += 1;
    }
    let last = cont.pts.len() - 1;
    imod_point_add(&mut fcont, Some(cont.pts[last]), index);
    index += 1;
    point.x = (cont.pts[0].x + cont.pts[last].x) / 2.;
    point.y = (cont.pts[0].y + cont.pts[last].y) / 2.;
    point.z = (cont.pts[0].z + cont.pts[last].z) / 2.;
    imod_point_add(&mut fcont, Some(point), index);
    Some(fcont)
}

/// Original: `imodContourScale` (`icont.c:2188`).
pub fn imod_contour_scale(cont: &mut Icont, spoint: &Ipoint) {
    for pt in 0..cont.pts.len() {
        cont.pts[pt].x *= spoint.x;
        cont.pts[pt].y *= spoint.y;
        cont.pts[pt].z *= spoint.z;
    }
}

/// Original: `imodContourRotateZ` (`icont.c:2207`).
///
/// Rotates contour `cont` in the X/Y plane about the origin by `rot`.  Note
/// that `imodMatRot` treats its angle as degrees, as in the source.
pub fn imod_contour_rotate_z(cont: &mut Icont, rot: f64) {
    let Some(mut mat) = imod_mat_new(2) else {
        return;
    };
    let mut rpt = Ipoint::default();

    imod_mat_rot(&mut mat, rot, B3D_Z);
    for pt in 0..cont.pts.len() {
        imod_mat_transform2d(&mat, &cont.pts[pt], &mut rpt);
        cont.pts[pt] = rpt;
    }

    imod_mat_delete(&mut mat);
}

/// Original: `imodel_contour_swapxy` (`icont.c:2224`).
pub fn imodel_contour_swapxy(cont: &mut Icont) {
    for pt in 0..cont.pts.len() {
        let tmp = cont.pts[pt].x;
        cont.pts[pt].x = cont.pts[pt].y;
        cont.pts[pt].y = tmp;
    }
}

/// Original: `imodContourMakeDirection` (`icont.c:2243`).
pub fn imod_contour_make_direction(cont: &mut Icont, direction: i32) {
    if cont.pts.len() < 3 {
        return;
    }
    if direction != imod_cont_z_direction(Some(cont)) {
        imodel_contour_invert(cont);
    }
}

/// Original: `imodContourFill` (`icont.c:2255`).
///
/// Add points so that all points are about 1 to sqrt(2) pixels apart.  Used
/// only by the broken principal axis routine.
pub fn imod_contour_fill(cont: Option<&Icont>) -> Option<Icont> {
    let cont = cont?;

    let mut fcont = imod_contour_new()?;

    if cont.pts.len() < 2 {
        return None;
    }

    let mut point = Ipoint::default();
    point.z = cont.pts[0].z;
    for pt in 0..cont.pts.len() {
        let mut npt = pt + 1;
        if npt == cont.pts.len() {
            if cont.flags & ICONT_OPEN != 0 {
                break;
            }
            npt = 0;
        }
        let dx = cont.pts[npt].x - cont.pts[pt].x;
        let dy = cont.pts[npt].y - cont.pts[pt].y;
        let dz = cont.pts[npt].z - cont.pts[pt].z;

        let fdist = (((dx * dx) as f64) + ((dy * dy) as f64) + ((dz * dz) as f64)).sqrt() as f32;
        let dist = (fdist + 0.5f32) as i32;

        if dist == 0 {
            continue;
        }
        let xstep = dx / dist as f32;
        let ystep = dy / dist as f32;
        let zstep = dz / dist as f32;
        point = cont.pts[pt];

        for _i in 0..dist {
            imod_point_append(&mut fcont, point);
            point.x += xstep;
            point.y += ystep;
            point.z += zstep;
        }
    }
    Some(fcont)
}

/// Original: `imodel_contour_scan` (`icont.c:2332`).
///
/// Creates a scan contour from contour `incont`.  A scan contour consists of
/// pairs of points at the starts and ends of horizontal lines, in order by
/// increasing Y and by increasing X at each Y level.
pub fn imodel_contour_scan(incont: Option<&Icont>) -> Option<Icont> {
    let chunksize = 80usize;

    let incont = incont?;
    if incont.pts.is_empty() {
        return None;
    }

    /* DNM: move the duplication and unique calls to before the test for
    one point, mark it as scan type if only one point */
    let mut ocont = imod_contour_dup(incont)?;
    imodel_contour_whole(&mut ocont);
    imod_contour_unique(&mut ocont);

    if ocont.pts.len() == 1 {
        let mut cont = imod_contour_new()?;
        let p = ocont.pts[0];
        imod_point_append(&mut cont, p);
        imod_point_append(&mut cont, p);
        cont.pts[1].x += 1.0f32;
        cont.flags |= ICONT_SCANLINE;
        return Some(cont);
    }

    /* DNM: eliminate getting rid of lines parallel to y scan */
    let mut pmin = Ipoint::default();
    let mut pmax = Ipoint::default();
    imod_contour_get_bbox(Some(&ocont), &mut pmin, &mut pmax);
    let ymin = pmin.y as i32;
    let ymax = pmax.y as i32;

    /* DNM: but if there is just one line parallel to y, then make the scan
    contour from starting and ending points */
    if ymin == ymax {
        let last = ocont.pts.len() as i32 - 1;
        imodel_contour_sortx(&mut ocont, 0, last);
        let mut cont = imod_contour_new()?;
        let first = ocont.pts[0];
        let lastpt = ocont.pts[ocont.pts.len() - 1];
        imod_point_append(&mut cont, first);
        imod_point_append(&mut cont, lastpt);
        cont.flags |= ICONT_SCANLINE;
        cont.pts[0].z = incont.pts[0].z;
        return Some(cont);
    }

    /* active edge table */
    let mut aet = imod_contour_new()?;

    /* contour to be returned */
    let mut cont = imod_contour_new()?;
    cont.pts.reserve(chunksize);
    cont.surf = (pmin.x - 1.) as i32;
    cont.flags |= ICONT_SCANLINE;

    /* DNM: here and below, take ymin into account to allow negative y's */
    /* DNM: Make et an array of points; allocate aet point space also, and an
    array to keep track of y values of edges */
    let mut et: Vec<Ipoint> = vec![Ipoint::default(); ocont.pts.len()];
    let mut yvals: Vec<i32> = vec![0; ocont.pts.len()];

    /* DNM: get arrays for number of edges at each y, and pointers to et */
    let mut numaty: Vec<i32> = vec![0; (ymax - ymin + 2) as usize];
    let mut ptsaty: Vec<usize> = vec![0; (ymax - ymin + 2) as usize];

    /****************************************/
    /* Fill edge table                      */
    /* Convert contour to edge format       */
    /* Array of contour from ymin to ymax.  */

    /* `point` is declared once for the whole routine in the source, and the
    scan-conversion loop below sets only its X and Y: every emitted scan point
    therefore carries the Z (edge slope) left by the last edge processed here.
    That carry-over is observable in the output, so keep one variable. */
    let mut point = Ipoint::default();

    /* first initialize whether last pair of points were rising */
    let mut was_rising = 0i32;
    let mut ppt = ocont.pts.len() as i32 - 1;
    while ppt > 0 {
        if ocont.pts[ppt as usize].y != ocont.pts[0].y {
            was_rising = (ocont.pts[ppt as usize].y < ocont.pts[0].y) as i32;
            break;
        }
        ppt -= 1;
    }

    for pt in 0..ocont.pts.len() {
        /* Get point, next point and previous point */
        let mut npt = pt + 1;
        if npt > ocont.pts.len() - 1 {
            npt = 0;
        }

        if ocont.pts[pt].y == ocont.pts[npt].y {
            continue;
        }

        /* get start and end points of edge */
        let head = Ipoint {
            x: ocont.pts[npt].x,
            y: ocont.pts[npt].y,
            z: 0.,
        };
        let tail = Ipoint {
            x: ocont.pts[pt].x,
            y: ocont.pts[pt].y,
            z: 0.,
        };

        /* point.x is min x                     */
        /* point.y is max y                     */
        /* point.z is 1/m where m is slope.     */
        point.x = ocont.pts[pt].x;
        point.y = ocont.pts[pt].y;
        point.z = (ocont.pts[npt].x - ocont.pts[pt].x) / (ocont.pts[npt].y - ocont.pts[pt].y);

        /* if not local max or min and there are more than 2 points,
        shorten edge by 1 pixel at the tail */
        let rising = (ocont.pts[npt].y > ocont.pts[pt].y) as i32;
        let y: i32;
        if ocont.pts.len() > 2
            && ((rising != 0 && was_rising != 0) || (rising == 0 && was_rising == 0))
        {
            if tail.y > head.y {
                y = head.y as i32;
                point.y = (tail.y as f64 - 1.0) as f32;
                point.x = head.x;
            } else {
                point.y = head.y;
                point.x = tail.x + point.z;
                y = (tail.y as f64 + 1.0) as i32;
            }
        } else {
            if head.y > tail.y {
                point.y = head.y;
                point.x = tail.x;
                y = tail.y as i32;
            } else {
                point.y = tail.y;
                point.x = head.x;
                y = head.y as i32;
            }
        }
        /* Add edge to temporary edge table; keep track of its Y value and
        the number of edges at that y */
        yvals[aet.pts.len()] = y;
        aet.pts.push(point);
        numaty[(y - ymin) as usize] += 1;
        was_rising = rising;
    }

    /* Set up pointers into et array */
    let mut j = 0usize;
    for i in 0..=(ymax - ymin) as usize {
        ptsaty[i] = j;
        j += numaty[i] as usize;
        numaty[i] = 0;
    }

    /* repack edges into et in order */
    for j in 0..aet.pts.len() {
        let i = (yvals[j] - ymin) as usize;
        et[ptsaty[i] + numaty[i] as usize] = aet.pts[j];
        numaty[i] += 1;
    }

    aet.pts.clear();

    /************************************************************/
    /* Scan convert contour:                                    */
    /* aet stores the active edge table.                        */
    /* cont will contain start and stop points along the y scan */
    for i in 0..=(ymax - ymin) as usize {
        point.y = (i as i32 + ymin) as f32;

        /* Fill aet */
        if numaty[i] != 0 {
            for j in 0..numaty[i] as usize {
                aet.pts.push(et[ptsaty[i] + j]);
            }
        }
        let last = aet.pts.len() as i32 - 1;
        imodel_contour_sortx(&mut aet, 0, last);

        /* Fill cont */
        let mut j: i32 = 0;
        while j < aet.pts.len() as i32 {
            let ju = j as usize;
            point.x = aet.pts[ju].x;
            cont.pts.push(point);

            if aet.pts[ju].y <= (i as i32 + ymin) as f32 {
                /* Remove point from aet if it's done */
                aet.pts.remove(ju);
                j -= 1;
            } else {
                /* Or step it for next y */
                let step = aet.pts[ju].z;
                aet.pts[ju].x += step;
            }
            j += 1;
        }
    }

    /* make sure each scanline has an even pair. */
    let mut i: usize = 0;
    while i < cont.pts.len() {
        let bgnpt = i;
        while i < cont.pts.len() - 1 && cont.pts[i].y == cont.pts[i + 1].y {
            i += 1;
        }
        let endpt = i;

        if (endpt - bgnpt) % 2 == 0 {
            let p = cont.pts[i];
            imod_point_add(&mut cont, Some(p), i as i32);
            i += 1;
        }
        i += 1;
    }
    if cont.pts.is_empty() {
        return Some(cont);
    }
    cont.pts[0].z = incont.pts[0].z;

    Some(cont)
}

/// Original: `imodel_contour_overlap` (`icont.c:2621`).
///
/// Returns 1 if contour `c1` overlaps contour `c2` when projected onto the
/// X/Y plane.  Returns 0 if not, or if an error occurs.
pub fn imodel_contour_overlap(c1: &Icont, c2: &Icont) -> i32 {
    let mut pmax1 = Ipoint::default();
    let mut pmin1 = Ipoint::default();
    let mut pmax2 = Ipoint::default();
    let mut pmin2 = Ipoint::default();

    /* first check and see if bounding box overlaps. */
    imod_contour_get_bbox(Some(c1), &mut pmin1, &mut pmax1);
    imod_contour_get_bbox(Some(c2), &mut pmin2, &mut pmax2);

    if pmax1.x < pmin2.x {
        return 0;
    }
    if pmax2.x < pmin1.x {
        return 0;
    }
    if pmax1.y < pmin2.y {
        return 0;
    }
    if pmax2.y < pmin1.y {
        return 0;
    }

    /* then check to see if scanlines overlap */
    let Some(cs1) = imodel_contour_scan(Some(c1)) else {
        return 0;
    };
    let Some(cs2) = imodel_contour_scan(Some(c2)) else {
        return 0;
    };

    let mut jstrt = 0i32;
    let mut i = 0i32;
    while i < cs1.pts.len() as i32 - 1 {
        let mut j = jstrt;
        while j < cs2.pts.len() as i32 - 1 {
            if cs1.pts[i as usize].y == cs2.pts[j as usize].y {
                if (cs1.pts[i as usize].x >= cs2.pts[j as usize].x)
                    && (cs1.pts[i as usize].x <= cs2.pts[(j + 1) as usize].x)
                {
                    return 1;
                }
                if (cs1.pts[(i + 1) as usize].x >= cs2.pts[j as usize].x)
                    && (cs1.pts[(i + 1) as usize].x <= cs2.pts[(j + 1) as usize].x)
                {
                    return 1;
                }
                if (cs2.pts[j as usize].x >= cs1.pts[i as usize].x)
                    && (cs2.pts[j as usize].x <= cs1.pts[(i + 1) as usize].x)
                {
                    return 1;
                }
                if (cs2.pts[j as usize].x >= cs1.pts[i as usize].x)
                    && (cs2.pts[j as usize].x <= cs1.pts[(i + 1) as usize].x)
                {
                    return 1;
                }
            } else if cs1.pts[i as usize].y > cs2.pts[j as usize].y {
                jstrt = j;
            } else {
                break;
            }
            j += 2;
        }
        i += 2;
    }

    /* 3/26/01: memory leak fixed by Lambert Zijp */
    0
}

/// Original: `imodel_scans_overlap` (`icont.c:2694`).
///
/// Returns 1 if scan contour `cs1` overlaps scan contour `cs2`.  Returns 0 if
/// not, or for an error.
pub fn imodel_scans_overlap(
    cs1: Option<&Icont>,
    pmin1: Ipoint,
    pmax1: Ipoint,
    cs2: Option<&Icont>,
    pmin2: Ipoint,
    pmax2: Ipoint,
) -> i32 {
    let Some(cs1) = cs1 else {
        return 0;
    };
    let Some(cs2) = cs2 else {
        return 0;
    };

    /* first check and see if bounding box overlaps. */

    if pmax1.x < pmin2.x {
        return 0;
    }
    if pmax2.x < pmin1.x {
        return 0;
    }
    if pmax1.y < pmin2.y {
        return 0;
    }
    if pmax2.y < pmin1.y {
        return 0;
    }

    /* then check to see if scanlines overlap */

    let mut jstrt = 0i32;
    let mut i = 0i32;
    while i < cs1.pts.len() as i32 - 1 {
        let mut j = jstrt;
        while j < cs2.pts.len() as i32 - 1 {
            if cs1.pts[i as usize].y == cs2.pts[j as usize].y {
                if (cs1.pts[i as usize].x >= cs2.pts[j as usize].x)
                    && (cs1.pts[i as usize].x <= cs2.pts[(j + 1) as usize].x)
                {
                    return 1;
                }
                if (cs1.pts[(i + 1) as usize].x >= cs2.pts[j as usize].x)
                    && (cs1.pts[(i + 1) as usize].x <= cs2.pts[(j + 1) as usize].x)
                {
                    return 1;
                }
                if (cs2.pts[j as usize].x >= cs1.pts[i as usize].x)
                    && (cs2.pts[j as usize].x <= cs1.pts[(i + 1) as usize].x)
                {
                    return 1;
                }
                if (cs2.pts[j as usize].x >= cs1.pts[i as usize].x)
                    && (cs2.pts[j as usize].x <= cs1.pts[(i + 1) as usize].x)
                {
                    return 1;
                }
            } else if cs1.pts[i as usize].y > cs2.pts[j as usize].y {
                jstrt = j;
            } else {
                break;
            }
            j += 2;
        }
        i += 2;
    }
    0
}

/// Original: `imodel_overlap_fractions` (`icont.c:2754`).
///
/// Returns 1 if there is overlap between the contours `cs1p` and `cs2p`, and
/// returns fractions of each contour's area that overlaps in `frac1` and
/// `frac2`.  The bounding boxes of the contours must be provided.
///
/// The source takes `Icont **` so that a contour that is not already a scan
/// contour can be replaced by its scan conversion at the caller; here the
/// contour is replaced in place through the `&mut` reference, which has the
/// same effect for the caller.
pub fn imodel_overlap_fractions(
    cs1p: &mut Icont,
    pmin1: Ipoint,
    pmax1: Ipoint,
    cs2p: &mut Icont,
    pmin2: Ipoint,
    pmax2: Ipoint,
    frac1: &mut f32,
    frac2: &mut f32,
) -> i32 {
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sumover = 0.0f32;
    let mut didoverlap = 0;

    *frac1 = 0.0;
    *frac2 = 0.0;

    /* first check and see if bounding box overlaps. */

    if pmax1.x < pmin2.x {
        return 0;
    }
    if pmax2.x < pmin1.x {
        return 0;
    }
    if pmax1.y < pmin2.y {
        return 0;
    }
    if pmax2.y < pmin1.y {
        return 0;
    }

    /* Make sure contours are scan contours now */
    if (cs1p.flags & ICONT_SCANLINE) == 0 {
        let Some(scan) = imodel_contour_scan(Some(cs1p)) else {
            return 0;
        };
        *cs1p = scan;
    }

    if (cs2p.flags & ICONT_SCANLINE) == 0 {
        let Some(scan) = imodel_contour_scan(Some(cs2p)) else {
            return 0;
        };
        *cs2p = scan;
    }

    let cs1: &Icont = cs1p;
    let cs2: &Icont = cs2p;

    /* Now compute total scan line lengths for each */

    let mut i = 0i32;
    while i < cs1.pts.len() as i32 - 1 {
        sum1 += cs1.pts[(i + 1) as usize].x - cs1.pts[i as usize].x;
        i += 2;
    }

    i = 0;
    while i < cs2.pts.len() as i32 - 1 {
        sum2 += cs2.pts[(i + 1) as usize].x - cs2.pts[i as usize].x;
        i += 2;
    }

    /* then check to see if scanlines overlap */

    let mut jstrt = 0i32;
    i = 0;
    while i < cs1.pts.len() as i32 - 1 {
        let mut j = jstrt;
        while j < cs2.pts.len() as i32 - 1 {
            if cs1.pts[i as usize].y == cs2.pts[j as usize].y {
                /* the overlap zone starts at the max of the two starting
                points and ends at the min of the two ending points */

                let mut ovstart = cs1.pts[i as usize].x;
                if ovstart < cs2.pts[j as usize].x {
                    ovstart = cs2.pts[j as usize].x;
                }

                let mut ovend = cs1.pts[(i + 1) as usize].x;
                if ovend > cs2.pts[(j + 1) as usize].x {
                    ovend = cs2.pts[(j + 1) as usize].x;
                }

                if ovend >= ovstart {
                    sumover += ovend - ovstart;
                    didoverlap = 1;
                }
            } else if cs1.pts[i as usize].y > cs2.pts[j as usize].y {
                jstrt = j;
            } else {
                break;
            }
            j += 2;
        }
        i += 2;
    }

    if sum1 > 0.0 {
        *frac1 = sumover / sum1;
    } else {
        *frac1 = didoverlap as f32;
    }
    if sum2 > 0.0 {
        *frac2 = sumover / sum2;
    } else {
        *frac2 = didoverlap as f32;
    }

    didoverlap
}

/* DNM 4/22/05: removed uncompiled "broken" imodContourTracer and icts_offset
that it called */

/// Original: `imodContourAutoSort` (`icont.c:2856`).
///
/// Used to be used to sort points from auto contouring.
pub fn imod_contour_auto_sort(cont: Option<&mut Icont>) -> i32 {
    let Some(cont) = cont else {
        return -1;
    };

    /* Loop through points in contour. */
    /* DNM 2/12/01: change test from i < cont->psize - 1 to avoid unsigned
    int problems */
    let mut i = 0i32;
    while i + 1 < cont.pts.len() as i32 {
        let mut sindex = i + 1;
        let mut sdist = imodel_point_dist(&cont.pts[i as usize], &cont.pts[(i + 1) as usize]);

        /* Loop through remaining points for best match */
        for j in (i + 2)..cont.pts.len() as i32 {
            let distance = imodel_point_dist(&cont.pts[i as usize], &cont.pts[j as usize]);
            if sdist > distance {
                sdist = distance;
                sindex = j;
            }
        }

        let point = cont.pts[(i + 1) as usize];
        cont.pts[(i + 1) as usize] = cont.pts[sindex as usize];
        cont.pts[sindex as usize] = point;
        if !cont.sizes.is_empty() {
            let size = cont.sizes[(i + 1) as usize];
            cont.sizes[(i + 1) as usize] = cont.sizes[sindex as usize];
            cont.sizes[sindex as usize] = size;
        }
        i += 1;
    }
    0
}

/// Original: `imodContourSwap` (`icont.c:2902`).
///
/// Swaps the values in two contour structures `c1` and `c2` (Unused 4/22/05).
/// The source swaps only the point array, size, flags, time, surface and
/// sizes members; the store list and temporary value are left in place.
pub fn imod_contour_swap(c1: &mut Icont, c2: &mut Icont) {
    let tc_pts = std::mem::take(&mut c1.pts);
    c1.pts = std::mem::take(&mut c2.pts);
    c2.pts = tc_pts;
    let tc_flags = c1.flags;
    c1.flags = c2.flags;
    c2.flags = tc_flags;
    let tc_time = c1.time;
    c1.time = c2.time;
    c2.time = tc_time;
    let tc_surf = c1.surf;
    c1.surf = c2.surf;
    c2.surf = tc_surf;
    let tc_sizes = std::mem::take(&mut c1.sizes);
    c1.sizes = std::mem::take(&mut c2.sizes);
    c2.sizes = tc_sizes;
}

/// Original: `imodContourFindPoint` (`icont.c:2929`).
///
/// Returns index of first point found inside of contour that matches point,
/// or -1 if no match.  Unused 4/22/05.
pub fn imod_contour_find_point(cont: Option<&Icont>, point: Option<&Ipoint>, flag: i32) -> i32 {
    let mut index = -1i32;

    let Some(cont) = cont else {
        return index;
    };
    let Some(point) = point else {
        return index;
    };
    let size = cont.pts.len() as i32;

    match flag {
        ICONT_FIND_NOSORT => {
            for pt in 0..size {
                if (point.x == cont.pts[pt as usize].x)
                    && (point.y == cont.pts[pt as usize].y)
                    && (point.z == cont.pts[pt as usize].z)
                {
                    return pt;
                }
            }
        }

        ICONT_FIND_SORTX => {
            let mut low = 0i32;
            let mut high = size - 1;
            while low <= high {
                index = (low + high) / 2;

                if point.x < cont.pts[index as usize].x {
                    high = index - 1;
                    continue;
                }
                if point.x > cont.pts[index as usize].x {
                    low = index + 1;
                    continue;
                }

                loop {
                    index -= 1;
                    if index < 0 {
                        break;
                    }
                    if cont.pts[index as usize].x != point.x {
                        break;
                    }
                }
                index += 1;
                if index >= cont.pts.len() as i32 {
                    return -1;
                }
                while cont.pts[index as usize].x == point.x {
                    if (cont.pts[index as usize].y == point.y)
                        && (cont.pts[index as usize].z == point.z)
                    {
                        return index;
                    }
                    index += 1;
                    if index >= cont.pts.len() as i32 {
                        return -1;
                    }
                }
                break;
            }
        }

        ICONT_FIND_SORTY => {}

        ICONT_FIND_SORTXY => {}

        _ => return -1,
    }
    index
}

/*
 * Consolidated functions for dealing with contours and nesting in imodmesh,
 * imodinfo
 */

/// Original: `imodContourMakeZTables` (`icont.c:3019`).
///
/// Makes tables for object `obj` of contour z values, number and contours at
/// each z value.  Returns -1 if error.
///
/// The source's `malloc`ed arrays become `Vec`s here and the `int **contatz`
/// array of pointers becomes a `Vec<Vec<i32>>`.
///
/// Deviation note: for an object whose contours are all empty, the source
/// leaves `zmin`/`zmax` at `INT_MAX`/`INT_MIN` and computes array sizes that
/// overflow; the wrapping arithmetic is reproduced but allocation sizes are
/// clamped at zero so the translation cannot allocate a negative length.
pub fn imod_contour_make_z_tables(
    obj: &mut Iobj,
    incz: i32,
    clear_flag: u32,
    contzp: &mut Vec<i32>,
    zlistp: &mut Vec<i32>,
    numatzp: &mut Vec<i32>,
    contatzp: &mut Vec<Vec<i32>>,
    zminp: &mut i32,
    zmaxp: &mut i32,
    zlsizep: &mut i32,
    nummaxp: &mut i32,
) -> i32 {
    let mut zlsize = 0i32;

    /* Find min and max z values.
     * Clear the type value used to store connection information.
     */
    let mut contz: Vec<i32> = vec![0; obj.cont.len()];
    let mut zmin = i32::MAX;
    let mut zmax = i32::MIN;
    for co in 0..obj.cont.len() {
        let cz = imod_contour_z_value(Some(&obj.cont[co]));
        contz[co] = cz;
        obj.cont[co].flags &= !clear_flag;
        if !obj.cont[co].pts.is_empty() {
            if cz < zmin {
                zmin = cz;
            }
            if cz > zmax {
                zmax = cz;
            }
        }
    }

    /* get list of z sections to connect for skip */
    let zlist_len = zmax.wrapping_sub(zmin).wrapping_add(2).max(0) as usize;
    let numatz_len = zmax
        .wrapping_sub(zmin)
        .wrapping_add(2)
        .wrapping_add(incz)
        .max(0) as usize;
    let mut zlist: Vec<i32> = vec![0; zlist_len];
    let mut numatz: Vec<i32> = vec![0; numatz_len];
    let mut contatz: Vec<Vec<i32>> = vec![Vec::new(); numatz_len];
    for z in 1..=incz {
        numatz[(zmax + z - zmin) as usize] = 0;
        contatz[(zmax + z - zmin) as usize] = Vec::new();
    }
    for z in zmin..=zmax {
        numatz[(z - zmin) as usize] = 0;
        contatz[(z - zmin) as usize] = Vec::new();
        for co in 0..obj.cont.len() {
            let cz = contz[co];
            if !obj.cont[co].pts.is_empty() && cz == z {
                zlist[zlsize as usize] = z;
                zlsize += 1;
                break;
            }
        }
    }

    /* Get number of contours at each z and tables of each */
    for co in 0..obj.cont.len() {
        if !obj.cont[co].pts.is_empty() {
            numatz[(contz[co] - zmin) as usize] += 1;
        }
    }

    let mut nummax = 0i32;
    for i in 0..zmax.wrapping_add(1).wrapping_sub(zmin).max(0) as usize {
        contatz[i] = vec![0; (numatz[i] + 1) as usize];
        if numatz[i] > nummax {
            nummax = numatz[i];
        }
        numatz[i] = 0;
    }

    for co in 0..obj.cont.len() {
        if !obj.cont[co].pts.is_empty() {
            let i = (contz[co] - zmin) as usize;
            let n = numatz[i] as usize;
            contatz[i][n] = co as i32;
            numatz[i] += 1;
        }
    }

    *nummaxp = nummax;
    *zminp = zmin;
    *zmaxp = zmax;
    *contzp = std::mem::take(&mut contz);
    *zlistp = std::mem::take(&mut zlist);
    *numatzp = std::mem::take(&mut numatz);
    *contatzp = std::mem::take(&mut contatz);
    *zlsizep = zlsize;
    0
}

/// Original: `imodContourFreeZTables` (`icont.c:3120`).
///
/// Frees the tables of contour Z values, number and contours at Z created by
/// `imodContourMakeZTables`.
pub fn imod_contour_free_z_tables(
    numatz: &mut Vec<i32>,
    contatz: &mut Vec<Vec<i32>>,
    contz: &mut Vec<i32>,
    zlist: &mut Vec<i32>,
    zmin: i32,
    zmax: i32,
) {
    if !zlist.is_empty() && !numatz.is_empty() && !contatz.is_empty() {
        for i in 0..zmax.wrapping_add(1).wrapping_sub(zmin).max(0) as usize {
            if numatz[i] != 0 && !contatz[i].is_empty() {
                contatz[i] = Vec::new();
            }
        }
    }

    numatz.clear();
    contatz.clear();
    contz.clear();
    zlist.clear();
}

/// Original: `imodContourCheckNesting` (`icont.c:3155`).
///
/// Checks for one contour inside another and maintains nesting structures.
/// `co` and `eco` must be different indexes into `scancont`, as in the source.
pub fn imod_contour_check_nesting(
    co: i32,
    eco: i32,
    scancont: &mut [Icont],
    pmin: &[Ipoint],
    pmax: &[Ipoint],
    nests: &mut Vec<Nesting>,
    nestind: &mut [i32],
    numnests: &mut i32,
    numwarn: &mut i32,
) -> i32 {
    let mut frac1 = 0.0f32;
    let mut frac2 = 0.0f32;
    let mut need_warn = 0;

    let (sc1, sc2) = if co < eco {
        let (left, right) = scancont.split_at_mut(eco as usize);
        (&mut left[co as usize], &mut right[0])
    } else {
        let (left, right) = scancont.split_at_mut(co as usize);
        (&mut right[0], &mut left[eco as usize])
    };
    imodel_overlap_fractions(
        sc1,
        pmin[co as usize],
        pmax[co as usize],
        sc2,
        pmin[eco as usize],
        pmax[eco as usize],
        &mut frac1,
        &mut frac2,
    );

    /* Exact duplicates actually print as 0.999999 */
    if frac1 > 0.99998 && frac2 > 0.99998 {
        if *numwarn >= 0 {
            unsafe {
                libc::printf(
                    c"WARNING: Contours %d and %d are duplicates\n".as_ptr(),
                    co + 1,
                    eco + 1,
                );
            }
        }
        need_warn = 1;
    } else if frac1 > 0.99 || frac2 > 0.99 {
        let inco;
        let outco;
        if frac2 > frac1 {
            inco = eco;
            outco = co;
        } else {
            inco = co;
            outco = eco;
        }
        /* add outside one to the inside's lists;
        create them new list if necessary */
        let mut nind = nestind[inco as usize];
        if nind < 0 {
            nests.push(Nesting::default());
            nind = *numnests;
            *numnests += 1;
            nestind[inco as usize] = nind;
            let nest = &mut nests[nind as usize];
            nest.co = inco;
            nest.level = 0;
            nest.ninside = 0;
            nest.noutside = 0;
            nest.inscan = None;
        }
        let nest = &mut nests[nind as usize];

        nest.outside.push(outco);
        nest.noutside += 1;

        /* now add inside one to outside's list */
        let mut nind = nestind[outco as usize];
        if nind < 0 {
            nests.push(Nesting::default());
            nind = *numnests;
            *numnests += 1;
            nestind[outco as usize] = nind;
            let nest = &mut nests[nind as usize];
            nest.co = outco;
            nest.level = 0;
            nest.ninside = 0;
            nest.noutside = 0;
            nest.inscan = None;
        }
        let nest = &mut nests[nind as usize];

        nest.inside.push(inco);
        nest.ninside += 1;
    } else if (frac1 > 0.1 || frac2 > 0.1) && *numwarn >= 0 {
        unsafe {
            libc::printf(
                c"WARNING: Contours %d and %d overlap by %.3f and %.3f\n".as_ptr(),
                co + 1,
                eco + 1,
                frac1 as std::ffi::c_double,
                frac2 as std::ffi::c_double,
            );
        }
        need_warn = 1;
    }
    if need_warn != 0 && *numwarn == 0 {
        unsafe {
            libc::printf(
                c"To find these contours in 3dmod, you may first have to remove empty contours\n with the menu command Edit-Object-Clean\n".as_ptr(),
            );
        }
        *numwarn = 1;
    }
    0
}

/// Original: `imodContourFreeNests` (`icont.c:3261`).
///
/// Frees the array of `numnests` nests in `nests` and their internal arrays.
pub fn imod_contour_free_nests(nests: &mut Vec<Nesting>, numnests: i32) {
    for nind in 0..numnests as usize {
        if nind >= nests.len() {
            break;
        }
        let nest = &mut nests[nind];
        if nest.ninside > 0 && nest.inscan.is_some() {
            if let Some(inscan) = nest.inscan.as_mut() {
                imod_contour_delete(inscan);
            }
            nest.inscan = None;
        }
        if nest.ninside != 0 && !nest.inside.is_empty() {
            nest.inside = Vec::new();
        }
        if nest.noutside != 0 && !nest.outside.is_empty() {
            nest.outside = Vec::new();
        }
    }
    if numnests != 0 && !nests.is_empty() {
        nests.clear();
    }
}

/// Original: `imodContourNestLevels` (`icont.c:3283`).
///
/// Analyzes inside and outside contours to determine level.
pub fn imod_contour_nest_levels(nests: &mut [Nesting], nestind: &[i32], numnests: i32) {
    let mut level = 1;
    let mut more;
    loop {
        more = 0;
        for nind in 0..numnests as usize {
            if nests[nind].level != 0 {
                continue;
            }
            let mut ready = 1;
            /* if the only contours outside have level assigned but lower
            than the current level, then this contour can be assigned to
            the current level */
            for i in 0..nests[nind].noutside as usize {
                let oind = nestind[nests[nind].outside[i] as usize];
                if nests[oind as usize].level == 0 || nests[oind as usize].level >= level {
                    more = 1;
                    ready = 0;
                    break;
                }
            }
            if ready != 0 {
                nests[nind].level = level;
            }
        }
        level += 1;
        if more == 0 {
            break;
        }
    }
}

/****************************************************************************/
/* ACCESSOR FUNCTIONS                                                       */

/// Original: `imodContourGetMaxPoint` (`icont.c:3338`).
pub fn imod_contour_get_max_point(in_contour: Option<&Icont>) -> i32 {
    match in_contour {
        None => 0,
        Some(c) => c.pts.len() as i32,
    }
}

/// Original: `imodContourGetPoints` (`icont.c:3345`).
pub fn imod_contour_get_points(in_contour: Option<&Icont>) -> Option<&[Ipoint]> {
    let c = in_contour?;
    Some(&c.pts)
}

/// Original: `imodContourSetPointData` (`icont.c:3354`).
///
/// Sets the point array of `in_contour` to `in_point` and sets the number of
/// points to `in_max`.  The translated `Icont` keeps the array and its count
/// in one `Vec`, so `in_max` truncates the supplied array.
pub fn imod_contour_set_point_data(
    in_contour: Option<&mut Icont>,
    in_point: Vec<Ipoint>,
    in_max: i32,
) {
    let Some(c) = in_contour else {
        return;
    };
    c.pts = in_point;
    c.pts.truncate(in_max.max(0) as usize);
}

/// Original: `imodContourGetPoint` (`icont.c:3362`).
pub fn imod_contour_get_point(in_contour: Option<&Icont>, in_index: i32) -> Option<&Ipoint> {
    let c = in_contour?;
    if in_index < 0 || in_index >= c.pts.len() as i32 {
        return None;
    }
    Some(&c.pts[in_index as usize])
}

/// Original: `imodContourGetTimeIndex` (`icont.c:3372`).
pub fn imod_contour_get_time_index(in_contour: Option<&Icont>) -> i32 {
    match in_contour {
        None => 0,
        Some(c) => c.time,
    }
}

/// Original: `imodContourSetTimeIndex` (`icont.c:3379`).
pub fn imod_contour_set_time_index(in_contour: Option<&mut Icont>, in_time: i32) {
    if let Some(c) = in_contour {
        c.time = in_time;
    }
}

/// Original: `imodContourGetSurface` (`icont.c:3386`).
pub fn imod_contour_get_surface(in_contour: Option<&Icont>) -> i32 {
    match in_contour {
        None => 0,
        Some(c) => c.surf,
    }
}

/// Original: `imodContourSetSurface` (`icont.c:3393`).
pub fn imod_contour_set_surface(in_contour: Option<&mut Icont>, in_surface: i32) {
    if let Some(c) = in_contour {
        c.surf = in_surface;
    }
}

/// Original: `imodContourSetFlag` (`icont.c:3401`).
pub fn imod_contour_set_flag(in_contour: &mut Icont, in_flag: u32, in_state: i32) {
    if in_state != 0 {
        in_contour.flags |= in_flag;
    } else {
        in_contour.flags &= !in_flag;
    }
}

/// Original: `imodContourGetFlag` (`icont.c:3411`).
pub fn imod_contour_get_flag(in_contour: &Icont, in_flag: u32) -> u32 {
    in_contour.flags & in_flag
}

/// Original: `imodContourPointIsGap` (`icont.c:3430`).
pub fn imod_contour_point_is_gap(in_contour: &Icont, index: i32) -> i32 {
    if !in_contour.store.is_empty() {
        return istore_point_is_gap(&in_contour.store, index);
    }
    if in_contour.flags & ICONT_OPEN != 0 && index == in_contour.pts.len() as i32 - 1 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(size: f32) -> Icont {
        let mut cont = imod_contour_new().unwrap();
        cont.pts = vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 3.,
            },
            Ipoint {
                x: size,
                y: 0.,
                z: 3.,
            },
            Ipoint {
                x: size,
                y: size,
                z: 3.,
            },
            Ipoint {
                x: 0.,
                y: size,
                z: 3.,
            },
        ];
        cont
    }

    /// `imodContourArea` (`icont.c:324`) is half the magnitude of the summed
    /// cross products, so a 10x10 square in one Z plane has area 100.
    #[test]
    fn contour_area_matches_cross_product_sum() {
        assert_eq!(imod_contour_area(Some(&square(10.))), 100.);
        assert_eq!(imod_contour_area(None), 0.);
        let mut two = imod_contour_new().unwrap();
        two.pts = vec![Ipoint::default(), Ipoint::default()];
        assert_eq!(imod_contour_area(Some(&two)), 0.);
    }

    /// `imodContourGetBBox` (`icont.c:1024`) covers all three axes.
    #[test]
    fn bbox_covers_all_three_axes() {
        let mut cont = square(10.);
        cont.pts[2].z = -4.;
        let mut ll = Ipoint::default();
        let mut ur = Ipoint::default();
        assert_eq!(imod_contour_get_bbox(Some(&cont), &mut ll, &mut ur), 0);
        assert_eq!(
            (ll.x, ll.y, ll.z, ur.x, ur.y, ur.z),
            (0., 0., -4., 10., 10., 3.)
        );
        assert_eq!(imod_contour_get_bbox(None, &mut ll, &mut ur), -1);
        let empty = imod_contour_new().unwrap();
        assert_eq!(imod_contour_get_bbox(Some(&empty), &mut ll, &mut ur), -1);
    }

    /// `imodel_contour_scan` (`icont.c:2332`) emits point pairs ordered by Y,
    /// carries the Z of the first input point into the first scan point and
    /// records `pmin.x - 1` in `surf`.
    #[test]
    fn scan_of_square_has_paired_points_per_scan_line() {
        let cont = square(10.);
        let scan = imodel_contour_scan(Some(&cont)).unwrap();
        assert_ne!(scan.flags & ICONT_SCANLINE, 0);
        assert_eq!(scan.surf, -1);
        assert_eq!(scan.pts[0].z, 3.);
        assert_eq!(scan.pts.len() % 2, 0);
        // One pair per scan line from y = 0 through y = 10.
        assert_eq!(scan.pts.len(), 22);
        for pair in scan.pts.chunks(2) {
            assert_eq!(pair[0].y, pair[1].y);
            assert!(pair[0].x <= pair[1].x);
        }
        let ys: Vec<f32> = scan.pts.iter().step_by(2).map(|p| p.y).collect();
        assert_eq!(ys, (0..11).map(|y| y as f32).collect::<Vec<f32>>());
    }

    /// `imodel_contour_area` (`icont.c:357`) sums scan-line lengths, so the
    /// 10x10 square gives 10 pixels on each of 11 scan lines.
    #[test]
    fn scan_area_sums_scan_line_lengths() {
        assert_eq!(imodel_contour_area(Some(&square(10.))), 110);
        assert_eq!(imodel_contour_area(None), 0);
    }

    /// `imodContourCenterOfMass` (`icont.c:515`) puts the square's center of
    /// mass at its geometric center, and keeps Z from the first point.
    #[test]
    fn center_of_mass_of_square_is_its_center() {
        let mut cont = square(10.);
        let mut cmass = Ipoint::default();
        assert_eq!(imod_contour_center_of_mass(Some(&mut cont), &mut cmass), 0);
        assert!((cmass.x - 5.).abs() < 1e-4, "{cmass:?}");
        assert!((cmass.y - 5.).abs() < 1e-4, "{cmass:?}");
        assert!((cmass.z - 3.).abs() < 1e-4, "{cmass:?}");
        // A NULL contour zeroes the result and returns 0.
        cmass = Ipoint {
            x: 9.,
            y: 9.,
            z: 9.,
        };
        assert_eq!(imod_contour_center_of_mass(None, &mut cmass), 0);
        assert_eq!(cmass, Ipoint::default());
    }

    /// `imodel_contour_centroid` (`icont.c:551`) returns the raw sums; a single
    /// point contour short-circuits with weight 1.
    #[test]
    fn centroid_single_point_returns_weight_one() {
        let mut cont = imod_contour_new().unwrap();
        cont.pts = vec![Ipoint {
            x: 4.,
            y: 5.,
            z: 6.,
        }];
        let mut rcp = Ipoint::default();
        let mut rtw = 0.;
        assert_eq!(imodel_contour_centroid(Some(&cont), &mut rcp, &mut rtw), 0);
        assert_eq!((rcp.x, rcp.y, rcp.z, rtw), (4., 5., 6., 1.));
        assert_eq!(
            imodel_contour_centroid(Some(&imod_contour_new().unwrap()), &mut rcp, &mut rtw),
            -1
        );
    }

    /// `imodContourCircularity` (`icont.c:743`) is c^2 / (4 pi a); a square is
    /// close to the documented 1.27.
    #[test]
    fn circularity_of_square_is_near_source_documented_value() {
        let mut cont = square(100.);
        // imodel_contour_length excludes the closing segment, so append the
        // start point to get the full perimeter.
        cont.pts.push(Ipoint {
            x: 0.,
            y: 0.,
            z: 3.,
        });
        let circ = imod_contour_circularity(Some(&cont));
        assert!((circ - 1.27).abs() < 0.05, "{circ}");
        let mut two = imod_contour_new().unwrap();
        two.pts = vec![Ipoint::default(), Ipoint::default()];
        assert_eq!(imod_contour_circularity(Some(&two)), 1000.0);
    }

    /// `imodContourLongAxis` (`icont.c:759`) reports aspect 1 for a square and
    /// the elongation direction for a stretched rectangle.
    #[test]
    fn long_axis_finds_elongation_and_aspect() {
        let mut aspect = 0.;
        let mut longaxis = 0.;
        let angle = imod_contour_long_axis(Some(&square(10.)), 1.0, &mut aspect, &mut longaxis);
        assert!((aspect - 1.).abs() < 1e-3, "{aspect}");
        assert!((longaxis - 10.).abs() < 1e-3, "{longaxis}");

        let mut rect = imod_contour_new().unwrap();
        rect.pts = vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 40.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 40.,
                y: 10.,
                z: 0.,
            },
            Ipoint {
                x: 0.,
                y: 10.,
                z: 0.,
            },
        ];
        let angle = imod_contour_long_axis(Some(&rect), 1.0, &mut aspect, &mut longaxis);
        assert!((aspect - 4.).abs() < 1e-3, "{aspect}");
        assert!((longaxis - 40.).abs() < 1e-3, "{longaxis}");
        assert!(angle.abs() < 1e-6, "{angle}");

        // Two-point contours take the special branch.
        let mut two = imod_contour_new().unwrap();
        two.pts = vec![
            Ipoint::default(),
            Ipoint {
                x: 1.,
                y: 1.,
                z: 0.,
            },
        ];
        let angle = imod_contour_long_axis(Some(&two), 1.0, &mut aspect, &mut longaxis);
        assert_eq!(aspect, 1.0e6);
        assert!((longaxis - 2.0f32.sqrt()).abs() < 1e-6);
        assert!(
            (angle - std::f64::consts::FRAC_PI_4).abs() < 1e-6,
            "{angle}"
        );

        // Errors return 0 and leave aspect/longaxis at their defaults.
        assert_eq!(
            imod_contour_long_axis(None, 1.0, &mut aspect, &mut longaxis),
            0.0
        );
        assert_eq!((aspect, longaxis), (1., 0.));
    }

    /// `imodContourEquivEllipse` (`icont.c:674`) recovers the axes of an
    /// elongated ellipse.
    #[test]
    fn equiv_ellipse_recovers_axes_of_an_ellipse() {
        let mut cont = imod_contour_new().unwrap();
        for i in 0..90 {
            let t = i as f64 * std::f64::consts::TAU / 90.;
            cont.pts.push(Ipoint {
                x: (100. + 40. * t.cos()) as f32,
                y: (100. + 20. * t.sin()) as f32,
                z: 7.,
            });
        }
        let mut center = Ipoint::default();
        let (mut la, mut sa, mut ang) = (0., 0., 0.);
        assert_eq!(
            imod_contour_equiv_ellipse(Some(&cont), &mut center, &mut la, &mut sa, &mut ang),
            0
        );
        assert!((center.x - 100.).abs() < 0.5, "{center:?}");
        assert!((center.y - 100.).abs() < 0.5, "{center:?}");
        assert!((center.z - 7.).abs() < 1e-3, "{center:?}");
        assert!((la - 40.).abs() < 1., "long {la}");
        assert!((sa - 20.).abs() < 1., "short {sa}");
        assert!(ang < 1. || ang > 179., "angle {ang}");
        // Fewer than three points is an error.
        let mut two = imod_contour_new().unwrap();
        two.pts = vec![Ipoint::default(), Ipoint::default()];
        assert_eq!(
            imod_contour_equiv_ellipse(Some(&two), &mut center, &mut la, &mut sa, &mut ang),
            1
        );
    }

    /// `imodel_contour_whole` (`icont.c:2133`) rounds with `floor(v + 0.5)`,
    /// which is nearest-integer for negatives too.
    #[test]
    fn whole_rounds_negatives_with_floor_of_plus_half() {
        let mut cont = imod_contour_new().unwrap();
        cont.pts = vec![
            Ipoint {
                x: -1.5,
                y: -2.4,
                z: 1.7,
            },
            Ipoint {
                x: 2.5,
                y: 2.6,
                z: 1.7,
            },
        ];
        imodel_contour_whole(&mut cont);
        assert_eq!((cont.pts[0].x, cont.pts[0].y), (-1., -2.));
        assert_eq!((cont.pts[1].x, cont.pts[1].y), (3., 3.));
        // Z is untouched.
        assert_eq!(cont.pts[0].z, 1.7);
    }

    /// `imodContourUnique` (`icont.c:2068`) drops adjacent duplicates,
    /// including the wrap-around pair.
    #[test]
    fn unique_drops_adjacent_and_wrapped_duplicates() {
        let mut cont = imod_contour_new().unwrap();
        let a = Ipoint {
            x: 1.,
            y: 1.,
            z: 0.,
        };
        let b = Ipoint {
            x: 2.,
            y: 2.,
            z: 0.,
        };
        cont.pts = vec![a, a, b, b, a];
        imod_contour_unique(&mut cont);
        // The wrap-around duplicate deletes index 0, so `b` ends up first.
        assert_eq!(cont.pts, vec![b, a]);
    }

    /// `imodel_contour_sortx` (`icont.c:1672`) is a selection sort over the
    /// inclusive range and keeps point sizes with their points.
    #[test]
    fn sortx_orders_range_and_carries_sizes() {
        let mut cont = imod_contour_new().unwrap();
        cont.pts = vec![
            Ipoint {
                x: 3.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
        ];
        cont.sizes = vec![30., 10., 20.];
        assert_eq!(imodel_contour_sortx(&mut cont, 0, 2), 0);
        assert_eq!(
            cont.pts.iter().map(|p| p.x).collect::<Vec<f32>>(),
            vec![1., 2., 3.]
        );
        assert_eq!(cont.sizes, vec![10., 20., 30.]);
        assert_eq!(imodel_contour_sortx(&mut cont, -1, 2), -1);
        assert_eq!(imodel_contour_sortx(&mut cont, 0, 3), -1);
    }

    /// `imodContZDirection` (`icont.c:1090`) reports the winding sense.
    #[test]
    fn z_direction_reports_winding() {
        let ccw = square(10.);
        assert_eq!(imod_cont_z_direction(Some(&ccw)), 1);
        let mut cw = square(10.);
        cw.pts.reverse();
        assert_eq!(imod_cont_z_direction(Some(&cw)), -1);
        let mut degenerate = imod_contour_new().unwrap();
        degenerate.pts = vec![Ipoint::default(); 3];
        assert_eq!(imod_cont_z_direction(Some(&degenerate)), 0);
    }

    /// `imodel_contour_check_wild` (`icont.c:297`) uses nearest-integer Z.
    #[test]
    fn check_wild_uses_rounded_z() {
        let mut cont = square(10.);
        cont.pts[1].z = 3.4;
        cont.flags |= ICONT_WILD;
        imodel_contour_check_wild(Some(&mut cont));
        assert_eq!(cont.flags & ICONT_WILD, 0);
        cont.pts[1].z = 4.0;
        imodel_contour_check_wild(Some(&mut cont));
        assert_ne!(cont.flags & ICONT_WILD, 0);
    }

    /// `imodContourZValue`/`imodContourFloatZValue` (`icont.c:1060`, `:1070`).
    #[test]
    fn z_value_is_rounded_mean() {
        let mut cont = imod_contour_new().unwrap();
        cont.pts = vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 2.,
            },
            Ipoint {
                x: 0.,
                y: 0.,
                z: 3.,
            },
        ];
        assert_eq!(imod_contour_float_z_value(Some(&cont)), 2.5);
        assert_eq!(imod_contour_z_value(Some(&cont)), 3);
        assert_eq!(imod_contour_z_value(None), -1);
        assert_eq!(
            imod_contour_float_z_value(Some(&imod_contour_new().unwrap())),
            -1.0e20
        );
    }

    /// Differential harness against a driver compiled directly against the
    /// pinned `IMOD/libimod/icont.c` and linked to the reference build's
    /// `libimod`; the expected text below is that driver's verbatim output
    /// (`%.9g` throughout, so the C `printf` formatting is part of the
    /// contract).
    #[test]
    fn source_c_driver_differential() {
        fn g9(v: f64) -> String {
            let mut buf = [0u8; 64];
            unsafe {
                libc::snprintf(
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len(),
                    c"%.9g".as_ptr(),
                    v,
                );
            }
            let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
            String::from_utf8_lossy(&buf[..end]).into_owned()
        }

        fn mkcont(xy: &[f32], z: f32) -> Icont {
            let mut c = imod_contour_new().unwrap();
            for pair in xy.chunks(2) {
                imod_point_append(
                    &mut c,
                    Ipoint {
                        x: pair[0],
                        y: pair[1],
                        z,
                    },
                );
            }
            c
        }

        fn report(out: &mut String, tag: &str, c: &Icont) {
            out.push_str(&format!(
                "{tag} area {}
",
                g9(imod_contour_area(Some(c)) as f64)
            ));
            out.push_str(&format!(
                "{tag} scanarea {}
",
                imodel_contour_area(Some(c))
            ));
            out.push_str(&format!(
                "{tag} length {} {}
",
                g9(imod_contour_length(Some(c), 0) as f64),
                g9(imod_contour_length(Some(c), 1) as f64)
            ));
            out.push_str(&format!(
                "{tag} mlength {}
",
                g9(imodel_contour_length(Some(c)))
            ));
            let mut ll = Ipoint::default();
            let mut ur = Ipoint::default();
            imod_contour_get_bbox(Some(c), &mut ll, &mut ur);
            out.push_str(&format!(
                "{tag} bbox {} {} {} {} {} {}
",
                g9(ll.x as f64),
                g9(ll.y as f64),
                g9(ll.z as f64),
                g9(ur.x as f64),
                g9(ur.y as f64),
                g9(ur.z as f64)
            ));
            let mut cm = Ipoint::default();
            let mut work = c.clone();
            imod_contour_center_of_mass(Some(&mut work), &mut cm);
            out.push_str(&format!(
                "{tag} cmass {} {} {}
",
                g9(cm.x as f64),
                g9(cm.y as f64),
                g9(cm.z as f64)
            ));
            out.push_str(&format!(
                "{tag} circ {}
",
                g9(imod_contour_circularity(Some(c)))
            ));
            let mut aspect = 0.;
            let mut longaxis = 0.;
            let orient = imod_contour_long_axis(Some(c), 1.0, &mut aspect, &mut longaxis);
            out.push_str(&format!(
                "{tag} longaxis {} {} {}
",
                g9(orient),
                g9(aspect as f64),
                g9(longaxis as f64)
            ));
            out.push_str(&format!(
                "{tag} zdir {} zval {} fzval {}
",
                imod_cont_z_direction(Some(c)),
                imod_contour_z_value(Some(c)),
                g9(imod_contour_float_z_value(Some(c)) as f64)
            ));
            let mut rcp = Ipoint::default();
            let mut rtw = 0.;
            imodel_contour_centroid(Some(c), &mut rcp, &mut rtw);
            out.push_str(&format!(
                "{tag} centroid {} {} {} {}
",
                g9(rcp.x as f64),
                g9(rcp.y as f64),
                g9(rcp.z as f64),
                g9(rtw)
            ));
            out.push_str(&format!(
                "{tag} moment {} {} {}
",
                g9(imod_contour_moment(Some(c), 0, 0)),
                g9(imod_contour_moment(Some(c), 1, 0)),
                g9(imod_contour_moment(Some(c), 1, 1))
            ));
            let mut cen = Ipoint::default();
            let (mut la, mut sa, mut ang) = (0., 0., 0.);
            if imod_contour_equiv_ellipse(Some(c), &mut cen, &mut la, &mut sa, &mut ang) == 0 {
                out.push_str(&format!(
                    "{tag} ellipse {} {} {} {} {}
",
                    g9(cen.x as f64),
                    g9(cen.y as f64),
                    g9(la as f64),
                    g9(sa as f64),
                    g9(ang as f64)
                ));
            } else {
                out.push_str(&format!(
                    "{tag} ellipse ERR
"
                ));
            }
            if let Some(scan) = imodel_contour_scan(Some(c)) {
                out.push_str(&format!(
                    "{tag} scan n={} surf={} flags={}
",
                    scan.pts.len(),
                    scan.surf,
                    scan.flags
                ));
                for (i, p) in scan.pts.iter().enumerate() {
                    out.push_str(&format!(
                        "{tag} scanpt {i} {} {} {}
",
                        g9(p.x as f64),
                        g9(p.y as f64),
                        g9(p.z as f64)
                    ));
                }
            }
        }

        let mut out = String::new();
        report(
            &mut out,
            "sq",
            &mkcont(&[0., 0., 10., 0., 10., 10., 0., 10.], 3.),
        );
        report(
            &mut out,
            "rect",
            &mkcont(&[0., 0., 40., 0., 40., 10., 0., 10.], 0.),
        );
        report(
            &mut out,
            "tri",
            &mkcont(&[0.5, 0.25, 13.7, 2.2, 6.1, 11.9], -2.5),
        );
        let mut ell = imod_contour_new().unwrap();
        for i in 0..90 {
            let t = i as f64 * 2. * 3.14159265358979323846 / 90.;
            imod_point_append(
                &mut ell,
                Ipoint {
                    x: (100. + 40. * t.cos()) as f32,
                    y: (100. + 20. * t.sin()) as f32,
                    z: 7.,
                },
            );
        }
        report(&mut out, "ell", &ell);

        let want = EXPECTED_C_DRIVER_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from icont.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
    }

    /// Verbatim stdout of the driver linked against `IMOD/libimod/icont.c`.
    const EXPECTED_C_DRIVER_OUTPUT: &str = r#"sq area 100
sq scanarea 110
sq length 30 40
sq mlength 30
sq bbox 0 0 3 10 10 3
sq cmass 5 5 3
sq circ 0.651088403
sq longaxis 1.57079637 1 10
sq zdir 1 zval 3 fzval 3
sq centroid 550 550 330 110
sq moment 110 550 2750
sq ellipse ERR
sq scan n=22 surf=-1 flags=131072
sq scanpt 0 0 0 3
sq scanpt 1 10 0 -0
sq scanpt 2 0 1 -0
sq scanpt 3 10 1 -0
sq scanpt 4 0 2 -0
sq scanpt 5 10 2 -0
sq scanpt 6 0 3 -0
sq scanpt 7 10 3 -0
sq scanpt 8 0 4 -0
sq scanpt 9 10 4 -0
sq scanpt 10 0 5 -0
sq scanpt 11 10 5 -0
sq scanpt 12 0 6 -0
sq scanpt 13 10 6 -0
sq scanpt 14 0 7 -0
sq scanpt 15 10 7 -0
sq scanpt 16 0 8 -0
sq scanpt 17 10 8 -0
sq scanpt 18 0 9 -0
sq scanpt 19 10 9 -0
sq scanpt 20 0 10 -0
sq scanpt 21 10 10 -0
rect area 400
rect scanarea 440
rect length 90 100
rect mlength 90
rect bbox 0 0 0 40 10 0
rect cmass 20 5 0
rect circ 1.46494891
rect longaxis -0 4 40
rect zdir 1 zval 0 fzval 0
rect centroid 8800 2200 0 440
rect moment 440 2200 44000
rect ellipse ERR
rect scan n=22 surf=-1 flags=131072
rect scanpt 0 0 0 0
rect scanpt 1 40 0 -0
rect scanpt 2 0 1 -0
rect scanpt 3 40 1 -0
rect scanpt 4 0 2 -0
rect scanpt 5 40 2 -0
rect scanpt 6 0 3 -0
rect scanpt 7 40 3 -0
rect scanpt 8 0 4 -0
rect scanpt 9 40 4 -0
rect scanpt 10 0 5 -0
rect scanpt 11 40 5 -0
rect scanpt 12 0 6 -0
rect scanpt 13 40 6 -0
rect scanpt 14 0 7 -0
rect scanpt 15 40 7 -0
rect scanpt 16 0 8 -0
rect scanpt 17 40 8 -0
rect scanpt 18 0 9 -0
rect scanpt 19 40 9 -0
rect scanpt 20 0 10 -0
rect scanpt 21 40 10 -0
tri area 71.4300003
tri scanarea 73
tri length 25.6660004 38.5920372
tri mlength 25.6659994
tri bbox 0.5 0.25 -2.5 13.6999998 11.8999996 -2.5
tri cmass 6.64968491 4.60312223 -2.5
tri circ 0.718097864
tri longaxis 0.139626344 1.24036551 13.342926
tri zdir 1 zval -2 fzval -2.5
tri centroid 485.559998 336.119995 -182.550003 73.02
tri moment 73 337 2389.5
tri ellipse 6.63767672 4.82364082 5.25540638 4.81640434 22.0731049
tri scan n=26 surf=0 flags=131072
tri scanpt 0 1 0 -2.5
tri scanpt 1 1 0 0.416666657
tri scanpt 2 1.41666663 1 0.416666657
tri scanpt 3 7.5 1 0.416666657
tri scanpt 4 1.83333325 2 0.416666657
tri scanpt 5 14 2 0.416666657
tri scanpt 6 2.25 3 0.416666657
tri scanpt 7 13.1999998 3 0.416666657
tri scanpt 8 2.66666675 4 0.416666657
tri scanpt 9 12.3999996 4 0.416666657
tri scanpt 10 3.08333349 5 0.416666657
tri scanpt 11 11.5999994 5 0.416666657
tri scanpt 12 3.50000024 6 0.416666657
tri scanpt 13 10.7999992 6 0.416666657
tri scanpt 14 3.91666698 7 0.416666657
tri scanpt 15 9.99999905 7 0.416666657
tri scanpt 16 4.33333349 8 0.416666657
tri scanpt 17 9.19999886 8 0.416666657
tri scanpt 18 4.75 9 0.416666657
tri scanpt 19 8.39999866 9 0.416666657
tri scanpt 20 5.16666651 10 0.416666657
tri scanpt 21 7.59999847 10 0.416666657
tri scanpt 22 5.58333302 11 0.416666657
tri scanpt 23 6.79999828 11 0.416666657
tri scanpt 24 5.99999809 12 0.416666657
tri scanpt 25 5.99999952 12 0.416666657
ell area 2511.23096
ell scanarea 2546
ell length 192.331085 193.729614
ell mlength 192.331089
ell bbox 60 80.0121841 7 140 119.987816 7
ell cmass 99.9426575 100 7
ell circ 1.15619402
ell longaxis -0 2.00121927 80
ell zdir 1 zval 7 fzval 7
ell centroid 254454 254600 17822 2546
ell moment 2546 254600 25478403
ell ellipse 99.9220123 99.9986572 39.9535942 20.028389 1.67034304e-05
ell scan n=82 surf=59 flags=131072
ell scanpt 0 93 80 7
ell scanpt 1 107 80 0
ell scanpt 2 85 81 0
ell scanpt 3 110 81 0
ell scanpt 4 82 82 0
ell scanpt 5 118 82 0
ell scanpt 6 78 83 0
ell scanpt 7 120 83 0
ell scanpt 8 75 84 0
ell scanpt 9 125 84 0
ell scanpt 10 73 85 0
ell scanpt 11 127 85 0
ell scanpt 12 71 86 0
ell scanpt 13 129 86 0
ell scanpt 14 69 87 0
ell scanpt 15 131 87 0
ell scanpt 16 68 88 0
ell scanpt 17 132 88 0
ell scanpt 18 66 89 0
ell scanpt 19 134 89 0
ell scanpt 20 65.5 90 0
ell scanpt 21 134.5 90 0
ell scanpt 22 65 91 0
ell scanpt 23 135 91 0
ell scanpt 24 63 92 0
ell scanpt 25 137 92 0
ell scanpt 26 62 93 0
ell scanpt 27 138 93 0
ell scanpt 28 62 94 0
ell scanpt 29 138 94 0
ell scanpt 30 61.5 95 0
ell scanpt 31 138.5 95 0
ell scanpt 32 61 96 0
ell scanpt 33 139 96 0
ell scanpt 34 60 97 0
ell scanpt 35 140 97 0
ell scanpt 36 60 98 0
ell scanpt 37 140 98 0
ell scanpt 38 60 99 0
ell scanpt 39 140 99 0
ell scanpt 40 60 100 0
ell scanpt 41 140 100 0
ell scanpt 42 60 101 0
ell scanpt 43 140 101 0
ell scanpt 44 60 102 0
ell scanpt 45 140 102 0
ell scanpt 46 60 103 0
ell scanpt 47 140 103 0
ell scanpt 48 61 104 0
ell scanpt 49 139 104 0
ell scanpt 50 61.5 105 0
ell scanpt 51 138.5 105 0
ell scanpt 52 62 106 0
ell scanpt 53 138 106 0
ell scanpt 54 62 107 0
ell scanpt 55 138 107 0
ell scanpt 56 63 108 0
ell scanpt 57 137 108 0
ell scanpt 58 65 109 0
ell scanpt 59 135 109 0
ell scanpt 60 65.5 110 0
ell scanpt 61 134.5 110 0
ell scanpt 62 66 111 0
ell scanpt 63 134 111 0
ell scanpt 64 68 112 0
ell scanpt 65 132 112 0
ell scanpt 66 69 113 0
ell scanpt 67 131 113 0
ell scanpt 68 71 114 0
ell scanpt 69 129 114 0
ell scanpt 70 73 115 0
ell scanpt 71 127 115 0
ell scanpt 72 75 116 0
ell scanpt 73 125 116 0
ell scanpt 74 80 117 0
ell scanpt 75 122 117 0
ell scanpt 76 82 118 0
ell scanpt 77 118 118 0
ell scanpt 78 90 119 0
ell scanpt 79 115 119 0
ell scanpt 80 93 120 0
ell scanpt 81 107 120 0
"#;
}

/// Differential harness for the contour joining, splicing, breaking,
/// reduction, overlap, sorting, Z-table and nesting groups of `icont.c`,
/// against a driver compiled directly against the pinned source and linked to
/// the reference build's `libimod`.
#[cfg(test)]
mod source_driver_group2 {
    use super::*;
    use crate::imod::libimod::iobj::imod_object_add_contour;

    fn g9(v: f64) -> String {
        let mut buf = [0u8; 64];
        unsafe {
            libc::snprintf(
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                buf.len(),
                c"%.9g".as_ptr(),
                v,
            );
        }
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    fn mkcont(xy: &[f32], z: f32) -> Icont {
        let mut c = imod_contour_new().unwrap();
        for pair in xy.chunks(2) {
            imod_point_append(
                &mut c,
                Ipoint {
                    x: pair[0],
                    y: pair[1],
                    z,
                },
            );
        }
        c
    }

    fn dump(out: &mut String, tag: &str, c: Option<&Icont>) {
        let Some(c) = c else {
            out.push_str(&format!("{tag} NULL\n"));
            return;
        };
        out.push_str(&format!(
            "{tag} n={} flags={} surf={} time={}\n",
            c.pts.len(),
            c.flags,
            c.surf,
            c.time
        ));
        for i in 0..c.pts.len() {
            out.push_str(&format!(
                "{tag} {i} {} {} {}\n",
                g9(c.pts[i].x as f64),
                g9(c.pts[i].y as f64),
                g9(c.pts[i].z as f64)
            ));
        }
        if !c.sizes.is_empty() {
            for i in 0..c.pts.len() {
                out.push_str(&format!("{tag} sz {i} {}\n", g9(c.sizes[i] as f64)));
            }
        }
    }

    #[test]
    fn source_c_driver_differential_group2() {
        let sq = [0., 0., 10., 0., 10., 10., 0., 10.];
        let sq2 = [5., 5., 15., 5., 15., 15., 5., 15.];
        let sq3 = [2., 2., 8., 2., 8., 8., 2., 8.];
        let far = [100., 100., 110., 100., 110., 110., 100., 110.];
        let tri = [0.5, 0.25, 13.7, 2.2, 6.1, 11.9];
        let tilt = [0., 0., 10., 1., 10., 11., 0., 10., 3., 4., 7., 8.];
        let mut out = String::new();

        out.push_str("--- fitplane ---\n");
        {
            let mut fc = imod_contour_new().unwrap();
            let pts: [[f32; 3]; 8] = [
                [0., 0., 0.],
                [10., 0., 1.],
                [10., 10., 3.],
                [0., 10., 2.],
                [5., 0., 0.5],
                [10., 5., 2.],
                [5., 10., 2.5],
                [0., 5., 1.],
            ];
            for p in pts.iter() {
                imod_point_append(
                    &mut fc,
                    Ipoint {
                        x: p[0],
                        y: p[1],
                        z: p[2],
                    },
                );
            }
            let mut scale = Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            };
            let mut norm = Ipoint::default();
            let mut dval = 0.0f32;
            let mut alpha = 0.0f64;
            let mut beta = 0.0f64;
            out.push_str(&format!(
                "{}\n",
                imod_contour_fit_plane(&fc, &scale, &mut norm, &mut dval, &mut alpha, &mut beta)
            ));
            out.push_str(&format!(
                "{} {} {} {} {} {}\n",
                g9(norm.x as f64),
                g9(norm.y as f64),
                g9(norm.z as f64),
                g9(dval as f64),
                g9(alpha),
                g9(beta)
            ));
            scale.z = 2.5;
            out.push_str(&format!(
                "{}\n",
                imod_contour_fit_plane(&fc, &scale, &mut norm, &mut dval, &mut alpha, &mut beta)
            ));
            out.push_str(&format!(
                "{} {} {} {} {} {}\n",
                g9(norm.x as f64),
                g9(norm.y as f64),
                g9(norm.z as f64),
                g9(dval as f64),
                g9(alpha),
                g9(beta)
            ));
            let c = mkcont(&sq[..4], 0.);
            out.push_str(&format!(
                "{}\n",
                imod_contour_fit_plane(&c, &scale, &mut norm, &mut dval, &mut alpha, &mut beta)
            ));
        }

        out.push_str("--- on/nearest/inside ---\n");
        {
            let c = mkcont(&tilt, 4.);
            let mut i = 0usize;
            while i < 12 {
                out.push_str(&format!(
                    "on {i} {}\n",
                    imodel_contour_on(Some(&c), tilt[i] as i32, tilt[i + 1] as i32)
                ));
                i += 2;
            }
            out.push_str(&format!(
                "on none {}\n",
                imodel_contour_on(Some(&c), 55, 66)
            ));
            let p = Ipoint {
                x: 9.,
                y: 9.,
                z: 4.2,
            };
            out.push_str(&format!("nearest {}\n", imod_contour_nearest(Some(&c), &p)));
            out.push_str(&format!(
                "nearest2 {}\n",
                imodel_contour_nearest(Some(&c), 4, 4)
            ));
            let c1 = mkcont(&sq, 0.);
            let c2 = mkcont(&sq3, 0.);
            out.push_str(&format!(
                "inside {} {}\n",
                imod_contour_inside_cont(&c2, &c1),
                imod_contour_inside_cont(&c1, &c2)
            ));
        }

        out.push_str("--- join ---\n");
        {
            let mut c1 = mkcont(&sq, 0.);
            let mut c2 = mkcont(&far, 0.);
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), -1, -1, 0, 0);
            dump(&mut out, "join1", nc.as_ref());
        }
        {
            let mut c1 = mkcont(&sq, 0.);
            let mut c2 = mkcont(&far, 0.);
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), 1, 2, 1, 0);
            dump(&mut out, "join2", nc.as_ref());
        }
        {
            let mut c1 = mkcont(&sq, 0.);
            let mut c2 = mkcont(&far, 0.);
            imod_point_set_size(&mut c1, 0, 3.5);
            imod_point_set_size(&mut c2, 1, 6.25);
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), 0, 0, -1, 1);
            dump(&mut out, "join3", nc.as_ref());
        }
        {
            let mut big1 = [0.0f32; 20];
            let mut big2 = [0.0f32; 20];
            for k in 0..10 {
                let t = k as f64 * 2. * 3.14159265358979 / 10.;
                big1[2 * k] = (20. * t.cos()) as f32;
                big1[2 * k + 1] = (20. * t.sin()) as f32;
                big2[2 * k] = (60. + 20. * t.cos()) as f32;
                big2[2 * k + 1] = (20. * t.sin()) as f32;
            }
            let mut c1 = mkcont(&big1, 0.);
            let mut c2 = mkcont(&big2, 0.);
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), -1, -1, 14, 0);
            dump(&mut out, "join4", nc.as_ref());
        }

        out.push_str("--- splice/break ---\n");
        {
            let mut c1 = mkcont(&sq, 1.);
            let mut c2 = mkcont(&sq2, 2.);
            imod_point_set_size(&mut c1, 1, 9.);
            imod_point_set_size(&mut c2, 3, 8.);
            let nc = imod_contour_splice(Some(&c1), Some(&c2), 2, 1);
            dump(&mut out, "splice", nc.as_ref());
            out.push_str(&format!(
                "spliceerr {}\n",
                imod_contour_splice(Some(&c1), Some(&c2), 9, 1).is_none() as i32
            ));
        }
        {
            let mut c1 = mkcont(&sq, 1.);
            imod_point_set_size(&mut c1, 2, 7.);
            c1.flags |= ICONT_OPEN;
            c1.time = 5;
            c1.surf = 6;
            let nc = imod_contour_break(&mut c1, 1, 2);
            dump(&mut out, "break_new", nc.as_ref());
            dump(&mut out, "break_old", Some(&c1));
            let nc = imod_contour_break(&mut c1, 0, -1);
            dump(&mut out, "break2_new", nc.as_ref());
            dump(&mut out, "break2_old", Some(&c1));
        }

        out.push_str("--- scanadd ---\n");
        {
            let c1 = mkcont(&sq, 0.);
            let c2 = mkcont(&sq2, 0.);
            let s1 = imodel_contour_scan(Some(&c1)).unwrap();
            let s2 = imodel_contour_scan(Some(&c2)).unwrap();
            let sa = imod_contour_scan_add(Some(&s1), Some(&s2)).unwrap();
            /* Z of merged points is uninitialised in the source, so X and Y
            only */
            out.push_str(&format!("scanadd n={} flags={}\n", sa.pts.len(), sa.flags));
            for i in 0..sa.pts.len() {
                out.push_str(&format!(
                    "scanadd {i} {} {}\n",
                    g9(sa.pts[i].x as f64),
                    g9(sa.pts[i].y as f64)
                ));
            }
        }

        out.push_str("--- sort/reduce/shave/strip/double/fill ---\n");
        {
            let scat = [
                0., 0., 50., 50., 1., 1., 40., 40., 2., 3., 30., 30., 3., 2., 20., 20.,
            ];
            let mut c = mkcont(&scat, 1.);
            imod_point_set_size(&mut c, 0, 1.);
            imod_point_set_size(&mut c, 1, 2.);
            let scale = Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            };
            out.push_str(&format!("{}\n", imod_contour_sort3d(Some(&mut c), &scale)));
            dump(&mut out, "sort3d", Some(&c));
            let mut c = mkcont(&scat, 1.);
            out.push_str(&format!("{}\n", imodel_contour_sort(Some(&mut c))));
            dump(&mut out, "csort", Some(&c));
            let mut c = mkcont(&scat, 1.);
            out.push_str(&format!("{}\n", imod_contour_auto_sort(Some(&mut c))));
            dump(&mut out, "autosort", Some(&c));
        }
        {
            let mut line = [0.0f32; 22];
            for k in 0..11 {
                line[2 * k] = k as f32 * 3.;
                line[2 * k + 1] = if k % 2 != 0 { 0.2 } else { 0. };
            }
            let mut c = mkcont(&line, 0.);
            imod_contour_reduce(Some(&mut c), 1.0);
            dump(&mut out, "reduce", Some(&c));
            let mut c = mkcont(&line, 0.);
            imod_contour_reduce(Some(&mut c), 0.05);
            dump(&mut out, "reduce2", Some(&c));
        }
        {
            let close = [
                0., 0., 0.4, 0.1, 0.8, 0.2, 10., 0., 10., 10., 9.6, 9.9, 0., 10.,
            ];
            let mut c = mkcont(&close, 2.);
            out.push_str(&format!("{}\n", imod_contour_shave(&mut c, 1.0)));
            dump(&mut out, "shave", Some(&c));
        }
        {
            let coll = [0., 0., 1., 1., 2., 2., 3., 5., 3., 7., 3., 9.];
            let mut c = mkcont(&coll, 0.);
            out.push_str(&format!("{}\n", imod_contour_strip(&mut c)));
            dump(&mut out, "strip", Some(&c));
        }
        {
            let mut c = mkcont(&sq, 3.);
            let nc = imodel_contour_double(&c);
            dump(&mut out, "double", nc.as_ref());
            let nc = imod_contour_fill(Some(&c));
            dump(&mut out, "fill", nc.as_ref());
            c.flags |= ICONT_OPEN;
            let nc = imod_contour_fill(Some(&c));
            dump(&mut out, "fillopen", nc.as_ref());
        }

        out.push_str("--- overlap ---\n");
        {
            let c1 = mkcont(&sq, 0.);
            let c2 = mkcont(&sq2, 0.);
            let cf = mkcont(&far, 0.);
            let cin = mkcont(&sq3, 0.);
            let mut frac1 = 0.0f32;
            let mut frac2 = 0.0f32;
            out.push_str(&format!(
                "ov {} {} {}\n",
                imodel_contour_overlap(&c1, &c2),
                imodel_contour_overlap(&c1, &cf),
                imodel_contour_overlap(&c1, &cin)
            ));
            let mut s1 = imodel_contour_scan(Some(&c1)).unwrap();
            let mut s2 = imodel_contour_scan(Some(&c2)).unwrap();
            let mut mn1 = Ipoint::default();
            let mut mx1 = Ipoint::default();
            let mut mn2 = Ipoint::default();
            let mut mx2 = Ipoint::default();
            imod_contour_get_bbox(Some(&s1), &mut mn1, &mut mx1);
            imod_contour_get_bbox(Some(&s2), &mut mn2, &mut mx2);
            out.push_str(&format!(
                "so {}\n",
                imodel_scans_overlap(Some(&s1), mn1, mx1, Some(&s2), mn2, mx2)
            ));
            imodel_overlap_fractions(&mut s1, mn1, mx1, &mut s2, mn2, mx2, &mut frac1, &mut frac2);
            out.push_str(&format!("of {} {}\n", g9(frac1 as f64), g9(frac2 as f64)));
            let mut u1 = imod_contour_dup(&c1).unwrap();
            let mut u2 = imod_contour_dup(&cin).unwrap();
            imod_contour_get_bbox(Some(&u1), &mut mn1, &mut mx1);
            imod_contour_get_bbox(Some(&u2), &mut mn2, &mut mx2);
            out.push_str(&format!(
                "of2 {}\n",
                imodel_overlap_fractions(
                    &mut u1, mn1, mx1, &mut u2, mn2, mx2, &mut frac1, &mut frac2
                )
            ));
            out.push_str(&format!("of2 {} {}\n", g9(frac1 as f64), g9(frac2 as f64)));
            out.push_str(&format!("of2 flags {} {}\n", u1.flags, u2.flags));
        }

        out.push_str("--- swap/findpoint ---\n");
        {
            let mut c1 = mkcont(&sq, 1.);
            let mut c2 = mkcont(&tri, 2.);
            c1.flags = 8;
            c1.time = 3;
            c1.surf = 4;
            imod_contour_swap(&mut c1, &mut c2);
            dump(&mut out, "swap1", Some(&c1));
            dump(&mut out, "swap2", Some(&c2));
        }
        {
            let mut c = mkcont(&sq, 1.);
            let mut p = Ipoint {
                x: 10.,
                y: 10.,
                z: 1.,
            };
            out.push_str(&format!(
                "fp {}\n",
                imod_contour_find_point(Some(&c), Some(&p), ICONT_FIND_NOSORT)
            ));
            p.x = 99.;
            out.push_str(&format!(
                "fp {}\n",
                imod_contour_find_point(Some(&c), Some(&p), ICONT_FIND_NOSORT)
            ));
            let last = c.pts.len() as i32 - 1;
            imodel_contour_sortx(&mut c, 0, last);
            p.x = 10.;
            p.y = 10.;
            out.push_str(&format!(
                "fp {}\n",
                imod_contour_find_point(Some(&c), Some(&p), ICONT_FIND_SORTX)
            ));
            p.y = 77.;
            out.push_str(&format!(
                "fp {}\n",
                imod_contour_find_point(Some(&c), Some(&p), ICONT_FIND_SORTX)
            ));
            out.push_str(&format!(
                "fp {}\n",
                imod_contour_find_point(Some(&c), Some(&p), 9)
            ));
        }

        out.push_str("--- ztables/nesting ---\n");
        {
            let mut obj = Iobj::default();
            imod_object_add_contour(&mut obj, mkcont(&sq, 0.));
            imod_object_add_contour(&mut obj, mkcont(&sq3, 0.));
            imod_object_add_contour(&mut obj, mkcont(&far, 2.));
            let mut contz: Vec<i32> = Vec::new();
            let mut zlist: Vec<i32> = Vec::new();
            let mut numatz: Vec<i32> = Vec::new();
            let mut contatz: Vec<Vec<i32>> = Vec::new();
            let (mut zmin, mut zmax, mut zlsize, mut nummax) = (0, 0, 0, 0);
            out.push_str(&format!(
                "zt {}\n",
                imod_contour_make_z_tables(
                    &mut obj,
                    1,
                    ICONT_TEMPUSE,
                    &mut contz,
                    &mut zlist,
                    &mut numatz,
                    &mut contatz,
                    &mut zmin,
                    &mut zmax,
                    &mut zlsize,
                    &mut nummax,
                )
            ));
            out.push_str(&format!(
                "zt zmin {zmin} zmax {zmax} zlsize {zlsize} nummax {nummax}\n"
            ));
            for i in 0..obj.cont.len() {
                out.push_str(&format!("zt contz {i} {}\n", contz[i]));
            }
            for i in 0..zlsize as usize {
                out.push_str(&format!("zt zlist {i} {}\n", zlist[i]));
            }
            for i in 0..(zmax + 1 - zmin) as usize {
                out.push_str(&format!("zt numatz {i} {}", numatz[i]));
                for j in 0..numatz[i] as usize {
                    out.push_str(&format!(" {}", contatz[i][j]));
                }
                out.push('\n');
            }
            imod_contour_free_z_tables(
                &mut numatz,
                &mut contatz,
                &mut contz,
                &mut zlist,
                zmin,
                zmax,
            );

            let mut scans = vec![
                imodel_contour_scan(Some(&obj.cont[0])).unwrap(),
                imodel_contour_scan(Some(&obj.cont[1])).unwrap(),
            ];
            let mut pmin = [Ipoint::default(); 2];
            let mut pmax = [Ipoint::default(); 2];
            let (mut a, mut b) = (Ipoint::default(), Ipoint::default());
            imod_contour_get_bbox(Some(&scans[0]), &mut a, &mut b);
            pmin[0] = a;
            pmax[0] = b;
            imod_contour_get_bbox(Some(&scans[1]), &mut a, &mut b);
            pmin[1] = a;
            pmax[1] = b;
            let mut nests: Vec<Nesting> = Vec::new();
            let mut nestind = [-1i32, -1i32];
            let mut numnests = 0i32;
            let mut numwarn = 0i32;
            out.push_str(&format!(
                "nest {}\n",
                imod_contour_check_nesting(
                    0,
                    1,
                    &mut scans,
                    &pmin,
                    &pmax,
                    &mut nests,
                    &mut nestind,
                    &mut numnests,
                    &mut numwarn,
                )
            ));
            out.push_str(&format!(
                "nest numnests {numnests} numwarn {numwarn} ind {} {}\n",
                nestind[0], nestind[1]
            ));
            for i in 0..numnests as usize {
                out.push_str(&format!(
                    "nest {i} co {} level {} nin {} nout {}",
                    nests[i].co, nests[i].level, nests[i].ninside, nests[i].noutside
                ));
                for j in 0..nests[i].ninside as usize {
                    out.push_str(&format!(" in{}", nests[i].inside[j]));
                }
                for j in 0..nests[i].noutside as usize {
                    out.push_str(&format!(" out{}", nests[i].outside[j]));
                }
                out.push('\n');
            }
            imod_contour_nest_levels(&mut nests, &nestind, numnests);
            for i in 0..numnests as usize {
                out.push_str(&format!("nest level {i} {}\n", nests[i].level));
            }
            imod_contour_free_nests(&mut nests, numnests);
        }

        let want = super::EXPECTED_GROUP2_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from icont.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
    }
}

#[cfg(test)]
/// Verbatim stdout of the driver linked against `IMOD/libimod/icont.c`.
const EXPECTED_GROUP2_OUTPUT: &str = r#"--- fitplane ---
0
-0.0975899771 -0.195179939 0.975900114 1.1920929e-07 -0.19644091 0.0996686177
0
-0.218217894 -0.436435699 0.872871637 -0 -0.451633335 0.244978647
1
--- on/nearest/inside ---
on 0 1
on 2 10
on 4 1
on 6 10
on 8 1
on 10 1
on none 0
nearest 2
nearest2 4
inside 1 0
--- join ---
join1 n=10 flags=0 surf=0 time=0
join1 0 0 10 0
join1 1 9 10 0
join1 2 100 101 0
join1 3 100 110 0
join1 4 110 110 0
join1 5 110 100 0
join1 6 101 100 0
join1 7 10 9 0
join1 8 10 0 0
join1 9 0 0 0
join2 n=12 flags=16 surf=0 time=0
join2 0 0 10 0
join2 1 10 10 0
join2 2 10 0 0
join2 3 60 55 0.75
join2 4 110 110 0
join2 5 110 100 0
join2 6 100 100 0
join2 7 100 110 0
join2 8 110 110 0
join2 9 60 55 0.75
join2 10 10 0 0
join2 11 0 0 0
join3 n=12 flags=16 surf=0 time=0
join3 0 0 10 0
join3 1 10 10 0
join3 2 10 0 0
join3 3 0 0 0
join3 4 50 50 -0.75
join3 5 100 100 0
join3 6 110 100 0
join3 7 110 110 0
join3 8 100 110 0
join3 9 100 100 0
join3 10 50 50 -0.75
join3 11 0 0 0
join3 sz 0 -1
join3 sz 1 -1
join3 sz 2 -1
join3 sz 3 3.5
join3 sz 4 -1
join3 sz 5 -1
join3 sz 6 6.25
join3 sz 7 -1
join3 sz 8 -1
join3 sz 9 -1
join3 sz 10 -1
join3 sz 11 3.5
join4 n=22 flags=0 surf=0 time=0
join4 0 16.1803398 -11.7557049 0
join4 1 6.18033981 -19.0211296 0
join4 2 -6.18033981 -19.0211296 0
join4 3 -16.1803398 -11.7557049 0
join4 4 -20 6.46217834e-14 0
join4 5 -16.1803398 11.7557049 0
join4 6 -6.18033981 19.0211296 0
join4 7 6.18033981 19.0211296 0
join4 8 16.1803398 11.7557049 0
join4 9 19.3819656 1.90211308 0
join4 10 40.6180344 1.90211308 0
join4 11 43.8196602 11.7557049 0
join4 12 53.8196602 19.0211296 0
join4 13 66.1803436 19.0211296 0
join4 14 76.1803436 11.7557049 0
join4 15 80 0 0
join4 16 76.1803436 -11.7557049 0
join4 17 66.1803436 -19.0211296 0
join4 18 53.8196602 -19.0211296 0
join4 19 43.8196602 -11.7557049 0
join4 20 40.6180344 -1.90211308 0
join4 21 19.3819656 -1.90211308 0
--- splice/break ---
splice n=6 flags=16 surf=0 time=0
splice 0 0 0 1
splice 1 10 0 1
splice 2 10 10 1
splice 3 15 5 2
splice 4 15 15 2
splice 5 5 15 2
splice sz 0 -1
splice sz 1 9
splice sz 2 -1
splice sz 3 -1
splice sz 4 -1
splice sz 5 8
spliceerr 1
break_new n=2 flags=8 surf=6 time=5
break_new 0 10 0 1
break_new 1 10 10 1
break_new sz 0 -1
break_new sz 1 7
break_old n=2 flags=8 surf=6 time=5
break_old 0 0 0 1
break_old 1 0 10 1
break_old sz 0 -1
break_old sz 1 -1
break2_new n=2 flags=8 surf=6 time=5
break2_new 0 0 0 1
break2_new 1 0 10 1
break2_new sz 0 -1
break2_new sz 1 -1
break2_old n=0 flags=8 surf=6 time=5
--- scanadd ---
scanadd n=32 flags=131072
scanadd 0 0 0
scanadd 1 10 0
scanadd 2 0 1
scanadd 3 10 1
scanadd 4 0 2
scanadd 5 10 2
scanadd 6 0 3
scanadd 7 10 3
scanadd 8 0 4
scanadd 9 10 4
scanadd 10 0 5
scanadd 11 15 5
scanadd 12 0 6
scanadd 13 15 6
scanadd 14 0 7
scanadd 15 15 7
scanadd 16 0 8
scanadd 17 15 8
scanadd 18 0 9
scanadd 19 15 9
scanadd 20 0 10
scanadd 21 15 10
scanadd 22 5 11
scanadd 23 15 11
scanadd 24 5 12
scanadd 25 15 12
scanadd 26 5 13
scanadd 27 15 13
scanadd 28 5 14
scanadd 29 15 14
scanadd 30 5 15
scanadd 31 15 15
--- sort/reduce/shave/strip/double/fill ---
0
sort3d n=8 flags=0 surf=0 time=0
sort3d 0 0 0 1
sort3d 1 1 1 1
sort3d 2 2 3 1
sort3d 3 3 2 1
sort3d 4 20 20 1
sort3d 5 30 30 1
sort3d 6 40 40 1
sort3d 7 50 50 1
sort3d sz 0 1
sort3d sz 1 -1
sort3d sz 2 -1
sort3d sz 3 -1
sort3d sz 4 -1
sort3d sz 5 -1
sort3d sz 6 -1
sort3d sz 7 2
0
csort n=8 flags=0 surf=0 time=0
csort 0 0 0 1
csort 1 1 1 1
csort 2 2 3 1
csort 3 3 2 1
csort 4 20 20 1
csort 5 30 30 1
csort 6 40 40 1
csort 7 50 50 1
0
autosort n=8 flags=0 surf=0 time=0
autosort 0 0 0 1
autosort 1 1 1 1
autosort 2 2 3 1
autosort 3 3 2 1
autosort 4 20 20 1
autosort 5 30 30 1
autosort 6 40 40 1
autosort 7 50 50 1
reduce n=2 flags=0 surf=0 time=0
reduce 0 0 0 0
reduce 1 30 0 0
reduce2 n=11 flags=0 surf=0 time=0
reduce2 0 0 0 0
reduce2 1 3 0.200000003 0
reduce2 2 6 0 0
reduce2 3 9 0.200000003 0
reduce2 4 12 0 0
reduce2 5 15 0.200000003 0
reduce2 6 18 0 0
reduce2 7 21 0.200000003 0
reduce2 8 24 0 0
reduce2 9 27 0.200000003 0
reduce2 10 30 0 0
0
shave n=6 flags=0 surf=0 time=0
shave 0 0 0 2
shave 1 0.800000012 0.200000003 2
shave 2 10 0 2
shave 3 10 10 2
shave 4 9.60000038 9.89999962 2
shave 5 0 10 2
0
strip n=4 flags=0 surf=0 time=0
strip 0 0 0 0
strip 1 2 2 0
strip 2 3 5 0
strip 3 3 9 0
double n=8 flags=0 surf=0 time=0
double 0 0 0 3
double 1 5 0 3
double 2 10 0 3
double 3 10 5 3
double 4 10 10 3
double 5 5 10 3
double 6 0 10 3
double 7 0 5 3
fill n=40 flags=0 surf=0 time=0
fill 0 0 0 3
fill 1 1 0 3
fill 2 2 0 3
fill 3 3 0 3
fill 4 4 0 3
fill 5 5 0 3
fill 6 6 0 3
fill 7 7 0 3
fill 8 8 0 3
fill 9 9 0 3
fill 10 10 0 3
fill 11 10 1 3
fill 12 10 2 3
fill 13 10 3 3
fill 14 10 4 3
fill 15 10 5 3
fill 16 10 6 3
fill 17 10 7 3
fill 18 10 8 3
fill 19 10 9 3
fill 20 10 10 3
fill 21 9 10 3
fill 22 8 10 3
fill 23 7 10 3
fill 24 6 10 3
fill 25 5 10 3
fill 26 4 10 3
fill 27 3 10 3
fill 28 2 10 3
fill 29 1 10 3
fill 30 0 10 3
fill 31 0 9 3
fill 32 0 8 3
fill 33 0 7 3
fill 34 0 6 3
fill 35 0 5 3
fill 36 0 4 3
fill 37 0 3 3
fill 38 0 2 3
fill 39 0 1 3
fillopen n=30 flags=0 surf=0 time=0
fillopen 0 0 0 3
fillopen 1 1 0 3
fillopen 2 2 0 3
fillopen 3 3 0 3
fillopen 4 4 0 3
fillopen 5 5 0 3
fillopen 6 6 0 3
fillopen 7 7 0 3
fillopen 8 8 0 3
fillopen 9 9 0 3
fillopen 10 10 0 3
fillopen 11 10 1 3
fillopen 12 10 2 3
fillopen 13 10 3 3
fillopen 14 10 4 3
fillopen 15 10 5 3
fillopen 16 10 6 3
fillopen 17 10 7 3
fillopen 18 10 8 3
fillopen 19 10 9 3
fillopen 20 10 10 3
fillopen 21 9 10 3
fillopen 22 8 10 3
fillopen 23 7 10 3
fillopen 24 6 10 3
fillopen 25 5 10 3
fillopen 26 4 10 3
fillopen 27 3 10 3
fillopen 28 2 10 3
fillopen 29 1 10 3
--- overlap ---
ov 1 0 1
so 1
of 0.272727281 0.272727281
of2 1
of2 0.381818175 1
of2 flags 131072 131072
--- swap/findpoint ---
swap1 n=3 flags=0 surf=0 time=0
swap1 0 0.5 0.25 2
swap1 1 13.6999998 2.20000005 2
swap1 2 6.0999999 11.8999996 2
swap2 n=4 flags=8 surf=4 time=3
swap2 0 0 0 1
swap2 1 10 0 1
swap2 2 10 10 1
swap2 3 0 10 1
fp 2
fp -1
fp 2
fp -1
fp -1
--- ztables/nesting ---
zt 0
zt zmin 0 zmax 2 zlsize 2 nummax 2
zt contz 0 0
zt contz 1 0
zt contz 2 2
zt zlist 0 0
zt zlist 1 2
zt numatz 0 2 0 1
zt numatz 1 0
zt numatz 2 1 2
nest 0
nest numnests 2 numwarn 0 ind 1 0
nest 0 co 1 level 0 nin 0 nout 1 out0
nest 1 co 0 level 0 nin 1 nout 0 in1
nest level 0 2
nest level 1 1
"#;

/// Differential harness for the `icont.c` joining, splicing, breaking,
/// reduction and shaving routines when the contours carry general storage
/// lists, against a driver compiled directly against the pinned source and
/// linked to the reference build's `libimod`.
#[cfg(test)]
mod source_driver_store {
    use super::*;
    use crate::imod::libimod::istore::istore_insert_change;

    fn g9(v: f64) -> String {
        let mut buf = [0u8; 64];
        unsafe {
            libc::snprintf(
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                buf.len(),
                c"%.9g".as_ptr(),
                v,
            );
        }
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    fn mkcont(xy: &[f32], z: f32) -> Icont {
        let mut c = imod_contour_new().unwrap();
        for pair in xy.chunks(2) {
            imod_point_append(
                &mut c,
                Ipoint {
                    x: pair[0],
                    y: pair[1],
                    z,
                },
            );
        }
        c
    }

    fn addchange(c: &mut Icont, index: i32, type_: i16, val: i32) {
        let st = Istore {
            type_,
            flags: 0,
            index: StoreUnion { i: index },
            value: StoreUnion { i: val },
        };
        istore_insert_change(&mut c.store, st);
    }

    fn addgap(c: &mut Icont, index: i32) {
        /* GEN_STORE_GAP (`istore.h:41`), GEN_STORE_ONEPOINT (`istore.h:32`) */
        let st = Istore {
            type_: 4,
            flags: 1 << 7,
            index: StoreUnion { i: index },
            value: StoreUnion { i: 0 },
        };
        istore_add_one_index_item(&mut c.store, st);
    }

    fn dumps(out: &mut String, tag: &str, c: Option<&Icont>) {
        let Some(c) = c else {
            out.push_str(&format!("{tag} NULL\n"));
            return;
        };
        out.push_str(&format!(
            "{tag} n={} flags={} store={}\n",
            c.pts.len(),
            c.flags,
            c.store.len()
        ));
        for i in 0..c.pts.len() {
            out.push_str(&format!(
                "{tag} {i} {} {} {}\n",
                g9(c.pts[i].x as f64),
                g9(c.pts[i].y as f64),
                g9(c.pts[i].z as f64)
            ));
        }
        for i in 0..c.store.len() {
            let st = &c.store[i];
            out.push_str(&format!(
                "{tag} st {i} type={} flags={} index={} value={}\n",
                st.type_,
                st.flags,
                unsafe { st.index.i },
                unsafe { st.value.i }
            ));
        }
    }

    #[test]
    fn source_c_driver_differential_store() {
        let sq = [0.0f32, 0., 10., 0., 10., 10., 0., 10.];
        let far = [100.0f32, 100., 110., 100., 110., 110., 100., 110.];
        let mut out = String::new();

        out.push_str("--- join with store ---\n");
        {
            let mut c1 = mkcont(&sq, 0.);
            let mut c2 = mkcont(&far, 0.);
            addchange(&mut c1, 1, 1, 0x00ff00);
            addchange(&mut c1, 3, 3, 50);
            addchange(&mut c2, 0, 1, 0x0000ff);
            addgap(&mut c2, 2);
            dumps(&mut out, "j0c1", Some(&c1));
            dumps(&mut out, "j0c2", Some(&c2));
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), 1, 2, 1, 0);
            dumps(&mut out, "j0", nc.as_ref());
        }

        out.push_str("--- join open with store ---\n");
        {
            let mut c1 = mkcont(&sq, 0.);
            let mut c2 = mkcont(&far, 0.);
            c2.flags |= ICONT_OPEN;
            addchange(&mut c1, 0, 1, 0x112233);
            let nc = imod_contour_join(Some(&mut c1), Some(&mut c2), 0, 1, 0, 0);
            dumps(&mut out, "j1", nc.as_ref());
            dumps(&mut out, "j1c1", Some(&c1));
            dumps(&mut out, "j1c2", Some(&c2));
        }

        out.push_str("--- splice with store ---\n");
        {
            let mut c1 = mkcont(&sq, 1.);
            let mut c2 = mkcont(&far, 2.);
            addchange(&mut c1, 1, 1, 7);
            addchange(&mut c2, 2, 3, 33);
            let nc = imod_contour_splice(Some(&c1), Some(&c2), 2, 1);
            dumps(&mut out, "sp", nc.as_ref());
        }

        out.push_str("--- break with store ---\n");
        {
            let mut c1 = mkcont(&sq, 1.);
            addchange(&mut c1, 0, 1, 5);
            addchange(&mut c1, 2, 3, 9);
            let nc = imod_contour_break(&mut c1, 1, 2);
            dumps(&mut out, "bk", nc.as_ref());
            dumps(&mut out, "bkold", Some(&c1));
        }

        out.push_str("--- reduce/shave with store ---\n");
        {
            let mut line = [0.0f32; 22];
            for k in 0..11 {
                line[2 * k] = k as f32 * 3.;
                line[2 * k + 1] = if k % 2 != 0 { 0.2 } else { 0. };
            }
            let mut c1 = mkcont(&line, 0.);
            addchange(&mut c1, 4, 1, 12);
            imod_contour_reduce(Some(&mut c1), 1.0);
            dumps(&mut out, "rd", Some(&c1));
            let close = [
                0., 0., 0.4, 0.1, 0.8, 0.2, 10., 0., 10., 10., 9.6, 9.9, 0., 10.,
            ];
            let mut c1 = mkcont(&close, 2.);
            addchange(&mut c1, 1, 1, 4);
            out.push_str(&format!("sh {}\n", imod_contour_shave(&mut c1, 1.0)));
            dumps(&mut out, "sh", Some(&c1));
        }

        let want = super::EXPECTED_STORE_DRIVER_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from icont.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
    }
}

#[cfg(test)]
/// Verbatim stdout of the driver linked against `IMOD/libimod/icont.c`.
const EXPECTED_STORE_DRIVER_OUTPUT: &str = r#"--- join with store ---
j0c1 n=4 flags=0 store=2
j0c1 0 0 0 0
j0c1 1 10 0 0
j0c1 2 10 10 0
j0c1 3 0 10 0
j0c1 st 0 type=1 flags=0 index=1 value=65280
j0c1 st 1 type=3 flags=0 index=3 value=50
j0c2 n=4 flags=0 store=2
j0c2 0 100 100 0
j0c2 1 110 100 0
j0c2 2 110 110 0
j0c2 3 100 110 0
j0c2 st 0 type=1 flags=0 index=0 value=255
j0c2 st 1 type=4 flags=128 index=2 value=0
j0 n=12 flags=16 store=10
j0 0 0 10 0
j0 1 10 10 0
j0 2 10 0 0
j0 3 60 55 0.75
j0 4 110 110 0
j0 5 110 100 0
j0 6 100 100 0
j0 7 100 110 0
j0 8 110 110 0
j0 9 60 55 0.75
j0 10 10 0 0
j0 11 0 0 0
j0 st 0 type=1 flags=0 index=0 value=65280
j0 st 1 type=3 flags=0 index=0 value=50
j0 st 2 type=3 flags=32 index=1 value=50
j0 st 3 type=1 flags=0 index=3 value=65280
j0 st 4 type=1 flags=0 index=4 value=255
j0 st 5 type=1 flags=0 index=7 value=255
j0 st 6 type=4 flags=128 index=7 value=0
j0 st 7 type=1 flags=0 index=9 value=255
j0 st 8 type=1 flags=0 index=10 value=65280
j0 st 9 type=1 flags=32 index=11 value=65280
--- join open with store ---
j1 n=10 flags=8 store=3
j1 0 100 110 0
j1 1 110 110 0
j1 2 110 101 0
j1 3 0 1 0
j1 4 0 10 0
j1 5 10 10 0
j1 6 10 0 0
j1 7 1 0 0
j1 8 109 100 0
j1 9 100 100 0
j1 st 0 type=1 flags=0 index=3 value=1122867
j1 st 1 type=1 flags=0 index=4 value=1122867
j1 st 2 type=1 flags=32 index=8 value=1122867
j1c1 n=4 flags=0 store=2
j1c1 0 0 10 0
j1c1 1 10 10 0
j1c1 2 10 0 0
j1c1 3 0 0 0
j1c1 st 0 type=1 flags=0 index=0 value=1122867
j1c1 st 1 type=1 flags=32 index=4 value=1122867
j1c2 n=4 flags=8 store=0
j1c2 0 100 110 0
j1c2 1 110 110 0
j1c2 2 110 100 0
j1c2 3 100 100 0
--- splice with store ---
sp n=6 flags=16 store=4
sp 0 0 0 1
sp 1 10 0 1
sp 2 10 10 1
sp 3 110 100 2
sp 4 110 110 2
sp 5 100 110 2
sp st 0 type=1 flags=0 index=1 value=7
sp st 1 type=1 flags=32 index=3 value=7
sp st 2 type=3 flags=0 index=4 value=33
sp st 3 type=3 flags=32 index=6 value=33
--- break with store ---
bk n=2 flags=0 store=4
bk 0 10 0 1
bk 1 10 10 1
bk st 0 type=1 flags=0 index=0 value=5
bk st 1 type=3 flags=0 index=1 value=9
bk st 2 type=1 flags=32 index=2 value=5
bk st 3 type=3 flags=32 index=2 value=9
bkold n=2 flags=0 store=6
bkold 0 0 0 1
bkold 1 0 10 1
bkold st 0 type=1 flags=0 index=0 value=5
bkold st 1 type=1 flags=32 index=1 value=5
bkold st 2 type=1 flags=0 index=1 value=5
bkold st 3 type=3 flags=0 index=1 value=9
bkold st 4 type=1 flags=32 index=2 value=5
bkold st 5 type=3 flags=32 index=2 value=9
--- reduce/shave with store ---
rd n=3 flags=0 store=1
rd 0 0 0 0
rd 1 12 0 0
rd 2 30 0 0
rd st 0 type=1 flags=0 index=1 value=12
sh 0
sh n=7 flags=0 store=1
sh 0 0 0 2
sh 1 0.400000006 0.100000001 2
sh 2 0.800000012 0.200000003 2
sh 3 10 0 2
sh 4 10 10 2
sh 5 9.60000038 9.89999962 2
sh 6 0 10 2
sh st 0 type=1 flags=0 index=1 value=4
"#;

/// `imodContourGetLabel` (`icont.c:3320`).  Returns None for no contour or no
/// label.
pub fn imod_contour_get_label(
    in_contour: Option<&Icont>,
) -> Option<&crate::imod::libimod::ilabel::Ilabel> {
    in_contour?.label.as_ref()
}

/// `imodContourSetLabel` (`icont.c:3328`).  Frees any existing label first.
/// Note the source returns without doing anything when `inLabel` is NULL, so a
/// None label does **not** clear an existing one.
pub fn imod_contour_set_label(
    in_contour: Option<&mut Icont>,
    in_label: Option<crate::imod::libimod::ilabel::Ilabel>,
) {
    let (contour, label) = match (in_contour, in_label) {
        (Some(contour), Some(label)) => (contour, label),
        _ => return,
    };
    crate::imod::libimod::ilabel::imod_label_delete(contour.label.take());
    contour.label = Some(label);
}

/// `imodContourGetName` (`icont.c:3418`).  Returns the empty string when there
/// is no contour, no label, or no name — the source returns a pointer to a
/// static NUL byte.
pub fn imod_contour_get_name(in_contour: Option<&Icont>) -> &[u8] {
    match in_contour
        .and_then(|contour| contour.label.as_ref())
        .and_then(|label| label.name.as_deref())
    {
        Some(name) => name,
        None => b"\0",
    }
}
