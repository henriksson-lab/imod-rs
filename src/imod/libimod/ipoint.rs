//! Translation of `IMOD/libimod/ipoint.c` -- "Point editing functions for IMOD
//! models".
//!
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::imodel_contour_check_wild;
use crate::imod::libimod::imodel::{ICONT_WILD, Icont, Iobj, Iplane, Ipoint};
use crate::imod::libimod::iplane::imod_planes_clip;
use crate::imod::libimod::istore::{istore_delete_point, istore_shift_index};

/// Original: `imodPointAppend` (`ipoint.c:23`).
///
/// Adds point `pnt` to the end of contour `cont`.  Manages point sizes and
/// labels correctly.  Returns number of points in contour, or 0 if an error
/// occurs.
pub fn imod_point_append(cont: &mut Icont, pnt: Ipoint) -> i32 {
    let psize = cont.pts.len() as i32;
    imod_point_add(cont, Some(pnt), psize)
}

/// Original: `imodPointAppendXYZ` (`ipoint.c:33`).
///
/// Adds the point `x`, `y`, `z` to the end of contour `cont`.
pub fn imod_point_append_xyz(cont: &mut Icont, x: f32, y: f32, z: f32) -> i32 {
    let pnt = Ipoint { x, y, z };
    let psize = cont.pts.len() as i32;
    imod_point_add(cont, Some(pnt), psize)
}

/// Original: `imodPointAdd` (`ipoint.c:47`).
///
/// Adds point `point` to contour `cont` at the given `index`.
pub fn imod_point_add(cont: &mut Icont, point: Option<Ipoint>, mut index: i32) -> i32 {
    if index > cont.pts.len() as i32 {
        index = cont.pts.len() as i32;
    }
    if index < 0 {
        return 0;
    }
    let Some(point) = point else {
        return cont.pts.len() as i32;
    };
    let start_z = cont.pts.first().map_or(point.z, |first| first.z);
    cont.pts.insert(index as usize, point);
    /* `ipoint.c:90-93` shifts every point above `index` up by one and moves
    the label item with it. */
    for i in ((index as usize + 1)..cont.pts.len()).rev() {
        crate::imod::libimod::ilabel::imod_label_item_move(
            cont.label.as_mut(),
            i as i32,
            i as i32 - 1,
        );
    }
    if !cont.sizes.is_empty() {
        cont.sizes.insert(index as usize, -1.0);
    }
    istore_shift_index(&mut cont.store, index, -1, 1);
    if (start_z + 0.5).floor() as i32 != (point.z + 0.5).floor() as i32 {
        cont.flags |= ICONT_WILD;
    }
    cont.pts.len() as i32
}

/// Original: `imodPointDelete` (`ipoint.c:119`).
///
/// Deletes the point at `index` from contour `cont`.  Returns the size of the
/// contour or -1 for error.
pub fn imod_point_delete(cont: &mut Icont, index: i32) -> i32 {
    if index < 0 || index > cont.pts.len() as i32 - 1 {
        return -1;
    }

    crate::imod::libimod::ilabel::imod_label_item_delete(cont.label.as_mut(), index);
    let index = index as usize;
    for i in index..cont.pts.len() - 1 {
        cont.pts[i].x = cont.pts[i + 1].x;
        cont.pts[i].y = cont.pts[i + 1].y;
        cont.pts[i].z = cont.pts[i + 1].z;
        crate::imod::libimod::ilabel::imod_label_item_move(
            cont.label.as_mut(),
            i as i32,
            i as i32 + 1,
        );
    }
    if !cont.sizes.is_empty() {
        for i in index..cont.pts.len() - 1 {
            cont.sizes[i] = cont.sizes[i + 1];
        }
    }

    /* Manage the storage list.  Should catch error, but not clear what to
    do if one occurs */
    istore_delete_point(&mut cont.store, index as i32, cont.pts.len() as i32);
    cont.pts.pop();
    if !cont.sizes.is_empty() {
        cont.sizes.pop();
    }

    if cont.pts.is_empty() {
        cont.sizes.clear();
        return 0;
    }

    /* DNM: if the wild flag is set, recheck contour */
    if cont.flags & ICONT_WILD != 0 {
        imodel_contour_check_wild(Some(cont));
    }
    cont.pts.len() as i32
}

/// Original: `imodPointSetSize` (`ipoint.c:167`).
///
/// Sets the size of point at `pt` in contour `cont` to `size`.  Creates a size
/// array if necessary.
pub fn imod_point_set_size(cont: &mut Icont, pt: i32, size: f32) {
    if cont.sizes.is_empty() {
        cont.sizes = vec![0.0f32; cont.pts.len()];
        for i in 0..cont.pts.len() {
            cont.sizes[i] = -1.;
        }
    }
    cont.sizes[pt as usize] = size;
}

/// Original: `imodPointGetSize` (`ipoint.c:185`).
///
/// Returns the size of the point at `pt` in contour `cont` and object `obj`.
pub fn imod_point_get_size(obj: &Iobj, cont: &Icont, pt: i32) -> f32 {
    if cont.sizes.is_empty() {
        return obj.pdrawsize as f32;
    }
    if cont.sizes[pt as usize] < 0. {
        return obj.pdrawsize as f32;
    }
    cont.sizes[pt as usize]
}

/// Original: `imodel_point_dist` (`ipoint.c:197`).
///
/// Returns distance in the X/Y plane between `pnt1` and `pnt2`.
pub fn imodel_point_dist(pnt1: &Ipoint, pnt2: &Ipoint) -> f64 {
    let distance =
        ((pnt1.x - pnt2.x) * (pnt1.x - pnt2.x)) + ((pnt1.y - pnt2.y) * (pnt1.y - pnt2.y));
    (distance as f64).sqrt()
}

/// Original: `imodPointDistance` (`ipoint.c:211`).
///
/// Returns distance in the X/Y plane between `pnt1` and `pnt2`.
pub fn imod_point_distance(pnt1: &Ipoint, pnt2: &Ipoint) -> f32 {
    let mut distance =
        ((pnt1.x - pnt2.x) * (pnt1.x - pnt2.x)) + ((pnt1.y - pnt2.y) * (pnt1.y - pnt2.y));
    distance = (distance as f64).sqrt() as f32;
    distance
}

/// Original: `imodPoint3DScaleDistance` (`ipoint.c:225`).
///
/// Returns distance in 3D between points `p1` and `p2`, with coordinates
/// scaled by the values in `scale`.
pub fn imod_point3d_scale_distance(p1: &Ipoint, p2: &Ipoint, scale: &Ipoint) -> f32 {
    let xd = (p1.x - p2.x) * scale.x;
    let yd = (p1.y - p2.y) * scale.y;
    let zd = (p1.z - p2.z) * scale.z;
    let dist = ((xd * xd) + (yd * yd) + (zd * zd)) as f64;
    dist.sqrt() as f32
}

/// Original: `imodPointLineDistance` (`ipoint.c:235`).
///
/// Huh?  Used by icont_alldist which is used by bad principal axis.
pub fn imod_point_line_distance(ln: &Ipoint, p: &Ipoint) -> f32 {
    let l = (ln.x * p.x) + (ln.y * p.y) + ln.z;
    let d = (ln.x * ln.x) + (ln.y * ln.y);

    if d != 0. {
        (((l * l) / d) as f64).sqrt() as f32
    } else {
        unsafe {
            libc::printf(c"ipd00:\n".as_ptr());
        }
        0.0
    }
}

/// Original: `imodPoint2DAngle` (`ipoint.c:252`).
///
/// Returns angle of line from origin to X, Y coordinates of `pt`, between
/// -pi/2 and pi/2.
pub fn imod_point2d_angle(pt: &Ipoint) -> f64 {
    let angle;
    if pt.x != 0.0 {
        angle = ((pt.y / pt.x) as f64).atan();
    } else if pt.y != 0.0 {
        angle = 1.570796327;
    } else {
        angle = 0.0;
    }
    angle
}

/// Original: `imodPointLineSegDistance` (`ipoint.c:271`).
///
/// Returns the *square* of the distance between point `p` and the line segment
/// between `lp1` and `lp2`; `tval` gets the parameter of closest approach.
pub fn imod_point_line_seg_distance(lp1: &Ipoint, lp2: &Ipoint, p: &Ipoint, tval: &mut f32) -> f32 {
    let mut t: f32 = 0.;
    let a = lp2.x - lp1.x;
    let b = lp2.y - lp1.y;
    let c = lp2.z - lp1.z;
    if a != 0. || b != 0. || c != 0. {
        t = (a * (p.x - lp1.x) + b * (p.y - lp1.y) + c * (p.z - lp1.z)) / (a * a + b * b + c * c);
    }
    /* B3DMAX(0., B3DMIN(1., t)) -- the literals are double, so the comparisons
    and the result are done in double before the assignment back to float. */
    let mut td: f64 = if 1.0f64 < t as f64 { 1.0f64 } else { t as f64 };
    td = if 0.0f64 > td { 0.0f64 } else { td };
    t = td as f32;
    *tval = t;
    let d = a * t + lp1.x - p.x;
    let e = b * t + lp1.y - p.y;
    let f = c * t + lp1.z - p.z;
    d * d + e * e + f * f
}

/// Original: `imodPointContDistance` (`ipoint.c:299`).
///
/// Returns the closest distance between the line segments in `cont` and point
/// `pt`, and in `closest` the index of the point at the beginning of the
/// closest line segment.
pub fn imod_point_cont_distance(
    cont: &Icont,
    pt: &Ipoint,
    open: i32,
    three_d: i32,
    closest: &mut i32,
) -> f32 {
    let mut mindist: f32 = 1.0e36;
    let mut t: f32 = 0.;
    let mut dist: f32;
    let mut dx: f32;
    let mut dy: f32;
    let cpts = &cont.pts;
    let scale = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let mut num_seg = cont.pts.len() as i32;
    *closest = 0;
    if cont.pts.is_empty() {
        return 0.;
    }
    if cont.pts.len() == 1 {
        return if three_d != 0 {
            imod_point3d_scale_distance(pt, &cpts[0], &scale)
        } else {
            imod_point_distance(pt, &cpts[0])
        };
    }
    if open != 0 {
        num_seg -= 1;
    }
    for i in 0..num_seg as usize {
        let ni = (i + 1) % cont.pts.len();
        if three_d != 0 {
            dist = imod_point_line_seg_distance(&cpts[i], &cpts[ni], pt, &mut t);
        } else {
            dx = cpts[ni].x - cpts[i].x;
            dy = cpts[ni].y - cpts[i].y;
            if dx != 0. || dy != 0. {
                t = ((pt.x - cpts[i].x) * dx + (pt.y - cpts[i].y) * dy) / (dx * dx + dy * dy);
            }
            /* B3DMIN(1., B3DMAX(0., t)) in double, as above */
            let mut td: f64 = if 0.0f64 > t as f64 { 0.0f64 } else { t as f64 };
            td = if 1.0f64 < td { 1.0f64 } else { td };
            t = td as f32;
            dx = pt.x - (cpts[i].x + t * dx);
            dy = pt.y - (cpts[i].y + t * dy);
            dist = dx * dx + dy * dy;
        }
        if dist < mindist {
            *closest = i as i32;
            mindist = dist;
        }
    }
    (mindist as f64).sqrt() as f32
}

/// Original: `imodPointDot` (`ipoint.c:338`).
///
/// Returns dot product of `pnt1` and `pnt2`.
pub fn imod_point_dot(pnt1: &Ipoint, pnt2: &Ipoint) -> f32 {
    (pnt1.x * pnt2.x) + (pnt1.y * pnt2.y) + (pnt1.z * pnt2.z)
}

/// Original: `imodPointCross` (`ipoint.c:348`).
///
/// Returns cross product of `v1` and `v2` in `rp`.
pub fn imod_point_cross(v1: &Ipoint, v2: &Ipoint, rp: &mut Ipoint) {
    /* find the normal for the plane with the points p1,p2,p3 */
    rp.x = (v1.y * v2.z) - (v1.z * v2.y);
    rp.y = (v1.z * v2.x) - (v1.x * v2.z);
    rp.z = (v1.x * v2.y) - (v1.y * v2.x);
}

/// Original: `imodPointNormalize` (`ipoint.c:360`).
///
/// Normalizes vector in `n` to length 1.
pub fn imod_point_normalize(n: &mut Ipoint) {
    let mut dist;

    dist = (n.x * n.x) + (n.y * n.y) + (n.z * n.z);
    dist = (dist as f64).sqrt() as f32;
    if dist == 0.0 {
        n.x = 0.;
        n.y = 0.;
        n.z = 0.;
    } else {
        dist = 1. / dist;

        n.x *= dist;
        n.y *= dist;
        n.z *= dist;
    }
}

/// Original: `imodPointIsEqual` (`ipoint.c:385`).
///
/// Returns 1 if point `a` equals point `b`, 0 otherwise.
pub fn imod_point_is_equal(a: &Ipoint, b: &Ipoint) -> i32 {
    if (a.x == b.x) && (a.y == b.y) && (a.z == b.z) {
        return 1;
    }
    0
}

/// Original: `imodPointIntersect` (`ipoint.c:396`).
///
/// Returns 1 if the line segment between `a` and `b` intersects the line
/// segment between `c` and `d`, 0 otherwise.
pub fn imod_point_intersect(a: &Ipoint, b: &Ipoint, c: &Ipoint, d: &Ipoint) -> i32 {
    let epsilon: f32 = 1.0e-2;

    /* First test whether bounding boxes of the two segments overlap in X and Y */
    let mut ab_min = if a.x < b.x { a.x } else { b.x };
    let mut ab_max = if a.x > b.x { a.x } else { b.x };
    let mut cd_min = if c.x < d.x { c.x } else { d.x };
    let mut cd_max = if c.x > d.x { c.x } else { d.x };
    if ab_min > cd_max + epsilon || cd_min > ab_max + epsilon {
        return 0;
    }
    ab_min = if a.y < b.y { a.y } else { b.y };
    ab_max = if a.y > b.y { a.y } else { b.y };
    cd_min = if c.y < d.y { c.y } else { d.y };
    cd_max = if c.y > d.y { c.y } else { d.y };
    if ab_min > cd_max + epsilon || cd_min > ab_max + epsilon {
        return 0;
    }

    /* Compute parameters t and u for point of intersection along each
    extended line (t and u between 0 and 1 parameterize each line segment) */
    let dx1 = b.x - a.x;
    let dy1 = b.y - a.y;
    let dx2 = d.x - c.x;
    let dy2 = d.y - c.y;
    let dxs = c.x - a.x;
    let dys = c.y - a.y;
    let mut den = (dx2 * dy1 - dx1 * dy2) as f64;
    let tnum = (dys * dx2 - dxs * dy2) as f64;
    let unum = (dx1 * dys - dy1 * dxs) as f64;

    /* Check for parallel lines */
    if den.abs() < 1.0e-20
        || den.abs()
            < 1.0e-6
                * if tnum.abs() > unum.abs() {
                    tnum.abs()
                } else {
                    unum.abs()
                }
    {
        /* For parallel lines, check segment length, then check each endpoint
        against the other segment for being within the segment */
        den = (dx1 * dx1 + dy1 * dy1) as f64;
        if den.abs() < 1.0e-20 {
            return 0;
        }
        let mut t = ((dx1 * (c.x - a.x) + dy1 * (c.y - a.y)) as f64) / den;
        if t >= -0.000001
            && t <= 1.000001
            && (a.x as f64 + t * dx1 as f64 - c.x as f64).abs() < 1.0e-6
            && (a.y as f64 + t * dy1 as f64 - c.y as f64).abs() < 1.0e-6
        {
            return 1;
        }
        t = ((dx1 * (d.x - a.x) + dy1 * (d.y - a.y)) as f64) / den;
        if t >= -0.000001
            && t <= 1.000001
            && (a.x as f64 + t * dx1 as f64 - d.x as f64).abs() < 1.0e-6
            && (a.y as f64 + t * dy1 as f64 - d.y as f64).abs() < 1.0e-6
        {
            return 1;
        }

        den = (dx2 * dx2 + dy2 * dy2) as f64;
        if den.abs() < 1.0e-20 {
            return 0;
        }
        t = ((dx2 * (a.x - c.x) + dy2 * (a.y - c.y)) as f64) / den;
        if t >= -0.000001
            && t <= 1.000001
            && (c.x as f64 + t * dx2 as f64 - a.x as f64).abs() < 1.0e-6
            && (c.y as f64 + t * dy2 as f64 - a.y as f64).abs() < 1.0e-6
        {
            return 1;
        }
        t = ((dx2 * (b.x - c.x) + dy2 * (b.y - c.y)) as f64) / den;
        if t >= -0.000001
            && t <= 1.000001
            && (c.x as f64 + t * dx2 as f64 - b.x as f64).abs() < 1.0e-6
            && (c.y as f64 + t * dy2 as f64 - b.y as f64).abs() < 1.0e-6
        {
            return 1;
        }
        return 0;
    }

    /* Non-parallel lines: test for point of intersection within each line */
    let t = tnum / den;
    let u = unum / den;
    if t >= 0. && t <= 1. && u >= 0. && u <= 1. {
        1
    } else {
        0
    }
}

/// Original: `imodPointPlaneEdge` (`ipoint.c:469`).
///
/// Called from uncompiled part of imodinfo and unused and suspect
/// `imodObjectClip`.
pub fn imod_point_plane_edge(
    rpt: &mut Ipoint,
    plane: &[Iplane],
    planes: i32,
    pt1: &Ipoint,
    pt2: &Ipoint,
) -> i32 {
    let mut step = Ipoint::default();
    let mut cpt;
    let pdist = imod_point_distance(pt1, pt2);
    let p1f = imod_planes_clip(plane, planes, pt1);
    let p2f = imod_planes_clip(plane, planes, pt2);

    if pdist <= 0.0f32 {
        return -1;
    }
    if p1f != 0 && p2f != 0 {
        return 1;
    }
    if p1f == 0 && p2f == 0 {
        return 2;
    }

    step.x = (pt1.x - pt2.x) / pdist;
    step.y = (pt1.y - pt2.y) / pdist;
    step.z = (pt1.z - pt2.z) / pdist;

    let mut pt1 = pt1;
    let mut pt2 = pt2;
    if imod_planes_clip(plane, planes, pt1) == 0 {
        step.x *= -1.;
        step.y *= -1.;
        step.z *= -1.;
        let tpt = pt2;
        pt2 = pt1;
        pt1 = tpt;
    }

    let li = (pdist + 0.5f32) as i32;
    cpt = *pt1;
    *rpt = *pt1;
    for _i in 0..li {
        cpt.x += step.x;
        cpt.y += step.y;
        cpt.z += step.z;
        if imod_planes_clip(plane, planes, &cpt) != 0 {
            return 0;
        }
        *rpt = cpt;
    }
    0
}

/// Original: `imodPointArea` (`ipoint.c:515`).
///
/// Returns the area of the 3D triangle formed by `p1`, `p2`, and `p3`.
pub fn imod_point_area(p1: &Ipoint, p2: &Ipoint, p3: &Ipoint) -> f32 {
    let mut n = Ipoint::default();
    let mut n1 = Ipoint::default();
    let mut n2 = Ipoint::default();

    n1.x = p1.x - p2.x;
    n1.y = p1.y - p2.y;
    n1.z = p1.z - p2.z;
    n2.x = p3.x - p2.x;
    n2.y = p3.y - p2.y;
    n2.z = p3.z - p2.z;
    imod_point_cross(&n1, &n2, &mut n);

    (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32
}

/// Original: `imodPointAreaScale` (`ipoint.c:534`).
///
/// Returns the area of the 3D triangle formed by `p1`, `p2`, and `p3`, with
/// point coordinates scaled by `s`.
pub fn imod_point_area_scale(p1: &Ipoint, p2: &Ipoint, p3: &Ipoint, s: &Ipoint) -> f32 {
    let mut n = Ipoint::default();
    let mut n1 = Ipoint::default();
    let mut n2 = Ipoint::default();

    n1.x = (p1.x - p2.x) * s.x;
    n1.y = (p1.y - p2.y) * s.y;
    n1.z = (p1.z - p2.z) * s.z;
    n2.x = (p3.x - p2.x) * s.x;
    n2.y = (p3.y - p2.y) * s.y;
    n2.z = (p3.z - p2.z) * s.z;
    imod_point_cross(&n1, &n2, &mut n);

    (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32
}

/// Original: `imodPointInsideCont` (`ipoint.c:558`).
///
/// Returns 1 if the point `pt` is inside or on the contour `cont`, otherwise
/// returns 0.  Based on the algorithm in "Computational Geometry in C", Joseph
/// O'Rourke, 1998, with modifications to speed up search for ray crossings.
pub fn imod_point_inside_cont(cont: &Icont, pt: &Ipoint) -> i32 {
    let pts = &cont.pts;
    let mut rstrad: bool;
    let mut lstrad: bool;

    let mut nrcross = 0i32;
    let mut nlcross = 0i32;
    let np = cont.pts.len() as i32;
    let x = pt.x;
    let y = pt.y;
    let mut yp;
    let mut xp;
    let mut xc;
    let mut yc;
    let mut xcross;

    yp = pts[(np - 1) as usize].y;
    let mut j = 0i32;
    while j < np {
        if yp < y {
            /* if last point below y, search for first that is not below */
            while j < np && pts[j as usize].y < y {
                j += 1;
            }
        } else if yp > y {
            /* or if last point above y, search for first that is not above */
            while j < np && pts[j as usize].y > y {
                j += 1;
            }
        }

        if j < np {
            let mut jl = j - 1;
            if jl < 0 {
                jl = np - 1;
            }
            xp = pts[jl as usize].x;
            yp = pts[jl as usize].y;
            xc = pts[j as usize].x;
            yc = pts[j as usize].y;

            /* return if point is a vertex */
            if x == xc && y == yc {
                return 1;
            }

            /* does edge straddle the ray to the right or the left? */
            rstrad = (yc > y) != (yp > y);
            lstrad = (yc < y) != (yp < y);
            if lstrad || rstrad {
                /* if so, compute the crossing of the ray, add up crossings */
                xcross = xp + (y - yp) * (xc - xp) / (yc - yp);
                if rstrad && (xcross > x) {
                    nrcross += 1;
                }
                if lstrad && (xcross < x) {
                    nlcross += 1;
                }
            }
            yp = yc;
        }
        j += 1;
    }

    /* if left and right crossings don't match, it's on an edge
    otherwise, inside iff crossings are odd */
    if nrcross % 2 != nlcross % 2 {
        return 1;
    }
    ((nrcross % 2) > 0) as i32
}

/// Original: `imodPointInsideArea` (`ipoint.c:635`).
///
/// Tests whether the point `x`, `y` is inside any of the contours in object
/// `obj` listed in `list`.
pub fn imod_point_inside_area(obj: &Iobj, list: &[i32], nlist: i32, x: f32, y: f32) -> i32 {
    let pnt = Ipoint { x, y, z: 0. };
    for i in 0..nlist as usize {
        if imod_point_inside_cont(&obj.cont[list[i] as usize], &pnt) != 0 {
            return list[i];
        }
    }
    -1
}

/// Original: `makeAreaContList` (`ipoint.c:656`).
///
/// Makes a list of the contours in object `obj` on the Z section closest to
/// the Z value `iz`.
pub fn make_area_cont_list(
    obj: &Iobj,
    iz: i32,
    list: &mut [i32],
    nlist: &mut i32,
    list_size: i32,
) -> i32 {
    let mut dzmin;
    let mut izmin;
    let mut zco;
    let mut dz;
    izmin = -999;
    dzmin = 100000;
    for co in 0..obj.cont.len() {
        if obj.cont[co].pts.is_empty() {
            continue;
        }
        zco = (obj.cont[co].pts[0].z as f64 + 0.5).floor() as i32;
        dz = if iz - zco > zco - iz {
            iz - zco
        } else {
            zco - iz
        };
        if dz < dzmin {
            dzmin = dz;
            izmin = zco;
        }
    }

    if izmin == -999 {
        return 1;
    }
    *nlist = 0;
    for co in 0..obj.cont.len() {
        if obj.cont[co].pts.is_empty() {
            continue;
        }
        zco = (obj.cont[co].pts[0].z as f64 + 0.5).floor() as i32;
        if zco == izmin {
            if *nlist == list_size - 1 {
                return -1;
            }
            list[*nlist as usize] = co as i32;
            *nlist += 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::icont::{imod_contour_new, imod_contour_z_value};
    use crate::imod::libimod::iplane::imod_plane_init;

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

    /// Differential harness against a driver compiled directly against the
    /// pinned `IMOD/libimod/ipoint.c` and linked to the reference build's
    /// `libimod`; the expected text below is that driver's verbatim output.
    #[test]
    fn source_c_driver_differential() {
        let mut out = String::new();
        macro_rules! p {
            ($v:expr) => {
                out.push_str(&format!("{}\n", g9($v as f64)))
            };
        }
        macro_rules! p3 {
            ($v:expr) => {
                out.push_str(&format!(
                    "{} {} {}\n",
                    g9($v.x as f64),
                    g9($v.y as f64),
                    g9($v.z as f64)
                ))
            };
        }
        out.push_str("--- ipoint ---\n");

        let a = Ipoint {
            x: 1.25,
            y: -3.5,
            z: 4.,
        };
        let b = Ipoint {
            x: -2.75,
            y: 6.125,
            z: -1.5,
        };
        let s = Ipoint {
            x: 1.,
            y: 2.,
            z: 0.5,
        };
        p!(imodel_point_dist(&a, &b));
        p!(imod_point_distance(&a, &b));
        p!(imod_point3d_scale_distance(&a, &b, &s));
        p!(imod_point_line_distance(&a, &b));
        p!(imod_point2d_angle(&a));
        p!(imod_point2d_angle(&b));
        let z0 = Ipoint {
            x: 0.,
            y: 5.,
            z: 0.,
        };
        let z1 = Ipoint::default();
        p!(imod_point2d_angle(&z0));
        p!(imod_point2d_angle(&z1));
        p!(imod_point_dot(&a, &b));
        let mut r = Ipoint::default();
        imod_point_cross(&a, &b, &mut r);
        p3!(r);
        r = a;
        imod_point_normalize(&mut r);
        p3!(r);
        r = Ipoint::default();
        imod_point_normalize(&mut r);
        p3!(r);
        out.push_str(&format!(
            "{} {}\n",
            imod_point_is_equal(&a, &b),
            imod_point_is_equal(&a, &a)
        ));
        p!(imod_point_area(&a, &b, &s));
        p!(imod_point_area_scale(&a, &b, &s, &s));

        {
            let lp1 = Ipoint::default();
            let mut lp2 = Ipoint {
                x: 10.,
                y: 0.,
                z: 0.,
            };
            let mut pnt = Ipoint {
                x: 3.,
                y: 4.,
                z: 5.,
            };
            let mut t = 0.0f32;
            p!(imod_point_line_seg_distance(&lp1, &lp2, &pnt, &mut t));
            p!(t);
            pnt.x = -5.;
            p!(imod_point_line_seg_distance(&lp1, &lp2, &pnt, &mut t));
            p!(t);
            pnt.x = 25.;
            p!(imod_point_line_seg_distance(&lp1, &lp2, &pnt, &mut t));
            p!(t);
            lp2 = lp1;
            p!(imod_point_line_seg_distance(&lp1, &lp2, &pnt, &mut t));
            p!(t);
        }
        {
            let aa = Ipoint::default();
            let bb = Ipoint {
                x: 10.,
                y: 10.,
                z: 0.,
            };
            let mut cc = Ipoint {
                x: 0.,
                y: 10.,
                z: 0.,
            };
            let mut dd = Ipoint {
                x: 10.,
                y: 0.,
                z: 0.,
            };
            out.push_str(&format!("{}\n", imod_point_intersect(&aa, &bb, &cc, &dd)));
            cc.x = 20.;
            cc.y = 20.;
            dd.x = 30.;
            dd.y = 30.;
            out.push_str(&format!("{}\n", imod_point_intersect(&aa, &bb, &cc, &dd)));
            cc.x = 5.;
            cc.y = 5.;
            dd.x = 15.;
            dd.y = 15.;
            out.push_str(&format!("{}\n", imod_point_intersect(&aa, &bb, &cc, &dd)));
            cc.x = 0.;
            cc.y = 1.;
            dd.x = 10.;
            dd.y = 11.;
            out.push_str(&format!("{}\n", imod_point_intersect(&aa, &bb, &cc, &dd)));
        }
        {
            let sq = [0., 0., 10., 0., 10., 10., 0., 10.];
            let tri = [0.5, 0.25, 13.7, 2.2, 6.1, 11.9];
            let mut cs = mkcont(&sq, 3.);
            let ct = mkcont(&tri, -2.5);
            let mut pnt = Ipoint {
                x: 5.,
                y: 5.,
                z: 0.,
            };
            let mut cl = 0i32;
            out.push_str(&format!(
                "{} {}\n",
                imod_point_inside_cont(&cs, &pnt),
                imod_point_inside_cont(&ct, &pnt)
            ));
            pnt.x = -1.;
            out.push_str(&format!(
                "{} {}\n",
                imod_point_inside_cont(&cs, &pnt),
                imod_point_inside_cont(&ct, &pnt)
            ));
            pnt.x = 10.;
            pnt.y = 10.;
            out.push_str(&format!("{}\n", imod_point_inside_cont(&cs, &pnt)));
            pnt.x = 3.;
            pnt.y = 7.;
            pnt.z = 1.;
            p!(imod_point_cont_distance(&cs, &pnt, 0, 0, &mut cl));
            out.push_str(&format!("{cl}\n"));
            p!(imod_point_cont_distance(&cs, &pnt, 1, 0, &mut cl));
            out.push_str(&format!("{cl}\n"));
            p!(imod_point_cont_distance(&cs, &pnt, 0, 1, &mut cl));
            out.push_str(&format!("{cl}\n"));
            p!(imod_point_cont_distance(&ct, &pnt, 1, 1, &mut cl));
            out.push_str(&format!("{cl}\n"));

            imod_point_set_size(&mut cs, 2, 4.5);
            for i in 0..cs.pts.len() {
                p!(cs.sizes[i]);
            }
            let mut obj = Iobj::default();
            obj.pdrawsize = 7;
            p!(imod_point_get_size(&obj, &cs, 2));
            p!(imod_point_get_size(&obj, &cs, 1));
            p!(imod_point_get_size(&obj, &ct, 0));

            pnt.x = 99.;
            pnt.y = 98.;
            pnt.z = 3.;
            out.push_str(&format!("{}\n", imod_point_add(&mut cs, Some(pnt), 1)));
            for i in 0..cs.pts.len() {
                p3!(cs.pts[i]);
                p!(cs.sizes[i]);
            }
            out.push_str(&format!("{}\n", imod_point_delete(&mut cs, 0)));
            for i in 0..cs.pts.len() {
                p3!(cs.pts[i]);
                p!(cs.sizes[i]);
            }
            out.push_str(&format!(
                "{}\n",
                imod_point_append_xyz(&mut cs, 1.5, 2.5, 9.5)
            ));
            out.push_str(&format!("{}\n", cs.flags));
            p3!(cs.pts[cs.pts.len() - 1]);
        }
        {
            let sq = [0., 0., 10., 0., 10., 10., 0., 10.];
            let sq2 = [20., 20., 30., 20., 30., 30., 20., 30.];
            let mut obj = Iobj::default();
            obj.cont.push(mkcont(&sq, 3.));
            obj.cont.push(mkcont(&sq2, 5.));
            let mut list = [0i32; 10];
            let mut nlist = 0i32;
            out.push_str(&format!(
                "{}\n",
                make_area_cont_list(&obj, 4, &mut list, &mut nlist, 10)
            ));
            out.push_str(&format!("nlist {nlist}"));
            for i in 0..nlist as usize {
                out.push_str(&format!(" {}", list[i]));
            }
            out.push('\n');
            out.push_str(&format!(
                "{}\n",
                make_area_cont_list(&obj, 5, &mut list, &mut nlist, 10)
            ));
            out.push_str(&format!("nlist {nlist}"));
            for i in 0..nlist as usize {
                out.push_str(&format!(" {}", list[i]));
            }
            out.push('\n');
            out.push_str(&format!(
                "{}\n",
                imod_point_inside_area(&obj, &list, nlist, 25., 25.)
            ));
            let l2 = [0i32, 1i32];
            out.push_str(&format!(
                "{}\n",
                imod_point_inside_area(&obj, &l2, 2, 5., 5.)
            ));
            out.push_str(&format!(
                "{}\n",
                imod_point_inside_area(&obj, &l2, 2, -5., -5.)
            ));
            let _ = imod_contour_z_value(Some(&obj.cont[0]));
        }
        {
            let mut pls = [Iplane::default(); 2];
            imod_plane_init(&mut pls[0]);
            pls[0].c = -1.;
            pls[0].d = 5.;
            let mut rpt = Ipoint::default();
            let pt1 = Ipoint::default();
            let mut pt2 = Ipoint {
                x: 12.,
                y: 5.,
                z: 20.,
            };
            out.push_str(&format!(
                "{}\n",
                imod_point_plane_edge(&mut rpt, &pls, 1, &pt1, &pt2)
            ));
            p3!(rpt);
            out.push_str(&format!(
                "{}\n",
                imod_point_plane_edge(&mut rpt, &pls, 1, &pt2, &pt1)
            ));
            p3!(rpt);
            pt2 = Ipoint::default();
            out.push_str(&format!(
                "{}\n",
                imod_point_plane_edge(&mut rpt, &pls, 1, &pt1, &pt2)
            ));
        }

        let want = EXPECTED_C_DRIVER_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from ipoint.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
    }

    /// Verbatim stdout of the driver linked against `IMOD/libimod/ipoint.c`.
    const EXPECTED_C_DRIVER_OUTPUT: &str = r#"--- ipoint ---
10.4230814
10.4230814
19.852581
5.61681795
-1.22777238
-1.1487913
1.57079633
0
-30.875
-19.25 -9.125 -1.96875
0.228934273 -0.641015947 0.732589662
0 0 0
0 1
11.7805147
19.9206181
41
0.300000012
66
0
266
1
666
0
1
0
1
0
1 1
0 0
1
3
2
3
2
3.60555124
2
6.48739243
1
-1
-1
4.5
-1
4.5
7
7
5
0 0 3
-1
99 98 3
-1
10 0 3
-1
10 10 3
4.5
0 10 3
-1
4
99 98 3
-1
10 0 3
-1
10 10 3
4.5
0 10 3
-1
5
16
1.5 2.5 9.5
0
nlist 1 0
0
nlist 1 1
1
0
-1
0
0 0 0
0
0 0 0
-1
"#;
}
