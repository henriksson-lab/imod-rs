//! Mechanical-first translation of `IMOD/libwarp/nnpi.c`; reconciled with the shared libwarp layouts.
#![allow(dead_code, unused_mut, unused_assignments)]

use crate::imod::libwarp::delaunay::{
    Circle, Delaunay, Triangle, TriangleNeighbours, delaunay_build, delaunay_circles_find,
    delaunay_destroy, delaunay_xytoi,
};
use crate::imod::libwarp::hash::{
    Hashtable, ht_create_d2, ht_create_i2, ht_delete, ht_destroy, ht_find, ht_getnentries,
    ht_insert, ht_process,
};
use crate::imod::libwarp::istack::{Istack, istack_create, istack_destroy, istack_push};
use crate::imod::libwarp::nn::{NN_RULE, NN_TEST_VERTICE, NN_VERBOSE, NnRule, Point};
use crate::imod::libwarp::nnai::NnWeights;
use crate::imod::libwarp::nncommon::nn_quit;
use crate::imod::libwarp::nncommon_vulnerable::circle_build2;
use core::ffi::c_void;
use libc::{calloc, fprintf, free, malloc, memcpy, qsort, rand, realloc};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
    fn fabs(value: f64) -> f64;
}

pub struct Nnpi {
    pub d: *mut Delaunay,
    pub wmin: f64,
    pub n: i32,
    pub ncircles: i32,
    pub nvertices: i32,
    pub nallocated: i32,
    pub vertices: *mut i32,
    pub weights: *mut f64,
    pub dx: f64,
    pub dy: f64,
    pub bad: *mut Hashtable,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct IndexedValue {
    pub v: *mut f64,
    pub i: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct IndexedPoint {
    pub p0: *mut Point,
    pub p1: *mut Point,
    pub p: *mut Point,
    pub i: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Nnhpi {
    pub nnpi: *mut Nnpi,
    pub ht_data: *mut Hashtable,
    pub ht_weights: *mut Hashtable,
    pub n: i32,
}
pub const NULL: *mut core::ffi::c_void = ::core::ptr::null_mut::<core::ffi::c_void>();
pub const RAND_MAX: i32 = 2147483647 as i32;
pub const __ASSERT_FUNCTION: [core::ffi::c_char; 41] = unsafe {
    ::core::mem::transmute::<[u8; 41], [core::ffi::c_char; 41]>(
        *b"void nnhpi_modify_data(Nnhpi *, Point *)\0",
    )
};
static mut NaN: f64 = f64::NAN;
pub const NSTART: i32 = 10 as i32;
pub const NINC: i32 = 10 as i32;
pub const EPS_SHIFT: f64 = 1.0e-5f64;
pub const BIGNUMBER: f64 = 1.0e+100f64;
pub const EPS_WMIN: f64 = 1.0e-6f64;
pub const HT_SIZE: i32 = 100 as i32;
pub const EPS_SAME: f64 = 1.0e-8f64;
pub unsafe extern "C" fn nnpi_create(mut d: *mut Delaunay) -> *mut Nnpi {
    let mut nn: *mut Nnpi = malloc(::core::mem::size_of::<Nnpi>() as usize) as *mut Nnpi;
    (*nn).d = d;
    (*nn).wmin = -f64::MAX;
    (*nn).n = 0 as i32;
    (*nn).ncircles = 0 as i32;
    (*nn).vertices = calloc(NSTART as usize, ::core::mem::size_of::<i32>() as usize) as *mut i32;
    (*nn).weights = calloc(NSTART as usize, ::core::mem::size_of::<f64>() as usize) as *mut f64;
    (*nn).nvertices = 0 as i32;
    (*nn).nallocated = NSTART;
    (*nn).bad = ::core::ptr::null_mut::<Hashtable>();
    return nn;
}
pub unsafe extern "C" fn nnpi_destroy(mut nn: *mut Nnpi) {
    free((*nn).weights as *mut core::ffi::c_void);
    free((*nn).vertices as *mut core::ffi::c_void);
    free(nn as *mut core::ffi::c_void);
}
pub unsafe extern "C" fn nnpi_reset(mut nn: *mut Nnpi) {
    (*nn).nvertices = 0 as i32;
    (*nn).ncircles = 0 as i32;
    if !(*nn).bad.is_null() {
        ht_destroy((*nn).bad);
        (*nn).bad = ::core::ptr::null_mut::<Hashtable>();
    }
}
unsafe extern "C" fn nnpi_add_weight(mut nn: *mut Nnpi, mut vertex: i32, mut w: f64) {
    let mut i: i32 = 0;
    i = 0 as i32;
    while i < (*nn).nvertices {
        if *(*nn).vertices.offset(i as isize) == vertex {
            break;
        }
        i += 1;
    }
    if i == (*nn).nvertices {
        if (*nn).nvertices == (*nn).nallocated {
            (*nn).vertices = realloc(
                (*nn).vertices as *mut core::ffi::c_void,
                (((*nn).nallocated + NINC) as usize)
                    .wrapping_mul(::core::mem::size_of::<i32>() as usize),
            ) as *mut i32;
            (*nn).weights = realloc(
                (*nn).weights as *mut core::ffi::c_void,
                (((*nn).nallocated + NINC) as usize)
                    .wrapping_mul(::core::mem::size_of::<f64>() as usize),
            ) as *mut f64;
            (*nn).nallocated += NINC;
        }
        *(*nn).vertices.offset(i as isize) = vertex;
        *(*nn).weights.offset(i as isize) = w;
        (*nn).nvertices += 1;
    } else {
        *(*nn).weights.offset(i as isize) += w;
    };
}
unsafe extern "C" fn nnpi_triangle_process(mut nn: *mut Nnpi, mut p: *mut Point, mut i: i32) {
    let mut d: *mut Delaunay = (*nn).d;
    let mut t: *mut Triangle = (*d).triangles.offset(i as isize) as *mut Triangle;
    let mut c: *mut Circle = (*d).circles.offset(i as isize) as *mut Circle;
    let mut cs: [Circle; 3] = [Circle {
        x: 0.,
        y: 0.,
        r: 0.,
    }; 3];
    let mut j: i32 = 0;
    j = 0 as i32;
    while j < 3 as i32 {
        let mut j1: i32 = (j + 1 as i32) % 3 as i32;
        let mut j2: i32 = (j + 2 as i32) % 3 as i32;
        let mut v1: i32 = (*t).vids[j1 as usize];
        let mut v2: i32 = (*t).vids[j2 as usize];
        if circle_build2(
            (&raw mut cs as *mut Circle).offset(j as isize) as *mut Circle,
            (*d).points.offset(v1 as isize) as *mut Point,
            (*d).points.offset(v2 as isize) as *mut Point,
            p,
        ) == 0
        {
            let mut p1: *mut Point = (*d).points.offset(v1 as isize) as *mut Point;
            let mut p2: *mut Point = (*d).points.offset(v2 as isize) as *mut Point;
            if (fabs((*p1).x - (*p).x) + fabs((*p1).y - (*p).y)) / (*c).r < EPS_SAME {
                nnpi_add_weight(nn, v1, BIGNUMBER);
                return;
            } else if (fabs((*p2).x - (*p).x) + fabs((*p2).y - (*p).y)) / (*c).r < EPS_SAME {
                nnpi_add_weight(nn, v2, BIGNUMBER);
                return;
            }
        }
        j += 1;
    }
    j = 0 as i32;
    while j < 3 as i32 {
        let mut j1_0: i32 = (j + 1 as i32) % 3 as i32;
        let mut j2_0: i32 = (j + 2 as i32) % 3 as i32;
        let mut det: f64 = (cs[j1_0 as usize].x - (*c).x) * (cs[j2_0 as usize].y - (*c).y)
            - (cs[j2_0 as usize].x - (*c).x) * (cs[j1_0 as usize].y - (*c).y);
        if det.is_nan() as i32 != 0 {
            let mut j1bad: i32 = cs[j1_0 as usize].x.is_nan() as i32;
            let mut key: [i32; 2] = [0; 2];
            let mut v: *mut f64 = ::core::ptr::null_mut::<f64>();
            key[0 as i32 as usize] = (*t).vids[j as usize];
            if (*nn).bad.is_null() {
                (*nn).bad = ht_create_i2(HT_SIZE);
            }
            key[1 as i32 as usize] = if j1bad != 0 {
                (*t).vids[j2_0 as usize]
            } else {
                (*t).vids[j1_0 as usize]
            };
            v = ht_find((*nn).bad, &raw mut key as *mut core::ffi::c_void) as *mut f64;
            if v.is_null() {
                v = malloc((8 as usize).wrapping_mul(::core::mem::size_of::<f64>() as usize))
                    as *mut f64;
                if j1bad != 0 {
                    *v.offset(0 as i32 as isize) = cs[j2_0 as usize].x;
                    *v.offset(1 as i32 as isize) = cs[j2_0 as usize].y;
                } else {
                    *v.offset(0 as i32 as isize) = cs[j1_0 as usize].x;
                    *v.offset(1 as i32 as isize) = cs[j1_0 as usize].y;
                }
                *v.offset(2 as i32 as isize) = (*c).x;
                *v.offset(3 as i32 as isize) = (*c).y;
                ht_insert(
                    (*nn).bad,
                    &raw mut key as *mut core::ffi::c_void,
                    v as *mut core::ffi::c_void,
                );
                det = 0.0f64;
            } else {
                let mut ii: i32 = 0;
                if j1bad != 0 {
                    *v.offset(6 as i32 as isize) = cs[j2_0 as usize].x;
                    *v.offset(7 as i32 as isize) = cs[j2_0 as usize].y;
                } else {
                    *v.offset(6 as i32 as isize) = cs[j1_0 as usize].x;
                    *v.offset(7 as i32 as isize) = cs[j1_0 as usize].y;
                }
                *v.offset(4 as i32 as isize) = (*c).x;
                *v.offset(5 as i32 as isize) = (*c).y;
                det = 0 as i32 as f64;
                ii = 0 as i32;
                while ii < 4 as i32 {
                    let mut ii1: i32 = (ii + 1 as i32) % 4 as i32;
                    det += (*v.offset((ii * 2 as i32) as isize)
                        + *v.offset((ii1 * 2 as i32) as isize))
                        * (*v.offset((ii * 2 as i32 + 1 as i32) as isize)
                            - *v.offset((ii1 * 2 as i32 + 1 as i32) as isize));
                    ii += 1;
                }
                det = fabs(det);
                free(v as *mut core::ffi::c_void);
                ht_delete((*nn).bad, &raw mut key as *mut core::ffi::c_void);
            }
        }
        nnpi_add_weight(nn, (*t).vids[j as usize], det);
        j += 1;
    }
}
unsafe extern "C" fn compare_int(
    mut p1: *const core::ffi::c_void,
    mut p2: *const core::ffi::c_void,
) -> i32 {
    let mut v1: *mut i32 = p1 as *mut i32;
    let mut v2: *mut i32 = p2 as *mut i32;
    if *v1 > *v2 {
        return 1 as i32;
    } else if *v1 < *v2 {
        return -(1 as i32);
    } else {
        return 0 as i32;
    };
}
unsafe extern "C" fn onleftside(mut p: *mut Point, mut p0: *mut Point, mut p1: *mut Point) -> i32 {
    return (((*p0).x - (*p).x) * ((*p1).y - (*p).y) > ((*p1).x - (*p).x) * ((*p0).y - (*p).y))
        as i32;
}
unsafe extern "C" fn compare_indexedpoints(
    mut pp1: *const core::ffi::c_void,
    mut pp2: *const core::ffi::c_void,
) -> i32 {
    let mut ip1: *mut IndexedPoint = pp1 as *mut IndexedPoint;
    let mut ip2: *mut IndexedPoint = pp2 as *mut IndexedPoint;
    let mut p0: *mut Point = (*ip1).p0;
    let mut p1: *mut Point = (*ip1).p1;
    let mut a: *mut Point = (*ip1).p;
    let mut b: *mut Point = (*ip2).p;
    if onleftside(a, p0, b) != 0 {
        if onleftside(a, p0, p1) != 0 && onleftside(b, p0, p1) == 0 {
            return 1 as i32;
        } else {
            return -(1 as i32);
        }
    } else if onleftside(b, p0, p1) != 0 && onleftside(a, p0, p1) == 0 {
        return -(1 as i32);
    } else {
        return 1 as i32;
    };
}
unsafe extern "C" fn nnpi_getneighbours(
    mut nn: *mut Nnpi,
    mut p: *mut Point,
    mut nt: i32,
    mut tids: *mut i32,
    mut n: *mut i32,
    mut nids: *mut *mut i32,
) {
    let mut d: *mut Delaunay = (*nn).d;
    let mut neighbours: *mut Istack = istack_create();
    let mut v: *mut IndexedPoint = ::core::ptr::null_mut::<IndexedPoint>();
    let mut i: i32 = 0;
    i = 0 as i32;
    while i < nt {
        let mut t: *mut Triangle =
            (*d).triangles.offset(*tids.offset(i as isize) as isize) as *mut Triangle;
        istack_push(neighbours, (*t).vids[0 as i32 as usize]);
        istack_push(neighbours, (*t).vids[1 as i32 as usize]);
        istack_push(neighbours, (*t).vids[2 as i32 as usize]);
        i += 1;
    }
    qsort(
        (*neighbours).v as *mut core::ffi::c_void,
        (*neighbours).n as usize,
        ::core::mem::size_of::<i32>() as usize,
        Some(
            compare_int
                as unsafe extern "C" fn(*const core::ffi::c_void, *const core::ffi::c_void) -> i32,
        ),
    );
    v = malloc(
        (::core::mem::size_of::<IndexedPoint>() as usize).wrapping_mul((*neighbours).n as usize),
    ) as *mut IndexedPoint;
    let ref mut fresh1 = (*v.offset(0 as i32 as isize)).p;
    *fresh1 = (*d)
        .points
        .offset(*(*neighbours).v.offset(0 as i32 as isize) as isize) as *mut Point;
    (*v.offset(0 as i32 as isize)).i = *(*neighbours).v.offset(0 as i32 as isize);
    *n = 1 as i32;
    i = 1 as i32;
    while i < (*neighbours).n {
        if !(*(*neighbours).v.offset(i as isize)
            == *(*neighbours).v.offset((i - 1 as i32) as isize))
        {
            let ref mut fresh2 = (*v.offset(*n as isize)).p;
            *fresh2 = (*d)
                .points
                .offset(*(*neighbours).v.offset(i as isize) as isize)
                as *mut Point;
            (*v.offset(*n as isize)).i = *(*neighbours).v.offset(i as isize);
            *n += 1;
        }
        i += 1;
    }
    if *n > 3 as i32 {
        let ref mut fresh3 = (*v.offset(0 as i32 as isize)).p0;
        *fresh3 = ::core::ptr::null_mut::<Point>();
        let ref mut fresh4 = (*v.offset(0 as i32 as isize)).p1;
        *fresh4 = ::core::ptr::null_mut::<Point>();
        i = 1 as i32;
        while i < *n {
            let ref mut fresh5 = (*v.offset(i as isize)).p0;
            *fresh5 = p;
            let ref mut fresh6 = (*v.offset(i as isize)).p1;
            *fresh6 = (*v.offset(0 as i32 as isize)).p;
            i += 1;
        }
        qsort(
            v.offset(1 as i32 as isize) as *mut IndexedPoint as *mut core::ffi::c_void,
            (*n - 1 as i32) as usize,
            ::core::mem::size_of::<IndexedPoint>() as usize,
            Some(
                compare_indexedpoints
                    as unsafe extern "C" fn(
                        *const core::ffi::c_void,
                        *const core::ffi::c_void,
                    ) -> i32,
            ),
        );
    }
    *nids = malloc((*n as usize).wrapping_mul(::core::mem::size_of::<i32>() as usize)) as *mut i32;
    i = 0 as i32;
    while i < *n {
        *(*nids).offset(i as isize) = (*v.offset(i as isize)).i;
        i += 1;
    }
    istack_destroy(neighbours);
    free(v as *mut core::ffi::c_void);
}
unsafe extern "C" fn nnpi_neighbours_process(
    mut nn: *mut Nnpi,
    mut p: *mut Point,
    mut n: i32,
    mut nids: *mut i32,
) -> i32 {
    let mut d: *mut Delaunay = (*nn).d;
    let mut i: i32 = 0;
    i = 0 as i32;
    while i < n {
        let mut im1: i32 = (i + n - 1 as i32) % n;
        let mut ip1: i32 = (i + 1 as i32) % n;
        let mut p0: *mut Point =
            (*d).points.offset(*nids.offset(i as isize) as isize) as *mut Point;
        let mut pp1: *mut Point =
            (*d).points.offset(*nids.offset(ip1 as isize) as isize) as *mut Point;
        let mut pm1: *mut Point =
            (*d).points.offset(*nids.offset(im1 as isize) as isize) as *mut Point;
        let mut nom1: f64 = 0.;
        let mut nom2: f64 = 0.;
        let mut denom1: f64 = 0.;
        let mut denom2: f64 = 0.;
        denom1 =
            ((*p0).x - (*p).x) * ((*pp1).y - (*p).y) - ((*p0).y - (*p).y) * ((*pp1).x - (*p).x);
        denom2 =
            ((*p0).x - (*p).x) * ((*pm1).y - (*p).y) - ((*p0).y - (*p).y) * ((*pm1).x - (*p).x);
        if denom1 == 0.0f64 {
            if (*p).x == (*p0).x && (*p).y == (*p0).y {
                nnpi_add_weight(nn, *nids.offset(i as isize), BIGNUMBER);
                return 1 as i32;
            } else if (*p).x == (*pp1).x && (*p).y == (*pp1).y {
                nnpi_add_weight(nn, *nids.offset(ip1 as isize), BIGNUMBER);
                return 1 as i32;
            } else {
                (*nn).dx = EPS_SHIFT * ((*pp1).y - (*p0).y);
                (*nn).dy = -EPS_SHIFT * ((*pp1).x - (*p0).x);
                return 0 as i32;
            }
        }
        if denom2 == 0.0f64 {
            if (*p).x == (*pm1).x && (*p).y == (*pm1).y {
                nnpi_add_weight(nn, *nids.offset(im1 as isize), BIGNUMBER);
                return 1 as i32;
            } else {
                (*nn).dx = EPS_SHIFT * ((*pm1).y - (*p0).y);
                (*nn).dy = -EPS_SHIFT * ((*pm1).x - (*p0).x);
                return 0 as i32;
            }
        }
        nom1 =
            ((*p0).x - (*pp1).x) * ((*pp1).x - (*p).x) + ((*p0).y - (*pp1).y) * ((*pp1).y - (*p).y);
        nom2 =
            ((*p0).x - (*pm1).x) * ((*pm1).x - (*p).x) + ((*p0).y - (*pm1).y) * ((*pm1).y - (*p).y);
        nnpi_add_weight(nn, *nids.offset(i as isize), nom1 / denom1 - nom2 / denom2);
        i += 1;
    }
    return 1 as i32;
}
unsafe extern "C" fn _nnpi_calculate_weights(mut nn: *mut Nnpi, mut p: *mut Point) -> i32 {
    let mut tids: *mut i32 = ::core::ptr::null_mut::<i32>();
    let mut i: i32 = 0;
    delaunay_circles_find((*nn).d, p, &raw mut (*nn).ncircles, &raw mut tids);
    if (*nn).ncircles == 0 as i32 {
        return 1 as i32;
    }
    if NN_RULE == NnRule::Sibson {
        i = 0 as i32;
        while i < (*nn).ncircles {
            nnpi_triangle_process(nn, p, *tids.offset(i as isize));
            i += 1;
        }
        if !(*nn).bad.is_null() {
            let mut nentries: i32 = ht_getnentries((*nn).bad);
            if nentries > 0 as i32 {
                ht_process(
                    (*nn).bad,
                    free as unsafe extern "C" fn(*mut core::ffi::c_void) -> (),
                );
                return 0 as i32;
            }
        }
        return 1 as i32;
    } else if NN_RULE == NnRule::NonSibsonian {
        let mut nneigh: i32 = 0 as i32;
        let mut nids: *mut i32 = ::core::ptr::null_mut::<i32>();
        let mut status: i32 = 0;
        nnpi_getneighbours(nn, p, (*nn).ncircles, tids, &raw mut nneigh, &raw mut nids);
        status = nnpi_neighbours_process(nn, p, nneigh, nids);
        free(nids as *mut core::ffi::c_void);
        return status;
    } else {
        nn_quit(
            b"programming error\0" as *const u8 as *const core::ffi::c_char
                as *mut core::ffi::c_char,
        );
    }
    return 0 as i32;
}
unsafe extern "C" fn nnpi_normalize_weights(mut nn: *mut Nnpi) {
    let mut n: i32 = (*nn).nvertices;
    let mut sum: f64 = 0.0f64;
    let mut i: i32 = 0;
    i = 0 as i32;
    while i < n {
        sum += *(*nn).weights.offset(i as isize);
        i += 1;
    }
    i = 0 as i32;
    while i < n {
        *(*nn).weights.offset(i as isize) /= sum;
        i += 1;
    }
}
pub unsafe extern "C" fn nnpi_calculate_weights(mut nn: *mut Nnpi, mut p: *mut Point) {
    let mut pp: Point = Point {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let mut nvertices: i32 = 0 as i32;
    let mut vertices: *mut i32 = ::core::ptr::null_mut::<i32>();
    let mut weights: *mut f64 = ::core::ptr::null_mut::<f64>();
    let mut i: i32 = 0;
    nnpi_reset(nn);
    if _nnpi_calculate_weights(nn, p) != 0 {
        nnpi_normalize_weights(nn);
        return;
    }
    nnpi_reset(nn);
    (*nn).dx = ((*(*nn).d).xmax - (*(*nn).d).xmin) * EPS_SHIFT;
    (*nn).dy = ((*(*nn).d).ymax - (*(*nn).d).ymin) * EPS_SHIFT;
    pp.x = (*p).x + (*nn).dx;
    pp.y = (*p).y + (*nn).dy;
    while _nnpi_calculate_weights(nn, &raw mut pp) == 0 {
        nnpi_reset(nn);
        pp.x = (*p).x + (*nn).dx * rand() as f64 / (RAND_MAX as f64 + 1.0f64);
        pp.y = (*p).y + (*nn).dy * rand() as f64 / (RAND_MAX as f64 + 1.0f64);
    }
    nnpi_normalize_weights(nn);
    nvertices = (*nn).nvertices;
    if nvertices > 0 as i32 {
        vertices = malloc((nvertices as usize).wrapping_mul(::core::mem::size_of::<i32>() as usize))
            as *mut i32;
        memcpy(
            vertices as *mut core::ffi::c_void,
            (*nn).vertices as *const core::ffi::c_void,
            (nvertices as usize).wrapping_mul(::core::mem::size_of::<i32>() as usize),
        );
        weights = malloc((nvertices as usize).wrapping_mul(::core::mem::size_of::<f64>() as usize))
            as *mut f64;
        memcpy(
            weights as *mut core::ffi::c_void,
            (*nn).weights as *const core::ffi::c_void,
            (nvertices as usize).wrapping_mul(::core::mem::size_of::<f64>() as usize),
        );
    }
    nnpi_reset(nn);
    pp.x = 2.0f64 * (*p).x - pp.x;
    pp.y = 2.0f64 * (*p).y - pp.y;
    while _nnpi_calculate_weights(nn, &raw mut pp) == 0 || (*nn).nvertices == 0 as i32 {
        nnpi_reset(nn);
        pp.x = (*p).x + (*nn).dx * rand() as f64 / (RAND_MAX as f64 + 1.0f64);
        pp.y = (*p).y + (*nn).dy * rand() as f64 / (RAND_MAX as f64 + 1.0f64);
    }
    nnpi_normalize_weights(nn);
    if nvertices > 0 as i32 {
        i = 0 as i32;
        while i < (*nn).nvertices {
            *(*nn).weights.offset(i as isize) /= 2.0f64;
            i += 1;
        }
    }
    i = 0 as i32;
    while i < nvertices {
        nnpi_add_weight(
            nn,
            *vertices.offset(i as isize),
            *weights.offset(i as isize) / 2.0f64,
        );
        i += 1;
    }
    if nvertices > 0 as i32 {
        free(vertices as *mut core::ffi::c_void);
        free(weights as *mut core::ffi::c_void);
    }
}
unsafe extern "C" fn cmp_iv(
    mut p1: *const core::ffi::c_void,
    mut p2: *const core::ffi::c_void,
) -> i32 {
    let mut v1: f64 = *(*(p1 as *mut IndexedValue)).v;
    let mut v2: f64 = *(*(p2 as *mut IndexedValue)).v;
    if v1 > v2 {
        return -(1 as i32);
    }
    if v1 < v2 {
        return 1 as i32;
    }
    return 0 as i32;
}
pub unsafe extern "C" fn nnpi_interpolate_point(mut nn: *mut Nnpi, mut p: *mut Point) {
    let nn_verbose = NN_VERBOSE;
    let nn_test_vertice = NN_TEST_VERTICE;
    let mut d: *mut Delaunay = (*nn).d;
    let mut i: i32 = 0;
    nnpi_calculate_weights(nn, p);
    if nn_verbose != 0 {
        if nn_test_vertice == -(1 as i32) {
            let mut ivs: *mut IndexedValue = ::core::ptr::null_mut::<IndexedValue>();
            if (*nn).nvertices > 0 as i32 {
                ivs = malloc(
                    ((*nn).nvertices as usize)
                        .wrapping_mul(::core::mem::size_of::<IndexedValue>() as usize),
                ) as *mut IndexedValue;
                i = 0 as i32;
                while i < (*nn).nvertices {
                    (*ivs.offset(i as isize)).i = *(*nn).vertices.offset(i as isize);
                    let ref mut fresh0 = (*ivs.offset(i as isize)).v;
                    *fresh0 = (*nn).weights.offset(i as isize) as *mut f64;
                    i += 1;
                }
                qsort(
                    ivs as *mut core::ffi::c_void,
                    (*nn).nvertices as usize,
                    ::core::mem::size_of::<IndexedValue>() as usize,
                    Some(
                        cmp_iv
                            as unsafe extern "C" fn(
                                *const core::ffi::c_void,
                                *const core::ffi::c_void,
                            ) -> i32,
                    ),
                );
            }
            if (*nn).n == 0 as i32 {
                fprintf(
                    stderr,
                    b"weights:\n\0" as *const u8 as *const core::ffi::c_char,
                );
            }
            fprintf(
                stderr,
                b"  %d: (%.10g, %10g)\n\0" as *const u8 as *const core::ffi::c_char,
                (*nn).n,
                (*p).x,
                (*p).y,
            );
            fprintf(
                stderr,
                b"  %4s %15s %15s %15s %15s\n\0" as *const u8 as *const core::ffi::c_char,
                b"id\0" as *const u8 as *const core::ffi::c_char,
                b"x\0" as *const u8 as *const core::ffi::c_char,
                b"y\0" as *const u8 as *const core::ffi::c_char,
                b"z\0" as *const u8 as *const core::ffi::c_char,
                b"w\0" as *const u8 as *const core::ffi::c_char,
            );
            i = 0 as i32;
            while i < (*nn).nvertices {
                let mut ii: i32 = (*ivs.offset(i as isize)).i;
                let mut pp: *mut Point = (*d).points.offset(ii as isize) as *mut Point;
                fprintf(
                    stderr,
                    b"  %5d %15.10g %15.10g %15.10g %15f\n\0" as *const u8
                        as *const core::ffi::c_char,
                    ii,
                    (*pp).x,
                    (*pp).y,
                    (*pp).z,
                    *(*ivs.offset(i as isize)).v,
                );
                i += 1;
            }
            if (*nn).nvertices > 0 as i32 {
                free(ivs as *mut core::ffi::c_void);
            }
        } else {
            let mut w: f64 = 0.0f64;
            if (*nn).n == 0 as i32 {
                fprintf(
                    stderr,
                    b"weight of vertex %d:\n\0" as *const u8 as *const core::ffi::c_char,
                    nn_test_vertice,
                );
            }
            i = 0 as i32;
            while i < (*nn).nvertices {
                if *(*nn).vertices.offset(i as isize) == nn_test_vertice {
                    w = *(*nn).weights.offset(i as isize);
                    break;
                } else {
                    i += 1;
                }
            }
            fprintf(
                stderr,
                b"  (%.10g, %.10g): %.7g\n\0" as *const u8 as *const core::ffi::c_char,
                (*p).x,
                (*p).y,
                w,
            );
        }
    }
    (*nn).n += 1;
    if (*nn).nvertices == 0 as i32 {
        (*p).z = NaN;
        return;
    }
    (*p).z = 0.0f64;
    i = 0 as i32;
    while i < (*nn).nvertices {
        let mut weight: f64 = *(*nn).weights.offset(i as isize);
        if weight < (*nn).wmin {
            (*p).z = NaN;
            return;
        }
        (*p).z += (*(*d)
            .points
            .offset(*(*nn).vertices.offset(i as isize) as isize))
        .z * weight;
        i += 1;
    }
}
pub unsafe extern "C" fn nnpi_interpolate_points(
    mut nin: i32,
    mut pin: *mut Point,
    mut wmin: f64,
    mut nout: i32,
    mut pout: *mut Point,
) {
    let nn_verbose = NN_VERBOSE;
    let mut d: *mut Delaunay = delaunay_build(
        nin,
        pin,
        0 as i32,
        ::core::ptr::null_mut::<i32>(),
        0 as i32,
        ::core::ptr::null_mut::<f64>(),
    );
    let mut nn: *mut Nnpi = nnpi_create(d);
    let mut seed: i32 = 0 as i32;
    let mut i: i32 = 0;
    nnpi_setwmin(nn, wmin);
    if nn_verbose != 0 {
        fprintf(
            stderr,
            b"xytoi:\n\0" as *const u8 as *const core::ffi::c_char,
        );
        i = 0 as i32;
        while i < nout {
            let mut p: *mut Point = pout.offset(i as isize) as *mut Point;
            fprintf(
                stderr,
                b"(%.7g,%.7g) -> %d\n\0" as *const u8 as *const core::ffi::c_char,
                (*p).x,
                (*p).y,
                delaunay_xytoi(d, p, seed),
            );
            i += 1;
        }
    }
    i = 0 as i32;
    while i < nout {
        nnpi_interpolate_point(nn, pout.offset(i as isize) as *mut Point);
        i += 1;
    }
    if nn_verbose != 0 {
        fprintf(
            stderr,
            b"output:\n\0" as *const u8 as *const core::ffi::c_char,
        );
        i = 0 as i32;
        while i < nout {
            let mut p_0: *mut Point = pout.offset(i as isize) as *mut Point;
            fprintf(
                stderr,
                b"  %d:%15.7g %15.7g %15.7g\n\0" as *const u8 as *const core::ffi::c_char,
                i,
                (*p_0).x,
                (*p_0).y,
                (*p_0).z,
            );
            i += 1;
        }
    }
    nnpi_destroy(nn);
    delaunay_destroy(d);
}
pub unsafe extern "C" fn nnpi_setwmin(mut nn: *mut Nnpi, mut wmin: f64) {
    (*nn).wmin = if wmin == 0 as i32 as f64 {
        -EPS_WMIN
    } else {
        wmin
    };
}
pub unsafe extern "C" fn nnpi_get_nvertices(mut nn: *mut Nnpi) -> i32 {
    return (*nn).nvertices;
}
pub unsafe extern "C" fn nnpi_get_vertices(mut nn: *mut Nnpi) -> *mut i32 {
    return (*nn).vertices;
}
pub unsafe extern "C" fn nnpi_get_weights(mut nn: *mut Nnpi) -> *mut f64 {
    return (*nn).weights;
}
pub unsafe extern "C" fn nnhpi_create(mut d: *mut Delaunay, mut size: i32) -> *mut Nnhpi {
    let mut nn: *mut Nnhpi = malloc(::core::mem::size_of::<Nnhpi>() as usize) as *mut Nnhpi;
    let mut i: i32 = 0;
    (*nn).nnpi = nnpi_create(d);
    (*nn).ht_data = ht_create_d2((*d).npoints);
    (*nn).ht_weights = ht_create_d2(size);
    (*nn).n = 0 as i32;
    i = 0 as i32;
    while i < (*d).npoints {
        ht_insert(
            (*nn).ht_data,
            (*d).points.offset(i as isize) as *mut Point as *mut core::ffi::c_void,
            (*d).points.offset(i as isize) as *mut Point as *mut core::ffi::c_void,
        );
        i += 1;
    }
    return nn;
}
unsafe extern "C" fn free_nn_weights(mut data: *mut core::ffi::c_void) {
    let mut weights: *mut NnWeights = data as *mut NnWeights;
    free((*weights).vertices as *mut core::ffi::c_void);
    free((*weights).weights as *mut core::ffi::c_void);
    free(weights as *mut core::ffi::c_void);
}
pub unsafe extern "C" fn nnhpi_destroy(mut nn: *mut Nnhpi) {
    ht_destroy((*nn).ht_data);
    ht_process(
        (*nn).ht_weights,
        free_nn_weights as unsafe extern "C" fn(*mut core::ffi::c_void) -> (),
    );
    ht_destroy((*nn).ht_weights);
    nnpi_destroy((*nn).nnpi);
}
pub unsafe extern "C" fn nnhpi_interpolate(mut nnhpi: *mut Nnhpi, mut p: *mut Point) {
    let mut nnpi: *mut Nnpi = (*nnhpi).nnpi;
    let Nnpi = nnpi;
    let Nnhpi = nnhpi;
    let nn_verbose = NN_VERBOSE;
    let nn_test_vertice = NN_TEST_VERTICE;
    let mut d: *mut Delaunay = (*nnpi).d;
    let mut ht_weights: *mut Hashtable = (*nnhpi).ht_weights;
    let mut weights: *mut NnWeights = ::core::ptr::null_mut::<NnWeights>();
    let mut i: i32 = 0;
    if !ht_find(ht_weights, p as *mut core::ffi::c_void).is_null() {
        weights = ht_find(ht_weights, p as *mut core::ffi::c_void) as *mut NnWeights;
        if nn_verbose != 0 {
            fprintf(
                stderr,
                b"  <Hashtable>\n\0" as *const u8 as *const core::ffi::c_char,
            );
        }
    } else {
        nnpi_calculate_weights(nnpi, p);
        weights = malloc(::core::mem::size_of::<NnWeights>() as usize) as *mut NnWeights;
        (*weights).vertices = malloc(
            (::core::mem::size_of::<i32>() as usize).wrapping_mul((*Nnpi).nvertices as usize),
        ) as *mut i32;
        (*weights).weights = malloc(
            (::core::mem::size_of::<f64>() as usize).wrapping_mul((*Nnpi).nvertices as usize),
        ) as *mut f64;
        (*weights).nvertices = (*Nnpi).nvertices;
        i = 0 as i32;
        while i < (*Nnpi).nvertices {
            *(*weights).vertices.offset(i as isize) = *(*Nnpi).vertices.offset(i as isize);
            *(*weights).weights.offset(i as isize) = *(*Nnpi).weights.offset(i as isize);
            i += 1;
        }
        ht_insert(
            ht_weights,
            p as *mut core::ffi::c_void,
            weights as *mut core::ffi::c_void,
        );
        if nn_verbose != 0 {
            if nn_test_vertice == -(1 as i32) {
                if (*Nnpi).n == 0 as i32 {
                    fprintf(
                        stderr,
                        b"weights:\n\0" as *const u8 as *const core::ffi::c_char,
                    );
                }
                fprintf(
                    stderr,
                    b"  %d: {\0" as *const u8 as *const core::ffi::c_char,
                    (*Nnpi).n,
                );
                i = 0 as i32;
                while i < (*Nnpi).nvertices {
                    fprintf(
                        stderr,
                        b"(%d,%.5g)\0" as *const u8 as *const core::ffi::c_char,
                        *(*Nnpi).vertices.offset(i as isize),
                        *(*Nnpi).weights.offset(i as isize),
                    );
                    if i < (*Nnpi).nvertices - 1 as i32 {
                        fprintf(stderr, b", \0" as *const u8 as *const core::ffi::c_char);
                    }
                    i += 1;
                }
                fprintf(stderr, b"}\n\0" as *const u8 as *const core::ffi::c_char);
            } else {
                let mut w: f64 = 0.0f64;
                if (*Nnpi).n == 0 as i32 {
                    fprintf(
                        stderr,
                        b"weights for vertex %d:\n\0" as *const u8 as *const core::ffi::c_char,
                        nn_test_vertice,
                    );
                }
                i = 0 as i32;
                while i < (*Nnpi).nvertices {
                    if *(*Nnpi).vertices.offset(i as isize) == nn_test_vertice {
                        w = *(*Nnpi).weights.offset(i as isize);
                        break;
                    } else {
                        i += 1;
                    }
                }
                fprintf(
                    stderr,
                    b"%15.7g %15.7g %15.7g\n\0" as *const u8 as *const core::ffi::c_char,
                    (*p).x,
                    (*p).y,
                    w,
                );
            }
        }
        (*Nnpi).n += 1;
    }
    (*Nnhpi).n += 1;
    if (*weights).nvertices == 0 as i32 {
        (*p).z = NaN;
        return;
    }
    (*p).z = 0.0f64;
    i = 0 as i32;
    while i < (*weights).nvertices {
        if *(*weights).weights.offset(i as isize) < (*Nnpi).wmin {
            (*p).z = NaN;
            return;
        }
        (*p).z += (*(*d)
            .points
            .offset(*(*weights).vertices.offset(i as isize) as isize))
        .z * *(*weights).weights.offset(i as isize);
        i += 1;
    }
}
pub unsafe extern "C" fn nnhpi_modify_data(mut nnhpi: *mut Nnhpi, mut p: *mut Point) {
    let mut orig: *mut Point = ht_find((*nnhpi).ht_data, p as *mut core::ffi::c_void) as *mut Point;
    '_c2rust_label: {
        if !orig.is_null() {
        } else {
            assert!(!orig.is_null());
        }
    };
    (*orig).z = (*p).z;
}
pub unsafe extern "C" fn nnhpi_setwmin(mut nn: *mut Nnhpi, mut wmin: f64) {
    (*(*nn).nnpi).wmin = wmin;
}
pub const __DBL_MAX__: f64 = 1.7976931348623157e+308f64;
