//! Translation of `IMOD/libwarp/hullwrap.c` and `hullwrap.h`.
#![allow(dead_code)]

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// C `struct hullio` (`hullwrap.h`).
#[repr(C)]
pub struct HullIo {
    pub pointlist: *mut f64,
    pub numberofpoints: i32,
    pub trianglelist: *mut i32,
    pub neighborlist: *mut i32,
    pub numberoftriangles: i32,
    pub height_base_crit: f64,
    pub area_fraction_crit: f64,
    pub min_num_for_pruning: i32,
    pub verbose: i32,
}

/// Original `hull_triangulate` (`hullwrap.c:38`).
pub unsafe fn hull_triangulate(hio: *mut HullIo) -> i32 {
    unsafe {
        if (*hio).verbose != 0 {
            libc::fprintf(
                stderr,
                c"Number of points %d\n".as_ptr(),
                (*hio).numberofpoints,
            );
        }
        POINTS = (*hio).pointlist;
        COUNT = (*hio).numberofpoints;
        NEXT = 0;
        HIO = hio;
        let root = build_convex_hull(get_next_site, site_numm, 2, 1);
        TRI = 0;
        PRUNE = 0;
        TOTAL = 0.;
        PRUNE_CRIT = -1;
        visit_hull(root, mark_outside_get_area);
        if TRI >= (*hio).min_num_for_pruning
            && (*hio).area_fraction_crit > 0.
            && (*hio).height_base_crit > 0.
        {
            visit_hull(root, mark_for_pruning);
            PRUNE_CRIT = 0;
        }
        TRI = 0;
        visit_hull(root, set_triangle_number);
        if (*hio).verbose != 0 {
            libc::fprintf(
                stderr,
                c"%d triangles,  %d pruned at edge, total areax2 %.0f\n".as_ptr(),
                TRI,
                PRUNE,
                TOTAL,
            );
        }
        (*hio).numberoftriangles = TRI;
        (*hio).trianglelist = core::ptr::null_mut();
        (*hio).neighborlist = core::ptr::null_mut();
        if TRI == 0 {
            free_hull_storage();
            return 0;
        }
        (*hio).trianglelist = libc::malloc(3 * TRI as usize * core::mem::size_of::<i32>()).cast();
        (*hio).neighborlist = libc::malloc(3 * TRI as usize * core::mem::size_of::<i32>()).cast();
        if (*hio).trianglelist.is_null() || (*hio).neighborlist.is_null() {
            free_hull_storage();
            return 1;
        }
        LIST = 0;
        visit_hull(root, save_triangle);
        free_hull_storage();
        0
    }
}
/// Original static `markForPruning` (`hullwrap.c:153`).
unsafe fn mark_for_pruning(
    simplex: *mut Simplex,
    _: *mut core::ffi::c_void,
) -> *mut core::ffi::c_void {
    unsafe {
        if simplex.is_null() || (*simplex).mark == 0 {
            return core::ptr::null_mut();
        }
        for index in 0..3 {
            if site_numm((*simplex).neigh.as_mut_ptr().add(index).read().vert) < 0 {
                let neighbor = (*simplex).neigh.as_mut_ptr().add(index).read().simp;
                if !neighbor.is_null() {
                    let first = (index + 1) % 3;
                    let second = (index + 2) % 3;
                    let first_point = (*simplex).neigh.as_mut_ptr().add(first).read().vert;
                    let second_point = (*simplex).neigh.as_mut_ptr().add(second).read().vert;
                    let dx_base = *first_point - *second_point;
                    let dy_base = *first_point.add(1) - *second_point.add(1);
                    let base_square = dx_base * dx_base + dy_base * dy_base;
                    let first_neighbor = (*neighbor).neigh.as_mut_ptr().read().vert;
                    let second_neighbor = (*neighbor).neigh.as_mut_ptr().add(1).read().vert;
                    let third_neighbor = (*neighbor).neigh.as_mut_ptr().add(2).read().vert;
                    let double_area = ((*second_neighbor - *first_neighbor)
                        * (*third_neighbor.add(1) - *first_neighbor.add(1))
                        - (*third_neighbor - *first_neighbor)
                            * (*second_neighbor.add(1) - *first_neighbor.add(1)))
                    .abs();
                    if (*HIO).verbose != 0 {
                        let base = base_square.sqrt();
                        libc::fprintf(
                            stderr,
                            c"edge %.0f,%.0f to %.0f,%.0f, base %.1f, area %.0f, hgt %.1f  h/b %.3f"
                                .as_ptr(),
                            *first_point,
                            *first_point.add(1),
                            *second_point,
                            *second_point.add(1),
                            base,
                            double_area,
                            double_area / base,
                            double_area / base_square,
                        );
                    }
                    if double_area / base_square < (*HIO).height_base_crit
                        && double_area / TOTAL < (*HIO).area_fraction_crit
                    {
                        (*neighbor).mark = -1;
                        PRUNE += 1;
                        if (*HIO).verbose != 0 {
                            libc::fprintf(stderr, c" pruned".as_ptr());
                        }
                    }
                    if (*HIO).verbose != 0 {
                        libc::puts(c" ".as_ptr());
                    }
                }
                return core::ptr::null_mut();
            }
        }
        core::ptr::null_mut()
    }
}

use crate::imod::libwarp::hull::{Simplex, Site, visit_hull};
use crate::imod::libwarp::hull_ch::{HULL_INFINITY, build_convex_hull, free_hull_storage};
static mut POINTS: *mut f64 = core::ptr::null_mut();
static mut COUNT: i32 = 0;
static mut NEXT: i32 = 0;
static mut TRI: i32 = 0;
static mut LIST: i32 = 0;
static mut PRUNE: i32 = 0;
static mut PRUNE_CRIT: i32 = 0;
static mut TOTAL: f64 = 0.;
static mut HIO: *mut HullIo = core::ptr::null_mut();
/// Original static `site_numm` (`hullwrap.c:96`).
unsafe fn site_numm(point: Site) -> i32 {
    unsafe {
        if point == core::ptr::addr_of_mut!(HULL_INFINITY).cast() {
            return -1;
        }
        if point.is_null() {
            return -2;
        }
        let index = point.offset_from(POINTS) / 2;
        if index < 0 || index >= COUNT as isize {
            -3
        } else {
            if (*HIO).verbose > 1 {
                libc::fprintf(
                    stderr,
                    c"site_numm returning %d for %p\n".as_ptr(),
                    index as i32,
                    point,
                );
            }
            index as i32
        }
    }
}
/// Original static `get_next_site` (`hullwrap.c:112`).
unsafe fn get_next_site() -> Site {
    unsafe {
        if NEXT >= COUNT {
            core::ptr::null_mut()
        } else {
            let point = POINTS.add((2 * NEXT) as usize);
            if (*HIO).verbose > 1 {
                libc::fprintf(
                    stderr,
                    c"get_next_site returning %d  %f %f\n".as_ptr(),
                    NEXT,
                    *point,
                    *point.add(1),
                );
            }
            NEXT += 1;
            point
        }
    }
}
/// Original static `markOutsideGetArea` (`hullwrap.c:127`).
unsafe fn mark_outside_get_area(
    s: *mut Simplex,
    _: *mut core::ffi::c_void,
) -> *mut core::ffi::c_void {
    unsafe {
        if s.is_null() {
            return core::ptr::null_mut();
        }
        (*s).mark = -2;
        for i in 0..3 {
            if site_numm((*s).neigh.as_mut_ptr().add(i).read().vert) < 0 {
                return core::ptr::null_mut();
            }
        }
        (*s).mark = 0;
        let a = (*s).neigh.as_mut_ptr().read().vert;
        let b = (*s).neigh.as_mut_ptr().add(1).read().vert;
        let c = (*s).neigh.as_mut_ptr().add(2).read().vert;
        TOTAL += ((*b - *a) * (*c.add(1) - *a.add(1)) - (*c - *a) * (*b.add(1) - *a.add(1))).abs();
        TRI += 1;
        core::ptr::null_mut()
    }
}
/// Original static `setTriangleNumber` (`hullwrap.c:194`).
unsafe fn set_triangle_number(
    s: *mut Simplex,
    _: *mut core::ffi::c_void,
) -> *mut core::ffi::c_void {
    unsafe {
        if !s.is_null() && (*s).mark >= PRUNE_CRIT {
            (*s).mark = TRI;
            TRI += 1;
        }
        core::ptr::null_mut()
    }
}
/// Original static `saveTriangle` (`hullwrap.c:203`).
unsafe fn save_triangle(s: *mut Simplex, _: *mut core::ffi::c_void) -> *mut core::ffi::c_void {
    unsafe {
        if s.is_null() || (*s).mark < PRUNE_CRIT {
            return core::ptr::null_mut();
        }
        let a = (*s).neigh.as_mut_ptr().read().vert;
        let b = (*s).neigh.as_mut_ptr().add(1).read().vert;
        let c = (*s).neigh.as_mut_ptr().add(2).read().vert;
        let reverse =
            (*b - *a) * (*c.add(1) - *a.add(1)) - (*c - *a) * (*b.add(1) - *a.add(1)) < 0.;
        let mut i = if reverse { 2 } else { 0 };
        loop {
            *(*HIO).trianglelist.add(LIST as usize) =
                site_numm((*s).neigh.as_mut_ptr().add(i).read().vert);
            *(*HIO).neighborlist.add(LIST as usize) =
                (*(*s).neigh.as_mut_ptr().add(i).read().simp).mark;
            LIST += 1;
            if (reverse && i == 0) || (!reverse && i == 2) {
                break;
            }
            if reverse { i -= 1 } else { i += 1 }
        }
        core::ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_edge_pruning_removes_a_triangle_when_both_criteria_hold() {
        let mut points = [0., 0., 1., 0., 0., 1.];
        let mut io = HullIo {
            pointlist: points.as_mut_ptr(),
            numberofpoints: 3,
            trianglelist: core::ptr::null_mut(),
            neighborlist: core::ptr::null_mut(),
            numberoftriangles: -1,
            height_base_crit: 10.,
            area_fraction_crit: 2.,
            min_num_for_pruning: 1,
            verbose: 0,
        };
        unsafe {
            assert_eq!(hull_triangulate(&mut io), 0);
        }
        assert_eq!(io.numberoftriangles, 0);
        assert!(io.trianglelist.is_null());
        assert!(io.neighborlist.is_null());
    }
}
