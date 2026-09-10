//! Translation scaffolding for `IMOD/libwarp/delaunay.h`.
#![allow(dead_code)]

use crate::imod::libwarp::hullwrap::{HullIo, hull_triangulate};
use crate::imod::libwarp::istack::Istack;
use crate::imod::libwarp::istack::{
    istack_create, istack_destroy, istack_pop, istack_push, istack_reset,
};
use crate::imod::libwarp::nn::NN_VERBOSE;
use crate::imod::libwarp::nn::Point;
use crate::imod::libwarp::nncommon::circle_contains;
use crate::imod::libwarp::nncommon_vulnerable::circle_build1;

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// C `triangle` (`delaunay.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Triangle {
    pub vids: [i32; 3],
}

/// C `triangle_neighbours` (`delaunay.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriangleNeighbours {
    pub tids: [i32; 3],
}

/// C `circle` (`delaunay.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Circle {
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

/// C `struct delaunay` (`delaunay.h`).
#[repr(C)]
pub struct Delaunay {
    pub npoints: i32,
    pub points: *mut Point,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub ntriangles: i32,
    pub triangles: *mut Triangle,
    pub circles: *mut Circle,
    pub neighbours: *mut TriangleNeighbours,
    pub n_point_triangles: *mut i32,
    pub point_triangles: *mut *mut i32,
    pub nedges: i32,
    pub edges: *mut i32,
    pub flags: *mut i32,
    pub first_id: i32,
    pub t_in: *mut Istack,
    pub t_out: *mut Istack,
    pub nflags: i32,
    pub nflagsallocated: i32,
    pub flagids: *mut i32,
}

/// Original static `delaunay_create` (`IMOD/libwarp/delaunay.c:127`).
unsafe fn delaunay_create() -> *mut Delaunay {
    unsafe {
        let delaunay = libc::malloc(core::mem::size_of::<Delaunay>()).cast::<Delaunay>();
        (*delaunay).npoints = 0;
        (*delaunay).points = core::ptr::null_mut();
        (*delaunay).xmin = f64::MAX;
        (*delaunay).xmax = -f64::MAX;
        (*delaunay).ymin = f64::MAX;
        (*delaunay).ymax = -f64::MAX;
        (*delaunay).ntriangles = 0;
        (*delaunay).triangles = core::ptr::null_mut();
        (*delaunay).circles = core::ptr::null_mut();
        (*delaunay).neighbours = core::ptr::null_mut();
        (*delaunay).n_point_triangles = core::ptr::null_mut();
        (*delaunay).point_triangles = core::ptr::null_mut();
        (*delaunay).nedges = 0;
        (*delaunay).edges = core::ptr::null_mut();
        (*delaunay).flags = core::ptr::null_mut();
        (*delaunay).first_id = -1;
        (*delaunay).t_in = core::ptr::null_mut();
        (*delaunay).t_out = core::ptr::null_mut();
        (*delaunay).nflags = 0;
        (*delaunay).nflagsallocated = 0;
        (*delaunay).flagids = core::ptr::null_mut();
        delaunay
    }
}

/// Original static selected-`CLARKSON_HULL` `tio_destroy` (`delaunay.c:90`).
unsafe fn tio_destroy(hull: *mut HullIo) {
    unsafe {
        if !(*hull).pointlist.is_null() {
            libc::free((*hull).pointlist.cast());
        }
        if !(*hull).trianglelist.is_null() {
            libc::free((*hull).trianglelist.cast());
        }
        if !(*hull).neighborlist.is_null() {
            libc::free((*hull).neighborlist.cast());
        }
    }
}

/// Original static `tio2delaunay` selected `CLARKSON_HULL` branch (`delaunay.c:157`).
unsafe fn tio2delaunay(hull: *mut HullIo, delaunay: *mut Delaunay) {
    unsafe {
        for index in 0..(*delaunay).npoints {
            let point = (*delaunay).points.add(index as usize);
            (*delaunay).xmin = (*delaunay).xmin.min((*point).x);
            (*delaunay).xmax = (*delaunay).xmax.max((*point).x);
            (*delaunay).ymin = (*delaunay).ymin.min((*point).y);
            (*delaunay).ymax = (*delaunay).ymax.max((*point).y);
        }
        (*delaunay).ntriangles = (*hull).numberoftriangles;
        if NN_VERBOSE != 0 {
            libc::fprintf(stderr, c"input:\n".as_ptr());
            for index in 0..(*delaunay).npoints {
                let point = (*delaunay).points.add(index as usize);
                libc::fprintf(
                    stderr,
                    c"  %d: %15.7g %15.7g %15.7g\n".as_ptr(),
                    index,
                    (*point).x,
                    (*point).y,
                    (*point).z,
                );
            }
            libc::fprintf(stderr, c"triangles:\n".as_ptr());
        }
        if (*delaunay).ntriangles <= 0 {
            return;
        }
        let count = (*delaunay).ntriangles as usize;
        (*delaunay).triangles = libc::malloc(count * core::mem::size_of::<Triangle>()).cast();
        (*delaunay).neighbours =
            libc::malloc(count * core::mem::size_of::<TriangleNeighbours>()).cast();
        (*delaunay).circles = libc::malloc(count * core::mem::size_of::<Circle>()).cast();
        (*delaunay).n_point_triangles =
            libc::calloc((*delaunay).npoints as usize, core::mem::size_of::<i32>()).cast();
        (*delaunay).point_triangles =
            libc::malloc((*delaunay).npoints as usize * core::mem::size_of::<*mut i32>()).cast();
        (*delaunay).flags = libc::calloc(count, core::mem::size_of::<i32>()).cast();
        for index in 0..(*delaunay).ntriangles {
            let triangle = (*delaunay).triangles.add(index as usize);
            let neighbours = (*delaunay).neighbours.add(index as usize);
            for coordinate in 0..3 {
                (*triangle).vids[coordinate] = *(*hull)
                    .trianglelist
                    .add((3 * index + coordinate as i32) as usize);
                (*neighbours).tids[coordinate] = *(*hull)
                    .neighborlist
                    .add((3 * index + coordinate as i32) as usize);
                (*(*delaunay)
                    .n_point_triangles
                    .add((*triangle).vids[coordinate] as usize)) += 1;
            }
            let _ = circle_build1(
                (*delaunay).circles.add(index as usize),
                (*delaunay).points.add((*triangle).vids[0] as usize),
                (*delaunay).points.add((*triangle).vids[1] as usize),
                (*delaunay).points.add((*triangle).vids[2] as usize),
            );
            if NN_VERBOSE != 0 {
                libc::fprintf(
                    stderr,
                    c"  %d: (%d,%d,%d)\n".as_ptr(),
                    index,
                    (*triangle).vids[0],
                    (*triangle).vids[1],
                    (*triangle).vids[2],
                );
            }
        }
        for index in 0..(*delaunay).npoints {
            let n = *(*delaunay).n_point_triangles.add(index as usize);
            *(*delaunay).point_triangles.add(index as usize) = if n > 0 {
                libc::malloc(n as usize * core::mem::size_of::<i32>()).cast()
            } else {
                core::ptr::null_mut()
            };
            *(*delaunay).n_point_triangles.add(index as usize) = 0;
        }
        for index in 0..(*delaunay).ntriangles {
            let triangle = (*delaunay).triangles.add(index as usize);
            for coordinate in 0..3 {
                let vertex = (*triangle).vids[coordinate];
                let count = (*delaunay).n_point_triangles.add(vertex as usize);
                *(*(*delaunay).point_triangles.add(vertex as usize)).add(*count as usize) = index;
                *count += 1;
            }
        }
    }
}

/// Original `delaunay_destroy` (`IMOD/libwarp/delaunay.c:372`).
pub unsafe fn delaunay_destroy(delaunay: *mut Delaunay) {
    unsafe {
        if delaunay.is_null() {
            return;
        }
        if !(*delaunay).point_triangles.is_null() {
            for index in 0..(*delaunay).npoints {
                let triangles = *(*delaunay).point_triangles.add(index as usize);
                if !triangles.is_null() {
                    libc::free(triangles.cast());
                }
            }
            libc::free((*delaunay).point_triangles.cast());
        }
        if (*delaunay).nedges > 0 {
            libc::free((*delaunay).edges.cast());
        }
        if !(*delaunay).n_point_triangles.is_null() {
            libc::free((*delaunay).n_point_triangles.cast());
        }
        if !(*delaunay).flags.is_null() {
            libc::free((*delaunay).flags.cast());
        }
        if !(*delaunay).circles.is_null() {
            libc::free((*delaunay).circles.cast());
        }
        if !(*delaunay).neighbours.is_null() {
            libc::free((*delaunay).neighbours.cast());
        }
        if !(*delaunay).triangles.is_null() {
            libc::free((*delaunay).triangles.cast());
        }
        if !(*delaunay).t_in.is_null() {
            istack_destroy((*delaunay).t_in);
        }
        if !(*delaunay).t_out.is_null() {
            istack_destroy((*delaunay).t_out);
        }
        if !(*delaunay).flagids.is_null() {
            libc::free((*delaunay).flagids.cast());
        }
        libc::free(delaunay.cast());
    }
}

/// Original `delaunay_build` selected `CLARKSON_HULL` branch (`delaunay.c:279`).
pub unsafe fn delaunay_build(
    point_count: i32,
    points: *mut Point,
    _: i32,
    _: *mut i32,
    prune_count: i32,
    criteria: *mut f64,
) -> *mut Delaunay {
    unsafe {
        let delaunay = delaunay_create();
        if point_count == 0 {
            libc::free(delaunay.cast());
            return core::ptr::null_mut();
        }
        let coordinates =
            libc::malloc((2 * point_count) as usize * core::mem::size_of::<f64>()).cast::<f64>();
        for i in 0..point_count {
            *coordinates.add((2 * i) as usize) = (*points.add(i as usize)).x;
            *coordinates.add((2 * i + 1) as usize) = (*points.add(i as usize)).y;
        }
        let mut hull = HullIo {
            pointlist: coordinates,
            numberofpoints: point_count,
            trianglelist: core::ptr::null_mut(),
            neighborlist: core::ptr::null_mut(),
            numberoftriangles: 0,
            height_base_crit: 0.,
            area_fraction_crit: 0.,
            min_num_for_pruning: 2,
            verbose: NN_VERBOSE,
        };
        if prune_count > 0 && !criteria.is_null() {
            hull.min_num_for_pruning = prune_count;
            hull.height_base_crit = *criteria;
            hull.area_fraction_crit = *criteria.add(1);
        }
        if NN_VERBOSE != 0 {
            libc::fflush(stderr);
        }
        let _ = hull_triangulate(&mut hull);
        if NN_VERBOSE != 0 {
            libc::fflush(stderr);
        }
        (*delaunay).npoints = point_count;
        (*delaunay).points = points;
        tio2delaunay(&mut hull, delaunay);
        tio_destroy(&mut hull);
        delaunay
    }
}

/// Original static `onrightside` (`IMOD/libwarp/delaunay.c:408`).
unsafe fn on_right_side(point: *mut Point, point_zero: *mut Point, point_one: *mut Point) -> i32 {
    unsafe {
        (((*point_one).x - (*point).x) * ((*point_zero).y - (*point).y)
            > ((*point_zero).x - (*point).x) * ((*point_one).y - (*point).y)) as i32
    }
}

/// Original `delaunay_xytoi` (`IMOD/libwarp/delaunay.c:420`).
pub unsafe fn delaunay_xytoi(delaunay: *mut Delaunay, point: *mut Point, mut id: i32) -> i32 {
    unsafe {
        if (*point).x < (*delaunay).xmin
            || (*point).x > (*delaunay).xmax
            || (*point).y < (*delaunay).ymin
            || (*point).y > (*delaunay).ymax
        {
            return -1;
        }

        if id < 0 || id > (*delaunay).ntriangles {
            id = 0;
        }
        let mut triangle = (*delaunay).triangles.add(id as usize);
        loop {
            let mut index = 0;
            while index < 3 {
                let index_one = (index + 1) % 3;
                if on_right_side(
                    point,
                    (*delaunay)
                        .points
                        .add((*triangle).vids[index as usize] as usize),
                    (*delaunay)
                        .points
                        .add((*triangle).vids[index_one as usize] as usize),
                ) != 0
                {
                    id = (*delaunay).neighbours.add(id as usize).read().tids
                        [((index + 2) % 3) as usize];
                    if id < 0 {
                        return id;
                    }
                    triangle = (*delaunay).triangles.add(id as usize);
                    break;
                }
                index += 1;
            }
            if index == 3 {
                break;
            }
        }
        id
    }
}

/// Original static `delaunay_addflag` (`IMOD/libwarp/delaunay.c:448`).
unsafe fn delaunay_addflag(delaunay: *mut Delaunay, index: i32) {
    unsafe {
        if (*delaunay).nflags == (*delaunay).nflagsallocated {
            (*delaunay).nflagsallocated += 100;
            (*delaunay).flagids = libc::realloc(
                (*delaunay).flagids.cast(),
                (*delaunay).nflagsallocated as usize * core::mem::size_of::<i32>(),
            )
            .cast();
        }
        *(*delaunay).flagids.add((*delaunay).nflags as usize) = index;
        (*delaunay).nflags += 1;
    }
}

/// Original static `delaunay_resetflags` (`IMOD/libwarp/delaunay.c:458`).
unsafe fn delaunay_resetflags(delaunay: *mut Delaunay) {
    unsafe {
        for index in 0..(*delaunay).nflags {
            *(*delaunay)
                .flags
                .add(*(*delaunay).flagids.add(index as usize) as usize) = 0;
        }
        (*delaunay).nflags = 0;
    }
}

/// Original `delaunay_circles_find` (`IMOD/libwarp/delaunay.c:486`).
pub unsafe fn delaunay_circles_find(
    delaunay: *mut Delaunay,
    point: *mut Point,
    count: *mut i32,
    output: *mut *mut i32,
) {
    unsafe {
        let mut contains = 0;
        if (*delaunay).t_in.is_null() {
            (*delaunay).t_in = istack_create();
            (*delaunay).t_out = istack_create();
        }
        if (*delaunay).ntriangles <= 20 {
            istack_reset((*delaunay).t_out);
            for index in 0..(*delaunay).ntriangles {
                if circle_contains((*delaunay).circles.add(index as usize), point) != 0 {
                    istack_push((*delaunay).t_out, index);
                }
            }
            *count = (*(*delaunay).t_out).n;
            *output = (*(*delaunay).t_out).v;
            return;
        }
        if (*delaunay).first_id < 0
            || circle_contains(
                (*delaunay).circles.add((*delaunay).first_id as usize),
                point,
            ) == 0
        {
            (*delaunay).first_id = delaunay_xytoi(delaunay, point, (*delaunay).first_id);
            contains = ((*delaunay).first_id >= 0) as i32;
            if (*delaunay).first_id < 0 {
                let prior_count = (*(*delaunay).t_out).n;
                let mut triangle_id = -1;
                let mut index = 0;
                while index < prior_count {
                    triangle_id = *(*(*delaunay).t_out).v.add(index as usize);
                    if circle_contains((*delaunay).circles.add(triangle_id as usize), point) != 0 {
                        break;
                    }
                    index += 1;
                }
                if triangle_id < 0 || index == prior_count {
                    let triangle_count = (*delaunay).ntriangles as f64;
                    triangle_id = 0;
                    while (triangle_id as f64) < triangle_count {
                        if circle_contains((*delaunay).circles.add(triangle_id as usize), point)
                            != 0
                        {
                            break;
                        }
                        triangle_id += 1;
                    }
                    if (triangle_id as f64) == triangle_count {
                        istack_reset((*delaunay).t_out);
                        *count = 0;
                        *output = core::ptr::null_mut();
                        return;
                    }
                }
                (*delaunay).first_id = triangle_id;
            }
        }
        istack_reset((*delaunay).t_in);
        istack_reset((*delaunay).t_out);
        istack_push((*delaunay).t_in, (*delaunay).first_id);
        *(*delaunay).flags.add((*delaunay).first_id as usize) = 1;
        delaunay_addflag(delaunay, (*delaunay).first_id);
        while (*(*delaunay).t_in).n > 0 {
            let triangle_id = istack_pop((*delaunay).t_in);
            let triangle = (*delaunay).triangles.add(triangle_id as usize);
            if contains != 0
                || circle_contains((*delaunay).circles.add(triangle_id as usize), point) != 0
            {
                istack_push((*delaunay).t_out, triangle_id);
                for index in 0..3 {
                    let vertex_id = (*triangle).vids[index as usize];
                    let triangle_count = *(*delaunay).n_point_triangles.add(vertex_id as usize);
                    for neighbour_index in 0..triangle_count {
                        let neighbour_id = *(*(*delaunay).point_triangles.add(vertex_id as usize))
                            .add(neighbour_index as usize);
                        if *(*delaunay).flags.add(neighbour_id as usize) == 0 {
                            istack_push((*delaunay).t_in, neighbour_id);
                            *(*delaunay).flags.add(neighbour_id as usize) = 1;
                            delaunay_addflag(delaunay, neighbour_id);
                        }
                    }
                }
            }
            contains = 0;
        }
        *count = (*(*delaunay).t_out).n;
        *output = (*(*delaunay).t_out).v;
        delaunay_resetflags(delaunay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xytoi_walks_neighbours_and_rejects_outside_source_bounds() {
        let mut points = [
            Point {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 0.,
                y: 1.,
                z: 0.,
            },
            Point {
                x: 1.,
                y: 1.,
                z: 0.,
            },
        ];
        let mut triangles = [Triangle { vids: [0, 1, 2] }, Triangle { vids: [1, 3, 2] }];
        let mut neighbours = [
            TriangleNeighbours { tids: [1, -1, -1] },
            TriangleNeighbours { tids: [-1, 0, -1] },
        ];
        let mut delaunay: Delaunay = unsafe { core::mem::zeroed() };
        delaunay.points = points.as_mut_ptr();
        delaunay.xmin = 0.;
        delaunay.xmax = 1.;
        delaunay.ymin = 0.;
        delaunay.ymax = 1.;
        delaunay.ntriangles = 2;
        delaunay.triangles = triangles.as_mut_ptr();
        delaunay.neighbours = neighbours.as_mut_ptr();
        let mut upper_right = Point {
            x: 0.75,
            y: 0.75,
            z: 0.,
        };
        let mut outside = Point {
            x: 1.1,
            y: 0.5,
            z: 0.,
        };

        unsafe {
            assert_eq!(delaunay_xytoi(&mut delaunay, &mut upper_right, 0), 1);
            assert_eq!(delaunay_xytoi(&mut delaunay, &mut outside, 0), -1);
        }
    }

    #[test]
    fn circles_find_uses_the_source_small_triangulation_linear_search() {
        let mut points = [Point {
            x: 0.,
            y: 0.,
            z: 0.,
        }];
        let circles = [
            Circle {
                x: 0.,
                y: 0.,
                r: 1.,
            },
            Circle {
                x: 3.,
                y: 0.,
                r: 1.,
            },
        ];
        let mut delaunay = unsafe { delaunay_create() };
        unsafe {
            (*delaunay).points = points.as_mut_ptr();
            (*delaunay).npoints = 1;
            (*delaunay).circles = libc::malloc(2 * core::mem::size_of::<Circle>()).cast();
            core::ptr::copy_nonoverlapping(circles.as_ptr(), (*delaunay).circles, 2);
            (*delaunay).ntriangles = 2;
            let mut query = Point {
                x: 0.5,
                y: 0.,
                z: 0.,
            };
            let mut count = -1;
            let mut found = core::ptr::null_mut();
            delaunay_circles_find(delaunay, &mut query, &mut count, &mut found);
            assert_eq!(count, 1);
            assert_eq!(*found, 0);
            delaunay_destroy(delaunay);
        }
    }

    #[test]
    fn hull_backend_preserves_source_bounds_when_no_triangle_is_possible() {
        let mut points = [
            Point {
                x: -2.,
                y: 4.,
                z: 0.,
            },
            Point {
                x: 3.,
                y: -1.,
                z: 0.,
            },
        ];
        unsafe {
            let delaunay = delaunay_build(
                2,
                points.as_mut_ptr(),
                0,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
            );
            assert_eq!((*delaunay).ntriangles, 0);
            assert_eq!((*delaunay).xmin, -2.);
            assert_eq!((*delaunay).xmax, 3.);
            assert_eq!((*delaunay).ymin, -1.);
            assert_eq!((*delaunay).ymax, 4.);
            delaunay_destroy(delaunay);
        }
    }
}
