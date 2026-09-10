//! Translation of `IMOD/libwarp/lpi.c` and its `nn.h` declarations.
#![allow(dead_code)]

use crate::imod::libwarp::delaunay::{Delaunay, delaunay_build, delaunay_destroy, delaunay_xytoi};
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// C-private `lweights` (`lpi.c:34`).
#[repr(C)]
pub struct Lweights {
    pub w: [f64; 3],
}

/// C `struct lpi` (`lpi.c:38`, forward-declared by `nn.h`).
#[repr(C)]
pub struct Lpi {
    pub d: *mut Delaunay,
    pub weights: *mut Lweights,
}

/// Original `lpi_build` (`IMOD/libwarp/lpi.c:49`).
pub unsafe fn lpi_build(delaunay: *mut Delaunay) -> *mut Lpi {
    unsafe {
        let lpi = libc::malloc(core::mem::size_of::<Lpi>()).cast::<Lpi>();
        (*lpi).d = delaunay;
        (*lpi).weights =
            libc::malloc((*delaunay).ntriangles as usize * core::mem::size_of::<Lweights>())
                .cast::<Lweights>();

        for index in 0..(*delaunay).ntriangles {
            let triangle = (*delaunay).triangles.add(index as usize);
            let weights = (*lpi).weights.add(index as usize);
            let point_zero = (*delaunay).points.add((*triangle).vids[0] as usize);
            let point_one = (*delaunay).points.add((*triangle).vids[1] as usize);
            let point_two = (*delaunay).points.add((*triangle).vids[2] as usize);
            let x_zero = (*point_zero).x;
            let y_zero = (*point_zero).y;
            let z_zero = (*point_zero).z;
            let x_one = (*point_one).x;
            let y_one = (*point_one).y;
            let z_one = (*point_one).z;
            let x_two = (*point_two).x;
            let y_two = (*point_two).y;
            let z_two = (*point_two).z;
            let x_zero_two = x_zero - x_two;
            let y_zero_two = y_zero - y_two;
            let z_zero_two = z_zero - z_two;
            let x_one_two = x_one - x_two;
            let y_one_two = y_one - y_two;
            let z_one_two = z_one - z_two;

            if y_one_two != 0. {
                let y_zero_two_one_two = y_zero_two / y_one_two;
                (*weights).w[0] = (z_zero_two - z_one_two * y_zero_two_one_two)
                    / (x_zero_two - x_one_two * y_zero_two_one_two);
                (*weights).w[1] = (z_one_two - (*weights).w[0] * x_one_two) / y_one_two;
                (*weights).w[2] = z_two - (*weights).w[0] * x_two - (*weights).w[1] * y_two;
            } else {
                let x_zero_two_one_two = x_zero_two / x_one_two;
                (*weights).w[1] = (z_zero_two - z_one_two * x_zero_two_one_two)
                    / (y_zero_two - y_one_two * x_zero_two_one_two);
                (*weights).w[0] = (z_one_two - (*weights).w[1] * y_one_two) / x_one_two;
                (*weights).w[2] = z_two - (*weights).w[0] * x_two - (*weights).w[1] * y_two;
            }
        }
        lpi
    }
}

/// Original `lpi_destroy` (`IMOD/libwarp/lpi.c:100`).
pub unsafe fn lpi_destroy(lpi: *mut Lpi) {
    unsafe {
        libc::free((*lpi).weights.cast());
        libc::free(lpi.cast());
    }
}

/// Original `lpi_interpolate_point` (`IMOD/libwarp/lpi.c:111`).
pub unsafe fn lpi_interpolate_point(lpi: *mut Lpi, point: *mut Point) {
    unsafe {
        let delaunay = (*lpi).d;
        let triangle_id = delaunay_xytoi(delaunay, point, (*delaunay).first_id);
        if triangle_id >= 0 {
            let weights = (*lpi).weights.add(triangle_id as usize);
            (*delaunay).first_id = triangle_id;
            (*point).z =
                (*point).x * (*weights).w[0] + (*point).y * (*weights).w[1] + (*weights).w[2];
        } else {
            (*point).z = f64::NAN;
        }
    }
}

/// Original `lpi_interpolate_points` (`IMOD/libwarp/lpi.c:132`).
pub unsafe fn lpi_interpolate_points(
    input_count: i32,
    input: *mut Point,
    output_count: i32,
    output: *mut Point,
) {
    unsafe {
        let delaunay = delaunay_build(
            input_count,
            input,
            0,
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
        );
        let lpi = lpi_build(delaunay);
        if NN_VERBOSE != 0 {
            libc::fprintf(stderr, c"xytoi:\n".as_ptr());
            for index in 0..output_count {
                let point = output.add(index as usize);
                libc::fprintf(
                    stderr,
                    c"(%.7g,%.7g) -> %d\n".as_ptr(),
                    (*point).x,
                    (*point).y,
                    delaunay_xytoi(delaunay, point, 0),
                );
            }
        }
        for index in 0..output_count {
            lpi_interpolate_point(lpi, output.add(index as usize));
        }
        if NN_VERBOSE != 0 {
            libc::fprintf(stderr, c"output:\n".as_ptr());
            for index in 0..output_count {
                let point = output.add(index as usize);
                libc::fprintf(
                    stderr,
                    c"  %d:%15.7g %15.7g %15.7g\n".as_ptr(),
                    index,
                    (*point).x,
                    (*point).y,
                    (*point).z,
                );
            }
        }
        lpi_destroy(lpi);
        delaunay_destroy(delaunay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libwarp::delaunay::{Triangle, TriangleNeighbours};

    #[test]
    fn builds_plane_weights_and_interpolates_inside_triangle() {
        let mut points = [
            Point {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            Point {
                x: 1.,
                y: 0.,
                z: 3.,
            },
            Point {
                x: 0.,
                y: 1.,
                z: 4.,
            },
        ];
        let mut triangles = [Triangle { vids: [0, 1, 2] }];
        let mut neighbours = [TriangleNeighbours { tids: [-1, -1, -1] }];
        let mut delaunay: Delaunay = unsafe { core::mem::zeroed() };
        delaunay.points = points.as_mut_ptr();
        delaunay.xmin = 0.;
        delaunay.xmax = 1.;
        delaunay.ymin = 0.;
        delaunay.ymax = 1.;
        delaunay.ntriangles = 1;
        delaunay.triangles = triangles.as_mut_ptr();
        delaunay.neighbours = neighbours.as_mut_ptr();
        delaunay.first_id = -1;
        let mut inside = Point {
            x: 0.25,
            y: 0.5,
            z: -1.,
        };
        let mut outside = Point {
            x: 1.1,
            y: 0.5,
            z: 17.,
        };

        unsafe {
            let lpi = lpi_build(&mut delaunay);
            lpi_interpolate_point(lpi, &mut inside);
            lpi_interpolate_point(lpi, &mut outside);
            lpi_destroy(lpi);
        }
        assert_eq!(inside.z, 3.0);
        assert!(outside.z.is_nan());
        assert_eq!(delaunay.first_id, 0);
    }

    #[test]
    fn bulk_interpolation_constructs_delaunay_for_a_three_point_plane() {
        let mut input = [
            Point {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            Point {
                x: 1.,
                y: 0.,
                z: 3.,
            },
            Point {
                x: 0.,
                y: 1.,
                z: 4.,
            },
        ];
        let direct = unsafe {
            delaunay_build(
                3,
                input.as_mut_ptr(),
                0,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
            )
        };
        unsafe {
            assert_eq!(
                (*direct).ntriangles,
                1,
                "native hull did not retain one source triangle"
            );
            delaunay_destroy(direct);
        }
        let mut output = [Point {
            x: 0.25,
            y: 0.5,
            z: -1.,
        }];
        unsafe {
            lpi_interpolate_points(3, input.as_mut_ptr(), 1, output.as_mut_ptr());
        }
        assert!(
            (output[0].z - 3.).abs() < 1.0e-12,
            "constructed interpolation produced {}",
            output[0].z
        );
    }
}
