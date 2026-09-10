//! Translation of `IMOD/libwarp/nnai.c`.
#![allow(dead_code)]

use crate::imod::libwarp::delaunay::Delaunay;
use crate::imod::libwarp::nn::Point;
use crate::imod::libwarp::nnpi::{
    nnpi_calculate_weights, nnpi_create, nnpi_destroy, nnpi_get_nvertices, nnpi_get_vertices,
    nnpi_get_weights,
};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// C local `nn_weights` (`nnai.c`).
#[repr(C)]
pub struct NnWeights {
    pub nvertices: i32,
    pub vertices: *mut i32,
    pub weights: *mut f64,
}

/// C `struct nnai` (`nnai.c`).
#[repr(C)]
pub struct Nnai {
    pub d: *mut Delaunay,
    pub wmin: f64,
    pub n: f64,
    pub x: *mut f64,
    pub y: *mut f64,
    pub weights: *mut NnWeights,
}

/// Original `nnai_build` (`nnai.c:57`).
pub unsafe fn nnai_build(
    delaunay: *mut Delaunay,
    count: i32,
    x: *mut f64,
    y: *mut f64,
) -> *mut Nnai {
    unsafe {
        if count <= 0 {
            libc::fprintf(
                stderr,
                c"  error: libnn: nnai_create(): n = %d\n".as_ptr(),
                count,
            );
            libc::exit(1);
        }
        let nn = libc::malloc(core::mem::size_of::<Nnai>()).cast::<Nnai>();
        let point_interpolator = nnpi_create(delaunay);
        (*nn).d = delaunay;
        (*nn).wmin = -f64::MAX;
        (*nn).n = count as f64;
        (*nn).x = libc::malloc(count as usize * core::mem::size_of::<f64>()).cast();
        core::ptr::copy_nonoverlapping(x, (*nn).x, count as usize);
        (*nn).y = libc::malloc(count as usize * core::mem::size_of::<f64>()).cast();
        core::ptr::copy_nonoverlapping(y, (*nn).y, count as usize);
        (*nn).weights = libc::malloc(count as usize * core::mem::size_of::<NnWeights>()).cast();
        for index in 0..count {
            let weight = (*nn).weights.add(index as usize);
            let mut point = Point {
                x: *x.add(index as usize),
                y: *y.add(index as usize),
                z: 0.0,
            };
            nnpi_calculate_weights(point_interpolator, &mut point);
            (*weight).nvertices = nnpi_get_nvertices(point_interpolator);
            if (*weight).nvertices != 0 {
                let vertex_count = (*weight).nvertices as usize;
                (*weight).vertices =
                    libc::malloc(vertex_count * core::mem::size_of::<i32>()).cast();
                core::ptr::copy_nonoverlapping(
                    nnpi_get_vertices(point_interpolator),
                    (*weight).vertices,
                    vertex_count,
                );
                (*weight).weights = libc::malloc(vertex_count * core::mem::size_of::<f64>()).cast();
                core::ptr::copy_nonoverlapping(
                    nnpi_get_weights(point_interpolator),
                    (*weight).weights,
                    vertex_count,
                );
            }
        }
        nnpi_destroy(point_interpolator);
        nn
    }
}

/// Original `nnai_destroy` (`nnai.c:108`).
pub unsafe fn nnai_destroy(nn: *mut Nnai) {
    unsafe {
        for index in 0..(*nn).n as i32 {
            let weight = (*nn).weights.add(index as usize);
            if (*weight).nvertices != 0 {
                libc::free((*weight).vertices.cast());
                libc::free((*weight).weights.cast());
            }
        }
        libc::free((*nn).x.cast());
        libc::free((*nn).y.cast());
        libc::free((*nn).weights.cast());
        libc::free(nn.cast());
    }
}

/// Original `nnai_interpolate` (`nnai.c:131`).
pub unsafe fn nnai_interpolate(nn: *mut Nnai, zin: *mut f64, zout: *mut f64) {
    unsafe {
        for index in 0..(*nn).n as i32 {
            let weight = (*nn).weights.add(index as usize);
            let mut z = if (*weight).nvertices == 0 {
                f64::NAN
            } else {
                0.0
            };
            for vertex_index in 0..(*weight).nvertices {
                let vertex_weight = *(*weight).weights.add(vertex_index as usize);
                if vertex_weight < (*nn).wmin {
                    z = f64::NAN;
                    break;
                }
                z += vertex_weight
                    * *zin.add(*(*weight).vertices.add(vertex_index as usize) as usize);
            }
            *zout.add(index as usize) = z;
        }
    }
}

/// Original `nnai_setwmin` (`nnai.c:160`).
pub unsafe fn nnai_setwmin(nn: *mut Nnai, wmin: f64) {
    unsafe { (*nn).wmin = wmin }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libwarp::delaunay::{delaunay_build, delaunay_destroy};

    #[test]
    fn precomputed_natural_neighbour_weights_reproduce_a_plane() {
        unsafe {
            let points = libc::malloc(3 * core::mem::size_of::<Point>()).cast::<Point>();
            *points.add(0) = Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            *points.add(1) = Point {
                x: 1.0,
                y: 0.0,
                z: 1.0,
            };
            *points.add(2) = Point {
                x: 0.0,
                y: 1.0,
                z: 2.0,
            };
            let delaunay = delaunay_build(
                3,
                points,
                0,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
            );
            let mut x = [0.25];
            let mut y = [0.5];
            let array_interpolator = nnai_build(delaunay, 1, x.as_mut_ptr(), y.as_mut_ptr());
            let mut input = [0.0, 1.0, 2.0];
            let mut output = [f64::NAN];
            nnai_interpolate(array_interpolator, input.as_mut_ptr(), output.as_mut_ptr());
            assert!((output[0] - 1.25).abs() < 1.0e-12);
            nnai_destroy(array_interpolator);
            delaunay_destroy(delaunay);
        }
    }
}
