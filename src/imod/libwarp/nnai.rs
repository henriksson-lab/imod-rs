//! Translation of `IMOD/libwarp/nnai.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use crate::imod::libwarp::delaunay::Delaunay;
use crate::imod::libwarp::nn::Point;
use crate::imod::libwarp::nncommon::nn_quit;
use crate::imod::libwarp::nnpi::{
    nnpi_calculate_weights, nnpi_create, nnpi_destroy, nnpi_get_nvertices, nnpi_get_vertices,
    nnpi_get_weights,
};

/// C `NaN` (`nan.h:26`); see the note in `nnpi.rs`.
const NAN: f64 = f64::NAN;

/// C `nn_weights` (`nnai.c:38`).
///
/// `nnpi.c:791` declares an identical private type; both are translated in
/// their own module, as the source declares them.
pub struct NnWeights {
    pub nvertices: i32,
    /// vertex indices [nvertices]
    pub vertices: Vec<i32>,
    /// vertex weights [nvertices]
    pub weights: Vec<f64>,
}

/// C `struct nnai` (`nnai.c:43`).
pub struct Nnai {
    pub d: Delaunay,
    pub wmin: f64,
    /// number of output points — the source really does declare this `double`
    pub n: f64,
    /// [n]
    pub x: Vec<f64>,
    /// [n]
    pub y: Vec<f64>,
    pub weights: Vec<NnWeights>,
}

/// Original `nnai_build` (`nnai.c:60`).
///
/// Builds Natural Neighbours array interpolator. This includes calculation of
/// weights used in nnai_interpolate().
///
/// Two ownership deviations, both invisible:
///
/// * The source sets `nn->d = d` and then hands the *same* pointer to
///   `nnpi_create`, keeping it after `nnpi_destroy`. Rust cannot alias, so the
///   triangulation is moved into the point interpolator and taken back from
///   [`nnpi_destroy`] at the end. Nothing reads `nn->d` in between.
/// * `nn->wmin` is never assigned by the source, so it is `malloc` garbage
///   until `nnai_setwmin` runs — every caller in this tree calls that first
///   (`warpfiles.c:1137`). It starts at `nnpi_create`'s own default here.
pub fn nnai_build(d: Delaunay, n: i32, x: &[f64], y: &[f64]) -> Option<Nnai> {
    let mut point_interpolator = nnpi_create(d);
    let mut i;

    if n <= 0 {
        nn_quit(&c_format("nnai_create(): n = %d\n", &[CArg::Int(n as i64)]));
    }

    let mut nn_weights: Vec<NnWeights> = Vec::with_capacity(n as usize);
    let mut nn_x: Vec<f64> = vec![0.; n as usize];
    nn_x[..(n as usize)].copy_from_slice(&x[..(n as usize)]);
    let mut nn_y: Vec<f64> = vec![0.; n as usize];
    nn_y[..(n as usize)].copy_from_slice(&y[..(n as usize)]);

    i = 0;
    while i < n {
        let mut w = NnWeights {
            nvertices: 0,
            vertices: Vec::new(),
            weights: Vec::new(),
        };
        let mut p = Point::default();

        p.x = x[i as usize];
        p.y = y[i as usize];

        nnpi_calculate_weights(&mut point_interpolator, &p);

        // `int* vertices` and `double* weights` are declared at the top of the
        // source function; they are borrows of the interpolator's own arrays,
        // so they are taken inside the loop here where the borrow ends.
        let vertices = nnpi_get_vertices(&point_interpolator);
        let weights = nnpi_get_weights(&point_interpolator);

        w.nvertices = nnpi_get_nvertices(&point_interpolator);

        /* DNM: do not allocate or copy if no vertices */
        if w.nvertices != 0 {
            w.vertices = vec![0; w.nvertices as usize];
            w.vertices[..(w.nvertices as usize)]
                .copy_from_slice(&vertices[..(w.nvertices as usize)]);
            w.weights = vec![0.; w.nvertices as usize];
            w.weights[..(w.nvertices as usize)].copy_from_slice(&weights[..(w.nvertices as usize)]);
        }

        nn_weights.push(w);
        i += 1;
    }

    let d = nnpi_destroy(point_interpolator);

    Some(Nnai {
        d,
        wmin: -f64::MAX,
        n: n as f64,
        x: nn_x,
        y: nn_y,
        weights: nn_weights,
    })
}

/// Original `nnai_destroy` (`nnai.c:107`).
///
/// Every `free` in the source is a drop here; the triangulation `nn->d` is
/// dropped with the interpolator, where the source leaves it to the caller —
/// `warpfiles.c` keeps `sDelau` and `sNninterp` as two statics and frees both.
pub fn nnai_destroy(nn: Option<Nnai>) {
    drop(nn);
}

/// Original `nnai_interpolate` (`nnai.c:130`).
///
/// Conducts NN interpolation in a fixed array of output points using data
/// specified in a fixed array of input points. Uses pre-calculated weights.
pub fn nnai_interpolate(nn: &mut Nnai, zin: &[f64], zout: &mut [f64]) {
    let mut i;

    i = 0;
    while (i as f64) < nn.n {
        let w = &nn.weights[i as usize];
        let mut z = 0.0;
        let mut j;

        /* DNM: Set to NaN explicitly if no vertices */
        if w.nvertices == 0 {
            z = NAN;
        }
        j = 0;
        while j < w.nvertices {
            let weight = w.weights[j as usize];

            if weight < nn.wmin {
                z = NAN;
                break;
            }
            z += weight * zin[w.vertices[j as usize] as usize];
            j += 1;
        }

        zout[i as usize] = z;
        i += 1;
    }
}

/// Original `nnai_setwmin` (`nnai.c:167`).
pub fn nnai_setwmin(nn: &mut Nnai, wmin: f64) {
    nn.wmin = wmin;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libwarp::delaunay::delaunay_build;

    #[test]
    fn precomputed_natural_neighbour_weights_reproduce_a_plane() {
        let points = [
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point {
                x: 1.0,
                y: 0.0,
                z: 1.0,
            },
            Point {
                x: 0.0,
                y: 1.0,
                z: 2.0,
            },
        ];
        let delaunay = delaunay_build(3, &points, 0, None, 0, None).unwrap();
        let x = [0.25];
        let y = [0.5];
        let mut array_interpolator = nnai_build(delaunay, 1, &x, &y).unwrap();
        let input = [0.0, 1.0, 2.0];
        let mut output = [f64::NAN];
        nnai_interpolate(&mut array_interpolator, &input, &mut output);
        assert!((output[0] - 1.25).abs() < 1.0e-12);
        nnai_destroy(Some(array_interpolator));
    }
}
