//! Translation of `IMOD/libwarp/lpi.c`.
#![allow(dead_code)]

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::delaunay::{Delaunay, delaunay_build, delaunay_destroy, delaunay_xytoi};
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};

/// C `lweights` (`lpi.c:35`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Lweights {
    pub w: [f64; 3],
}

/// C `struct lpi` (`lpi.c:39`).
///
/// The C member is `delaunay* d`, an alias the interpolator does not own:
/// `lpi_destroy` frees only `l->weights` and `l`, and
/// `lpi_interpolate_points` destroys the triangulation itself afterwards.
/// Rust cannot hold that alias, so the `Box` moves in here and
/// [`lpi_destroy`] hands it back.
pub struct Lpi {
    pub d: Box<Delaunay>,
    pub weights: Vec<Lweights>,
}

/// Original `lpi_build` (`lpi.c:51`).
pub fn lpi_build(d: Box<Delaunay>) -> Box<Lpi> {
    let mut l = Box::new(Lpi {
        d,
        weights: Vec::new(),
    });

    l.weights = vec![Lweights::default(); l.d.ntriangles as usize];

    for i in 0..l.d.ntriangles {
        let t = l.d.triangles[i as usize];
        let x0 = l.d.points[t.vids[0] as usize].x;
        let y0 = l.d.points[t.vids[0] as usize].y;
        let z0 = l.d.points[t.vids[0] as usize].z;
        let x1 = l.d.points[t.vids[1] as usize].x;
        let y1 = l.d.points[t.vids[1] as usize].y;
        let z1 = l.d.points[t.vids[1] as usize].z;
        let x2 = l.d.points[t.vids[2] as usize].x;
        let y2 = l.d.points[t.vids[2] as usize].y;
        let z2 = l.d.points[t.vids[2] as usize].z;
        let x02 = x0 - x2;
        let y02 = y0 - y2;
        let z02 = z0 - z2;
        let x12 = x1 - x2;
        let y12 = y1 - y2;
        let z12 = z1 - z2;
        let lw = &mut l.weights[i as usize];

        if y12 != 0.0 {
            let y0212 = y02 / y12;

            lw.w[0] = (z02 - z12 * y0212) / (x02 - x12 * y0212);
            lw.w[1] = (z12 - lw.w[0] * x12) / y12;
            lw.w[2] = z2 - lw.w[0] * x2 - lw.w[1] * y2;
        } else {
            let x0212 = x02 / x12;

            lw.w[1] = (z02 - z12 * x0212) / (y02 - y12 * x0212);
            lw.w[0] = (z12 - lw.w[1] * y12) / x12;
            lw.w[2] = z2 - lw.w[0] * x2 - lw.w[1] * y2;
        }
    }

    l
}

/// Original `lpi_destroy` (`lpi.c:96`).
///
/// `free(l->weights); free(l);` — and the triangulation the C left alone comes
/// back out, because ownership of it had to move in.
pub fn lpi_destroy(l: Box<Lpi>) -> Box<Delaunay> {
    l.d
}

/// Original `lpi_interpolate_point` (`lpi.c:108`).
pub fn lpi_interpolate_point(l: &mut Lpi, p: &mut Point) {
    let tid = delaunay_xytoi(&l.d, p, l.d.first_id);

    if tid >= 0 {
        let lw = l.weights[tid as usize];

        l.d.first_id = tid;
        p.z = p.x * lw.w[0] + p.y * lw.w[1] + lw.w[2];
    } else {
        p.z = f64::NAN;
    }
}

/// Original `lpi_interpolate_points` (`lpi.c:128`).
pub fn lpi_interpolate_points(nin: i32, pin: &[Point], nout: i32, pout: &mut [Point]) {
    let d = delaunay_build(nin, pin, 0, None, 0, None);
    let mut l = lpi_build(d.unwrap());
    let seed = 0;

    if NN_VERBOSE.get() != 0 {
        let mut err = ImodFile::Stderr;
        let _ = err.write_all(b"xytoi:\n");
        for i in 0..nout {
            let p = &pout[i as usize];

            let _ = err.write_all(
                c_format(
                    "(%.7g,%.7g) -> %d\n",
                    &[
                        CArg::Dbl(p.x),
                        CArg::Dbl(p.y),
                        CArg::Int(delaunay_xytoi(&l.d, p, seed) as i64),
                    ],
                )
                .as_bytes(),
            );
        }
    }

    for i in 0..nout {
        let mut p = pout[i as usize];
        lpi_interpolate_point(&mut l, &mut p);
        pout[i as usize] = p;
    }

    if NN_VERBOSE.get() != 0 {
        let mut err = ImodFile::Stderr;
        let _ = err.write_all(b"output:\n");
        for i in 0..nout {
            let p = &pout[i as usize];
            let _ = err.write_all(
                c_format(
                    "  %d:%15.7g %15.7g %15.7g\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Dbl(p.x),
                        CArg::Dbl(p.y),
                        CArg::Dbl(p.z),
                    ],
                )
                .as_bytes(),
            );
        }
    }

    let d = lpi_destroy(l);
    delaunay_destroy(Some(d));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_interpolation_constructs_delaunay_for_a_three_point_plane() {
        let input = [
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
        let direct = delaunay_build(3, &input, 0, None, 0, None).unwrap();
        assert_eq!(
            direct.ntriangles, 1,
            "native hull did not retain one source triangle"
        );
        delaunay_destroy(Some(direct));

        let mut output = [Point {
            x: 0.25,
            y: 0.5,
            z: -1.,
        }];
        lpi_interpolate_points(3, &input, 1, &mut output);
        assert!(
            (output[0].z - 3.).abs() < 1.0e-12,
            "constructed interpolation produced {}",
            output[0].z
        );
    }
}
