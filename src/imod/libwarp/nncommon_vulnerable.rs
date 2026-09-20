//! Translation of `IMOD/libwarp/nncommon-vulnerable.c`.
//!
//! The file exists because these two circumcircle builders were "found to be
//! vulnerable from -O2 optimisation by gcc" (`nncommon-vulnerable.c:11`), and
//! `IMOD/libwarp/Makefile:30` compiles it with `$(NNVULNFLAGS)`, which
//! `IMOD/setup2:119` sets to `-ffloat-store`. On x86-64 every `double`
//! operation already goes through SSE2 at IEEE double precision, so
//! `-ffloat-store` is a no-op there and Rust's arithmetic matches it; the flag
//! only bites on 32-bit x87. Nothing in the translation can express it, and
//! nothing needs to — but the expression *order* below is load-bearing and
//! must not be regrouped.

use crate::imod::libwarp::delaunay::Circle;
use crate::imod::libwarp::nn::Point;

/// C `MULT` (`nncommon-vulnerable.c:33`).
const MULT: f64 = 1.0e+7;

/// Original `circle_build1` (`nncommon-vulnerable.c:35`).
///
/// `nan.h`'s `NaN` is `static const double NaN = 0.0 / 0.0`, which gcc folds
/// at compile time to the *positive* quiet NaN `7ff8000000000000` — the same
/// bit pattern as [`f64::NAN`]. (A run-time `0.0/0.0` on x86-64 gives the
/// negative one, `fff8…`, which `printf` renders `-nan`; the static
/// initialiser does not.)
pub fn circle_build1(c: &mut Circle, p1: &Point, p2: &Point, p3: &Point) -> i32 {
    let x2 = p2.x - p1.x;
    let y2 = p2.y - p1.y;
    let x3 = p3.x - p1.x;
    let y3 = p3.y - p1.y;

    let denom = x2 * y3 - y2 * x3;
    let frac;

    if denom == 0.0 {
        c.x = f64::NAN;
        c.y = f64::NAN;
        c.r = f64::NAN;
        return 0;
    }

    frac = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denom;
    c.x = (x3 + frac * y3) / 2.0;
    c.y = (y3 - frac * x3) / 2.0;
    c.r = c.x.hypot(c.y);
    c.x += p1.x;
    c.y += p1.y;

    1
}

/// Original `circle_build2` (`nncommon-vulnerable.c:65`).
pub fn circle_build2(c: &mut Circle, p1: &Point, p2: &Point, p3: &Point) -> i32 {
    let x2 = p2.x - p1.x;
    let y2 = p2.y - p1.y;
    let x3 = p3.x - p1.x;
    let y3 = p3.y - p1.y;

    let denom = x2 * y3 - y2 * x3;
    let frac;

    if denom == 0. {
        c.x = f64::NAN;
        c.y = f64::NAN;
        c.r = f64::NAN;
        return 0;
    }

    frac = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denom;
    c.x = (x3 + frac * y3) / 2.0;
    c.y = (y3 - frac * x3) / 2.0;
    c.r = c.x.hypot(c.y);
    if c.r > (x2.abs() + x3.abs() + y2.abs() + y3.abs()) * MULT {
        c.x = f64::NAN;
        c.y = f64::NAN;
    } else {
        c.x += p1.x;
        c.y += p1.y;
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_circle_builders_preserve_general_collinear_and_precision_paths() {
        let p1 = Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let p2 = Point {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        };
        let mut p3 = Point {
            x: 0.0,
            y: 2.0,
            z: 0.0,
        };
        let mut circle = Circle::default();
        assert_eq!(circle_build1(&mut circle, &p1, &p2, &p3), 1);
        assert_eq!((circle.x, circle.y, circle.r), (1.0, 1.0, 2.0_f64.sqrt()));
        p3 = Point {
            x: 4.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(circle_build1(&mut circle, &p1, &p2, &p3), 0);
        assert!(circle.x.is_nan() && circle.y.is_nan() && circle.r.is_nan());
        assert_eq!(circle_build2(&mut circle, &p1, &p2, &p3), 0);
    }
}
