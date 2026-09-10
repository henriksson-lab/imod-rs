//! Translation of `IMOD/libwarp/nncommon-vulnerable.c`.
#![allow(dead_code)]

use crate::imod::libwarp::delaunay::Circle;
use crate::imod::libwarp::nn::Point;

/// Original `circle_build1` (`nncommon-vulnerable.c:35`).
pub unsafe fn circle_build1(
    circle: *mut Circle,
    point1: *mut Point,
    point2: *mut Point,
    point3: *mut Point,
) -> i32 {
    unsafe {
        let x2 = (*point2).x - (*point1).x;
        let y2 = (*point2).y - (*point1).y;
        let x3 = (*point3).x - (*point1).x;
        let y3 = (*point3).y - (*point1).y;
        let denominator = x2 * y3 - y2 * x3;
        if denominator == 0.0 {
            (*circle).x = f64::NAN;
            (*circle).y = f64::NAN;
            (*circle).r = f64::NAN;
            return 0;
        }
        let fraction = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denominator;
        (*circle).x = (x3 + fraction * y3) / 2.0;
        (*circle).y = (y3 - fraction * x3) / 2.0;
        (*circle).r = (*circle).x.hypot((*circle).y);
        (*circle).x += (*point1).x;
        (*circle).y += (*point1).y;
        1
    }
}

/// Original `circle_build2` (`nncommon-vulnerable.c:65`).
pub unsafe fn circle_build2(
    circle: *mut Circle,
    point1: *mut Point,
    point2: *mut Point,
    point3: *mut Point,
) -> i32 {
    unsafe {
        let x2 = (*point2).x - (*point1).x;
        let y2 = (*point2).y - (*point1).y;
        let x3 = (*point3).x - (*point1).x;
        let y3 = (*point3).y - (*point1).y;
        let denominator = x2 * y3 - y2 * x3;
        if denominator == 0.0 {
            (*circle).x = f64::NAN;
            (*circle).y = f64::NAN;
            (*circle).r = f64::NAN;
            return 0;
        }
        let fraction = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denominator;
        (*circle).x = (x3 + fraction * y3) / 2.0;
        (*circle).y = (y3 - fraction * x3) / 2.0;
        (*circle).r = (*circle).x.hypot((*circle).y);
        if (*circle).r > (x2.abs() + x3.abs() + y2.abs() + y3.abs()) * 1.0e7 {
            (*circle).x = f64::NAN;
            (*circle).y = f64::NAN;
        } else {
            (*circle).x += (*point1).x;
            (*circle).y += (*point1).y;
        }
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_circle_builders_preserve_general_collinear_and_precision_paths() {
        unsafe {
            let mut p1 = Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            let mut p2 = Point {
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
            assert_eq!(circle_build1(&mut circle, &mut p1, &mut p2, &mut p3), 1);
            assert_eq!((circle.x, circle.y, circle.r), (1.0, 1.0, 2.0_f64.sqrt()));
            p3 = Point {
                x: 4.0,
                y: 0.0,
                z: 0.0,
            };
            assert_eq!(circle_build1(&mut circle, &mut p1, &mut p2, &mut p3), 0);
            assert!(circle.x.is_nan() && circle.y.is_nan() && circle.r.is_nan());
            assert_eq!(circle_build2(&mut circle, &mut p1, &mut p2, &mut p3), 0);
        }
    }
}
