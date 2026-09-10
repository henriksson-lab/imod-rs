//! Translation of `IMOD/libwarp/pointops.c` and `points.h`.
#![allow(dead_code)]

/// C `Coord` (`points.h`).
pub type Coord = f64;
/// C `point` (`points.h`).
pub type Point = *mut Coord;
/// C global `pdim` (`pointops.c:23`).
pub static mut PDIM: i32 = 0;

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// Original `maxdist` (`pointops.c:26`).
pub unsafe fn maxdist(dim: i32, mut p1: Point, mut p2: Point) -> Coord {
    unsafe {
        let mut distance = 0.0;
        let mut index = dim;
        while index != 0 {
            let x = *p1;
            p1 = p1.add(1);
            let y = *p2;
            p2 = p2.add(1);
            distance += if x < y { y - x } else { x - y };
            index -= 1;
        }
        distance
    }
}

/// Original `print_point` (`pointops.c:41`).
pub unsafe fn print_point(file: *mut libc::FILE, dim: i32, mut point: Point) {
    unsafe {
        if point.is_null() {
            libc::fputs(c"NULL".as_ptr(), file);
            return;
        }
        for _ in 0..dim {
            libc::fprintf(file, c"%g  ".as_ptr(), *point);
            point = point.add(1);
        }
    }
}

/// Original `print_point_int` (`pointops.c:51`).
pub unsafe fn print_point_int(file: *mut libc::FILE, dim: i32, mut point: Point) {
    unsafe {
        if point.is_null() {
            libc::fputs(c"NULL".as_ptr(), file);
            return;
        }
        for _ in 0..dim {
            libc::fprintf(file, c"%.20g  ".as_ptr(), *point);
            point = point.add(1);
        }
    }
}

/// Original static `scale` (`pointops.c:62`).
pub unsafe fn scale(dim: i32, point: Point) -> i32 {
    unsafe {
        let mut max = 0.0;
        for index in 0..dim {
            let value = *point.add(index as usize);
            let absolute = if value > 0.0 { value } else { -value };
            max = if absolute > max { absolute } else { max };
        }
        if max < 100.0 * f64::EPSILON {
            libc::fputs(c"fails to scale: ".as_ptr(), stderr);
            print_point(stderr, dim, point);
            libc::fflush(stderr);
            libc::fputs(c"\n".as_ptr(), stderr);
            return 1;
        }
        for index in 0..dim {
            *point.add(index as usize) /= max;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_metric_and_scaling_are_preserved() {
        unsafe {
            let mut left = [-2.0, 3.0, 1.0];
            let mut right = [4.0, -1.0, 1.0];
            assert_eq!(maxdist(3, left.as_mut_ptr(), right.as_mut_ptr()), 10.0);
            assert_eq!(scale(3, left.as_mut_ptr()), 0);
            assert_eq!(left, [-2.0 / 3.0, 1.0, 1.0 / 3.0]);
            let mut zero = [0.0];
            assert_eq!(scale(1, zero.as_mut_ptr()), 1);
        }
    }
}
