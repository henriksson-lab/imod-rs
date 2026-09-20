//! Translation of `IMOD/libwarp/pointops.c` and `points.h`.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};

/// C `Coord` (`points.h:10`).
pub type Coord = f64;
/// C `point` (`points.h:11`), a `Coord*` that every routine walks for `dim`
/// elements — a borrowed slice here.
pub type Point<'a> = &'a [Coord];
thread_local! {
    /// C global `pdim` (`pointops.c:23`).
    ///
    /// A `static mut` cannot be read without `unsafe`, so the source's global
    /// becomes a thread-local cell: read with `PDIM.get()`, write with
    /// `PDIM.set(v)`.
    pub static PDIM: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// Original `maxdist` (`pointops.c:26`).
pub fn maxdist(dim: i32, p1: &[Coord], p2: &[Coord]) -> Coord {
    let mut d: Coord = 0.;
    let mut i = dim;
    let mut k = 0usize;

    while i != 0 {
        let x = p1[k];
        let y = p2[k];
        k += 1;
        d += if x < y { y - x } else { x - y };
        i -= 1;
    }

    d
}

/// Original `print_point` (`pointops.c:41`).
///
/// The source's `FILE *F` is a write-only sink, so per NATIVE.md's shared
/// vocabulary it becomes `&mut dyn Write`, and `fprintf`'s `%g` goes through
/// [`c_format`] — Rust's `{}` is not C's `%g`.
pub fn print_point(f: &mut dyn Write, dim: i32, p: Option<Point<'_>>) {
    if p.is_none() {
        let _ = f.write_all(b"NULL");
        return;
    }
    let p = p.unwrap();
    for j in 0..dim {
        let _ = f.write_all(c_format("%g  ", &[CArg::Dbl(p[j as usize])]).as_bytes());
    }
}

/// Original `print_point_int` (`pointops.c:51`).
pub fn print_point_int(f: &mut dyn Write, dim: i32, p: Option<Point<'_>>) {
    if p.is_none() {
        let _ = f.write_all(b"NULL");
        return;
    }
    let p = p.unwrap();
    for j in 0..dim {
        let _ = f.write_all(c_format("%.20g  ", &[CArg::Dbl(p[j as usize])]).as_bytes());
    }
}

/// Original static `scale` (`pointops.c:62`).
pub fn scale(dim: i32, p: &mut [Coord]) -> i32 {
    let mut max: Coord = 0.;

    for i in 0..dim {
        let val = p[i as usize];
        let abs = if val > 0. { val } else { -val };
        max = if abs > max { abs } else { max };
    }

    if max < 100. * f64::EPSILON {
        let mut stderr = ImodFile::Stderr;
        let _ = stderr.write_all(b"fails to scale: ");
        print_point(&mut stderr, dim, Some(&p[..]));
        let _ = stderr.flush();
        let _ = stderr.write_all(b"\n");
        return 1;
    }

    for i in 0..dim {
        p[i as usize] /= max;
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_metric_and_scaling_are_preserved() {
        let mut left = [-2.0, 3.0, 1.0];
        let right = [4.0, -1.0, 1.0];
        assert_eq!(maxdist(3, &left, &right), 10.0);
        assert_eq!(scale(3, &mut left), 0);
        assert_eq!(left, [-2.0 / 3.0, 1.0, 1.0 / 3.0]);
        let mut zero = [0.0];
        assert_eq!(scale(1, &mut zero), 1);
    }

    #[test]
    fn printing_uses_the_c_format_conversions() {
        let mut out = Vec::<u8>::new();
        print_point(&mut out, 2, Some(&[1.0, 0.000012345678]));
        assert_eq!(out, b"1  1.23457e-05  ");
        out.clear();
        print_point_int(&mut out, 1, Some(&[0.1]));
        assert_eq!(out, b"0.10000000000000000555  ");
        out.clear();
        print_point(&mut out, 3, None);
        assert_eq!(out, b"NULL");
    }
}
