//! Translation of `IMOD/libfft/srfp.c`.
//!
//! `factor`, `sym` and `unsym` are the caller's `int[16]` scratch arrays,
//! written from index 1 upward exactly as the source writes them.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use std::io::Write as _;
/// C `srfp`.
pub fn srfp(
    pts: i32,
    pmax: i32,
    twogrp: i32,
    factor: &mut [i32],
    sym: &mut [i32],
    psym: &mut i32,
    unsym: &mut [i32],
    error: &mut i32,
) {
    let mut pp = [0_i32; 15];
    let mut qq = [0_i32; 8];
    let mut n = pts;
    let nest = 14_usize;
    let mut f = 2;
    let mut p = 0_usize;
    let mut q = 0_usize;
    *psym = 1;
    while n > 1 {
        let mut divisor = f;
        while divisor <= pmax && n != (n / divisor) * divisor {
            divisor += 1;
        }
        // `srfp.c:32-40`: two separate checks, each printing before it sets
        // `error`.  The translation had merged them into one condition and
        // emitted neither message; a C-versus-Rust differential over the
        // factorisation put 20 stdout lines on the C side and none on ours.
        //
        // These write to `ImodFile::Stdout`, the **C** stdout stream, not
        // Rust's, on purpose.  `cmplft` prints its own `invalid number of
        // points` message on the same stream immediately after `srfp` returns
        // with this flag set, and C stdio is block-buffered under redirection
        // while Rust's is not — a Rust `print!` here would reorder the two
        // lines in a captured file while looking correct on a terminal.  The
        // whole `libfft` output path is on that stream; NATIVE.md §1, §7b.
        if divisor > pmax {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "largest factor exceeds %d.  n = %d.\n",
                &[CArg::Int(pmax as i64), CArg::Int(pts as i64)],
            ));
            *error = 1;
            return;
        }
        if 2 * p + q >= nest {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "factor count exceeds %d.  n = %d.\n",
                &[CArg::Int(nest as i64), CArg::Int(pts as i64)],
            ));
            *error = 1;
            return;
        }
        f = divisor;
        n /= f;
        if n != (n / f) * f {
            q += 1;
            qq[q] = f;
        } else {
            n /= f;
            p += 1;
            pp[p] = f;
            *psym *= f;
        }
    }
    let r = if q == 0 { 0 } else { 1 };
    if p >= 1 {
        for index in 1..=p {
            sym[index] = pp[p + 1 - index];
            factor[index] = pp[p + 1 - index];
            factor[p + q + index] = pp[index];
            sym[p + r + index] = pp[index];
        }
    }
    if q >= 1 {
        for index in 1..=q {
            unsym[index] = qq[index];
            factor[p + index] = qq[index];
        }
        sym[p + 1] = pts / (*psym * *psym);
    }
    let mut count = 2 * p + q;
    factor[count + 1] = 0;
    let mut power_two = 1;
    let mut index = 0_usize;
    while factor[index + 1] != 0 {
        index += 1;
        if factor[index] != 2 {
            continue;
        }
        power_two *= 2;
        factor[index] = 1;
        if power_two < twogrp && factor[index + 1] == 2 {
            continue;
        }
        factor[index] = power_two;
        power_two = 1;
    }
    let r = if p == 0 { 0 } else { r };
    count = 2 * p + r;
    sym[count + 1] = 0;
    let q = if q <= 1 { 0 } else { q };
    unsym[q + 1] = 0;
    *error = 0;
}

#[cfg(test)]
mod tests {
    use super::srfp;
    #[test]
    fn source_factor_limit_rejects_prime_above_nineteen() {
        let mut factor = [0_i32; 16];
        let mut sym = [0_i32; 16];
        let mut unsym = [0_i32; 16];
        let mut psym = 0;
        let mut error = 0;
        srfp(
            23,
            19,
            8,
            &mut factor,
            &mut sym,
            &mut psym,
            &mut unsym,
            &mut error,
        );
        assert_eq!(error, 1);
    }
}
