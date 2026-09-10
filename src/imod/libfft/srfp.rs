//! Translation of `IMOD/libfft/srfp.c`.

/// C `srfp`.
pub unsafe fn srfp(
    pts: i32,
    pmax: i32,
    twogrp: i32,
    factor: *mut i32,
    sym: *mut i32,
    psym: *mut i32,
    unsym: *mut i32,
    error: *mut i32,
) {
    unsafe {
        let mut pp = [0_i32; 15];
        let mut qq = [0_i32; 8];
        let mut n = pts;
        let mut f = 2;
        let mut p = 0_usize;
        let mut q = 0_usize;
        *psym = 1;
        while n > 1 {
            let mut divisor = f;
            while divisor <= pmax && n != (n / divisor) * divisor {
                divisor += 1;
            }
            if divisor > pmax || 2 * p + q >= 14 {
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
                *sym.add(index) = pp[p + 1 - index];
                *factor.add(index) = pp[p + 1 - index];
                *factor.add(p + q + index) = pp[index];
                *sym.add(p + r + index) = pp[index];
            }
        }
        if q >= 1 {
            for index in 1..=q {
                *unsym.add(index) = qq[index];
                *factor.add(p + index) = qq[index];
            }
            *sym.add(p + 1) = pts / (*psym * *psym);
        }
        let mut count = 2 * p + q;
        *factor.add(count + 1) = 0;
        let mut power_two = 1;
        let mut index = 0_usize;
        while *factor.add(index + 1) != 0 {
            index += 1;
            if *factor.add(index) != 2 {
                continue;
            }
            power_two *= 2;
            *factor.add(index) = 1;
            if power_two < twogrp && *factor.add(index + 1) == 2 {
                continue;
            }
            *factor.add(index) = power_two;
            power_two = 1;
        }
        let r = if p == 0 { 0 } else { r };
        count = 2 * p + r;
        *sym.add(count + 1) = 0;
        let q = if q <= 1 { 0 } else { q };
        *unsym.add(q + 1) = 0;
        *error = 0;
    }
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
        unsafe {
            srfp(
                23,
                19,
                8,
                factor.as_mut_ptr(),
                sym.as_mut_ptr(),
                &mut psym,
                unsym.as_mut_ptr(),
                &mut error,
            );
        }
        assert_eq!(error, 1);
    }
}
