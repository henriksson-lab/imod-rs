//! c2rust parity baseline of `IMOD/libfft/mdftkd.c`; identifiers and API are normalized at the module boundary.
#![allow(non_snake_case, unused_mut, unsafe_op_in_unsafe_fn)]

unsafe extern "C" {
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn cos(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sin(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
}
pub unsafe extern "C" fn mdftkd(
    mut n: ::core::ffi::c_int,
    mut factor: *mut ::core::ffi::c_int,
    mut dim: *mut ::core::ffi::c_int,
    mut x: *mut ::core::ffi::c_float,
    mut y: *mut ::core::ffi::c_float,
) {
    let mut indFac: ::core::ffi::c_int = 0;
    let mut nReduced: ::core::ffi::c_int = 0;
    let mut pFac: ::core::ffi::c_int = 0;
    let mut stepAlong: ::core::ffi::c_int = 0;
    let mut sepAlong: ::core::ffi::c_int = 0;
    sepAlong = *dim.offset(2 as ::core::ffi::c_int as isize);
    indFac = 0 as ::core::ffi::c_int;
    nReduced = n;
    while *factor.offset((indFac + 1 as ::core::ffi::c_int) as isize) != 0 as ::core::ffi::c_int {
        indFac = indFac + 1 as ::core::ffi::c_int;
        pFac = *factor.offset(indFac as isize);
        nReduced = nReduced / pFac;
        stepAlong = nReduced * sepAlong;
        match pFac {
            1 => {}
            2 => {
                r2cftk(
                    n,
                    nReduced,
                    x,
                    y,
                    x.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    y.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    dim,
                );
            }
            3 => {
                r3cftk(
                    n,
                    nReduced,
                    x,
                    y,
                    x.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    y.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    x.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    dim,
                );
            }
            4 => {
                r4cftk(
                    n,
                    nReduced,
                    x,
                    y,
                    x.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    y.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    x.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    dim,
                );
            }
            5 => {
                r5cftk(
                    n,
                    nReduced,
                    x,
                    y,
                    x.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    y.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    x.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((4 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((4 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    dim,
                );
            }
            8 => {
                r8cftk(
                    n,
                    nReduced,
                    x,
                    y,
                    x.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    y.offset(stepAlong as isize) as *mut ::core::ffi::c_float,
                    x.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((2 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((3 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((4 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((4 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((5 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((5 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((6 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((6 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    x.offset((7 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    y.offset((7 as ::core::ffi::c_int * stepAlong) as isize)
                        as *mut ::core::ffi::c_float,
                    dim,
                );
            }
            6 => {
                printf(
                    b"\ntransfer error detected in mdftkd\n\n\0" as *const u8
                        as *const ::core::ffi::c_char,
                );
                return;
            }
            7 | _ => {
                rpcftk(n, nReduced, pFac, stepAlong, x, y, dim);
            }
        }
    }
}
pub unsafe extern "C" fn r2cftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut x0: *mut ::core::ffi::c_float,
    mut y0: *mut ::core::ffi::c_float,
    mut x1: *mut ::core::ffi::c_float,
    mut y1: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut nRed2: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRed2: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut c: ::core::ffi::c_float = 0.;
    let mut is: ::core::ffi::c_float = 0.;
    let mut iu: ::core::ffi::c_float = 0.;
    let mut rs: ::core::ffi::c_float = 0.;
    let mut ru: ::core::ffi::c_float = 0.;
    let mut sepAlong: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRed2: ::core::ffi::c_float = 0.;
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRed2 = nReduced * 2 as ::core::ffi::c_int;
    fnRed2 = nRed2 as ::core::ffi::c_float;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    sepNRed2 = sep * nRed2;
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRed2) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c = cos(angle) as ::core::ffi::c_float;
            sepAlong = sin(angle) as ::core::ffi::c_float;
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        rs = *x0.offset(k as isize) + *x1.offset(k as isize);
                        is = *y0.offset(k as isize) + *y1.offset(k as isize);
                        ru = *x0.offset(k as isize) - *x1.offset(k as isize);
                        iu = *y0.offset(k as isize) - *y1.offset(k as isize);
                        *x0.offset(k as isize) = rs;
                        *y0.offset(k as isize) = is;
                        if zero == 0 {
                            *x1.offset(k as isize) = ru * c + iu * sepAlong;
                            *y1.offset(k as isize) = iu * c - ru * sepAlong;
                        } else {
                            *x1.offset(k as isize) = ru;
                            *y1.offset(k as isize) = iu;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRed2;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            c = -c;
            itrip += 1;
        }
        j += 1;
    }
}
pub unsafe extern "C" fn r3cftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut x0: *mut ::core::ffi::c_float,
    mut y0: *mut ::core::ffi::c_float,
    mut x1: *mut ::core::ffi::c_float,
    mut y1: *mut ::core::ffi::c_float,
    mut x2: *mut ::core::ffi::c_float,
    mut y2: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut nRed3: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRed3: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut a: ::core::ffi::c_float = -0.5f64 as ::core::ffi::c_float;
    let mut b: ::core::ffi::c_float = 0.86602540f32;
    let mut c1: ::core::ffi::c_float = 0.;
    let mut c2: ::core::ffi::c_float = 0.;
    let mut s1: ::core::ffi::c_float = 0.;
    let mut s2: ::core::ffi::c_float = 0.;
    let mut t: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut i0: ::core::ffi::c_float = 0.;
    let mut i1: ::core::ffi::c_float = 0.;
    let mut i2: ::core::ffi::c_float = 0.;
    let mut ia: ::core::ffi::c_float = 0.;
    let mut ib: ::core::ffi::c_float = 0.;
    let mut is: ::core::ffi::c_float = 0.;
    let mut r0: ::core::ffi::c_float = 0.;
    let mut r1: ::core::ffi::c_float = 0.;
    let mut r2: ::core::ffi::c_float = 0.;
    let mut ra: ::core::ffi::c_float = 0.;
    let mut rb: ::core::ffi::c_float = 0.;
    let mut rs: ::core::ffi::c_float = 0.;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRed3: ::core::ffi::c_float = 0.;
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRed3 = nReduced * 3 as ::core::ffi::c_int;
    fnRed3 = nRed3 as ::core::ffi::c_float;
    sepNRed3 = sep * nRed3;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRed3) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c1 = cos(angle) as ::core::ffi::c_float;
            s1 = sin(angle) as ::core::ffi::c_float;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        r0 = *x0.offset(k as isize);
                        i0 = *y0.offset(k as isize);
                        rs = *x1.offset(k as isize) + *x2.offset(k as isize);
                        is = *y1.offset(k as isize) + *y2.offset(k as isize);
                        *x0.offset(k as isize) = r0 + rs;
                        *y0.offset(k as isize) = i0 + is;
                        ra = r0 + rs * a;
                        ia = i0 + is * a;
                        rb = (*x1.offset(k as isize) - *x2.offset(k as isize)) * b;
                        ib = (*y1.offset(k as isize) - *y2.offset(k as isize)) * b;
                        if zero == 0 {
                            r1 = ra + ib;
                            i1 = ia - rb;
                            r2 = ra - ib;
                            i2 = ia + rb;
                            *x1.offset(k as isize) = r1 * c1 + i1 * s1;
                            *y1.offset(k as isize) = i1 * c1 - r1 * s1;
                            *x2.offset(k as isize) = r2 * c2 + i2 * s2;
                            *y2.offset(k as isize) = i2 * c2 - r2 * s2;
                        } else {
                            *x1.offset(k as isize) = ra + ib;
                            *y1.offset(k as isize) = ia - rb;
                            *x2.offset(k as isize) = ra - ib;
                            *y2.offset(k as isize) = ia + rb;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRed3;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            t = c1 * a + s1 * b;
            s1 = c1 * b - s1 * a;
            c1 = t;
            t = c2 * a - s2 * b;
            s2 = -c2 * b - s2 * a;
            c2 = t;
            itrip += 1;
        }
        j += 1;
    }
}
pub unsafe extern "C" fn r4cftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut x0: *mut ::core::ffi::c_float,
    mut y0: *mut ::core::ffi::c_float,
    mut x1: *mut ::core::ffi::c_float,
    mut y1: *mut ::core::ffi::c_float,
    mut x2: *mut ::core::ffi::c_float,
    mut y2: *mut ::core::ffi::c_float,
    mut x3: *mut ::core::ffi::c_float,
    mut y3: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut nRed4: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRed4: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut c1: ::core::ffi::c_float = 0.;
    let mut c2: ::core::ffi::c_float = 0.;
    let mut c3: ::core::ffi::c_float = 0.;
    let mut s1: ::core::ffi::c_float = 0.;
    let mut s2: ::core::ffi::c_float = 0.;
    let mut s3: ::core::ffi::c_float = 0.;
    let mut t: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut i1: ::core::ffi::c_float = 0.;
    let mut i2: ::core::ffi::c_float = 0.;
    let mut i3: ::core::ffi::c_float = 0.;
    let mut is0: ::core::ffi::c_float = 0.;
    let mut is1: ::core::ffi::c_float = 0.;
    let mut iu0: ::core::ffi::c_float = 0.;
    let mut iu1: ::core::ffi::c_float = 0.;
    let mut r1: ::core::ffi::c_float = 0.;
    let mut r2: ::core::ffi::c_float = 0.;
    let mut r3: ::core::ffi::c_float = 0.;
    let mut rs0: ::core::ffi::c_float = 0.;
    let mut rs1: ::core::ffi::c_float = 0.;
    let mut ru0: ::core::ffi::c_float = 0.;
    let mut ru1: ::core::ffi::c_float = 0.;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRed4: ::core::ffi::c_float = 0.;
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRed4 = nReduced * 4 as ::core::ffi::c_int;
    fnRed4 = nRed4 as ::core::ffi::c_float;
    sepNRed4 = sep * nRed4;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRed4) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c1 = cos(angle) as ::core::ffi::c_float;
            s1 = sin(angle) as ::core::ffi::c_float;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        rs0 = *x0.offset(k as isize) + *x2.offset(k as isize);
                        is0 = *y0.offset(k as isize) + *y2.offset(k as isize);
                        ru0 = *x0.offset(k as isize) - *x2.offset(k as isize);
                        iu0 = *y0.offset(k as isize) - *y2.offset(k as isize);
                        rs1 = *x1.offset(k as isize) + *x3.offset(k as isize);
                        is1 = *y1.offset(k as isize) + *y3.offset(k as isize);
                        ru1 = *x1.offset(k as isize) - *x3.offset(k as isize);
                        iu1 = *y1.offset(k as isize) - *y3.offset(k as isize);
                        *x0.offset(k as isize) = rs0 + rs1;
                        *y0.offset(k as isize) = is0 + is1;
                        if zero == 0 {
                            r1 = ru0 + iu1;
                            i1 = iu0 - ru1;
                            r2 = rs0 - rs1;
                            i2 = is0 - is1;
                            r3 = ru0 - iu1;
                            i3 = iu0 + ru1;
                            *x2.offset(k as isize) = r1 * c1 + i1 * s1;
                            *y2.offset(k as isize) = i1 * c1 - r1 * s1;
                            *x1.offset(k as isize) = r2 * c2 + i2 * s2;
                            *y1.offset(k as isize) = i2 * c2 - r2 * s2;
                            *x3.offset(k as isize) = r3 * c3 + i3 * s3;
                            *y3.offset(k as isize) = i3 * c3 - r3 * s3;
                        } else {
                            *x2.offset(k as isize) = ru0 + iu1;
                            *y2.offset(k as isize) = iu0 - ru1;
                            *x1.offset(k as isize) = rs0 - rs1;
                            *y1.offset(k as isize) = is0 - is1;
                            *x3.offset(k as isize) = ru0 - iu1;
                            *y3.offset(k as isize) = iu0 + ru1;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRed4;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            t = c1;
            c1 = s1;
            s1 = t;
            c2 = -c2;
            t = c3;
            c3 = -s3;
            s3 = -t;
            itrip += 1;
        }
        j += 1;
    }
}
pub unsafe extern "C" fn r5cftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut x0: *mut ::core::ffi::c_float,
    mut y0: *mut ::core::ffi::c_float,
    mut x1: *mut ::core::ffi::c_float,
    mut y1: *mut ::core::ffi::c_float,
    mut x2: *mut ::core::ffi::c_float,
    mut y2: *mut ::core::ffi::c_float,
    mut x3: *mut ::core::ffi::c_float,
    mut y3: *mut ::core::ffi::c_float,
    mut x4: *mut ::core::ffi::c_float,
    mut y4: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut nRed5: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRed5: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut a1: ::core::ffi::c_float = 0.30901699f32;
    let mut a2: ::core::ffi::c_float = -0.80901699f64 as ::core::ffi::c_float;
    let mut b1: ::core::ffi::c_float = 0.95105652f32;
    let mut b2: ::core::ffi::c_float = 0.58778525f32;
    let mut c1: ::core::ffi::c_float = 0.;
    let mut c2: ::core::ffi::c_float = 0.;
    let mut c3: ::core::ffi::c_float = 0.;
    let mut c4: ::core::ffi::c_float = 0.;
    let mut s1: ::core::ffi::c_float = 0.;
    let mut s2: ::core::ffi::c_float = 0.;
    let mut s3: ::core::ffi::c_float = 0.;
    let mut s4: ::core::ffi::c_float = 0.;
    let mut t: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut r0: ::core::ffi::c_float = 0.;
    let mut r1: ::core::ffi::c_float = 0.;
    let mut r2: ::core::ffi::c_float = 0.;
    let mut r3: ::core::ffi::c_float = 0.;
    let mut r4: ::core::ffi::c_float = 0.;
    let mut ra1: ::core::ffi::c_float = 0.;
    let mut ra2: ::core::ffi::c_float = 0.;
    let mut rb1: ::core::ffi::c_float = 0.;
    let mut rb2: ::core::ffi::c_float = 0.;
    let mut rs1: ::core::ffi::c_float = 0.;
    let mut rs2: ::core::ffi::c_float = 0.;
    let mut ru1: ::core::ffi::c_float = 0.;
    let mut ru2: ::core::ffi::c_float = 0.;
    let mut i0: ::core::ffi::c_float = 0.;
    let mut i1: ::core::ffi::c_float = 0.;
    let mut i2: ::core::ffi::c_float = 0.;
    let mut i3: ::core::ffi::c_float = 0.;
    let mut i4: ::core::ffi::c_float = 0.;
    let mut ia1: ::core::ffi::c_float = 0.;
    let mut ia2: ::core::ffi::c_float = 0.;
    let mut ib1: ::core::ffi::c_float = 0.;
    let mut ib2: ::core::ffi::c_float = 0.;
    let mut is1: ::core::ffi::c_float = 0.;
    let mut is2: ::core::ffi::c_float = 0.;
    let mut iu1: ::core::ffi::c_float = 0.;
    let mut iu2: ::core::ffi::c_float = 0.;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRed5: ::core::ffi::c_float = 0.;
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRed5 = nReduced * 5 as ::core::ffi::c_int;
    fnRed5 = nRed5 as ::core::ffi::c_float;
    sepNRed5 = sep * nRed5;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRed5) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c1 = cos(angle) as ::core::ffi::c_float;
            s1 = sin(angle) as ::core::ffi::c_float;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
            c4 = c2 * c2 - s2 * s2;
            s4 = s2 * c2 + c2 * s2;
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        r0 = *x0.offset(k as isize);
                        i0 = *y0.offset(k as isize);
                        rs1 = *x1.offset(k as isize) + *x4.offset(k as isize);
                        is1 = *y1.offset(k as isize) + *y4.offset(k as isize);
                        ru1 = *x1.offset(k as isize) - *x4.offset(k as isize);
                        iu1 = *y1.offset(k as isize) - *y4.offset(k as isize);
                        rs2 = *x2.offset(k as isize) + *x3.offset(k as isize);
                        is2 = *y2.offset(k as isize) + *y3.offset(k as isize);
                        ru2 = *x2.offset(k as isize) - *x3.offset(k as isize);
                        iu2 = *y2.offset(k as isize) - *y3.offset(k as isize);
                        *x0.offset(k as isize) = r0 + rs1 + rs2;
                        *y0.offset(k as isize) = i0 + is1 + is2;
                        ra1 = r0 + rs1 * a1 + rs2 * a2;
                        ia1 = i0 + is1 * a1 + is2 * a2;
                        ra2 = r0 + rs1 * a2 + rs2 * a1;
                        ia2 = i0 + is1 * a2 + is2 * a1;
                        rb1 = ru1 * b1 + ru2 * b2;
                        ib1 = iu1 * b1 + iu2 * b2;
                        rb2 = ru1 * b2 - ru2 * b1;
                        ib2 = iu1 * b2 - iu2 * b1;
                        if zero == 0 {
                            r1 = ra1 + ib1;
                            i1 = ia1 - rb1;
                            r2 = ra2 + ib2;
                            i2 = ia2 - rb2;
                            r3 = ra2 - ib2;
                            i3 = ia2 + rb2;
                            r4 = ra1 - ib1;
                            i4 = ia1 + rb1;
                            *x1.offset(k as isize) = r1 * c1 + i1 * s1;
                            *y1.offset(k as isize) = i1 * c1 - r1 * s1;
                            *x2.offset(k as isize) = r2 * c2 + i2 * s2;
                            *y2.offset(k as isize) = i2 * c2 - r2 * s2;
                            *x3.offset(k as isize) = r3 * c3 + i3 * s3;
                            *y3.offset(k as isize) = i3 * c3 - r3 * s3;
                            *x4.offset(k as isize) = r4 * c4 + i4 * s4;
                            *y4.offset(k as isize) = i4 * c4 - r4 * s4;
                        } else {
                            *x1.offset(k as isize) = ra1 + ib1;
                            *y1.offset(k as isize) = ia1 - rb1;
                            *x2.offset(k as isize) = ra2 + ib2;
                            *y2.offset(k as isize) = ia2 - rb2;
                            *x3.offset(k as isize) = ra2 - ib2;
                            *y3.offset(k as isize) = ia2 + rb2;
                            *x4.offset(k as isize) = ra1 - ib1;
                            *y4.offset(k as isize) = ia1 + rb1;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRed5;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            t = c1 * a1 + s1 * b1;
            s1 = c1 * b1 - s1 * a1;
            c1 = t;
            t = c2 * a2 + s2 * b2;
            s2 = c2 * b2 - s2 * a2;
            c2 = t;
            t = c3 * a2 - s3 * b2;
            s3 = -c3 * b2 - s3 * a2;
            c3 = t;
            t = c4 * a1 - s4 * b1;
            s4 = -c4 * b1 - s4 * a1;
            c4 = t;
            itrip += 1;
        }
        j += 1;
    }
}
pub unsafe extern "C" fn r8cftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut x0: *mut ::core::ffi::c_float,
    mut y0: *mut ::core::ffi::c_float,
    mut x1: *mut ::core::ffi::c_float,
    mut y1: *mut ::core::ffi::c_float,
    mut x2: *mut ::core::ffi::c_float,
    mut y2: *mut ::core::ffi::c_float,
    mut x3: *mut ::core::ffi::c_float,
    mut y3: *mut ::core::ffi::c_float,
    mut x4: *mut ::core::ffi::c_float,
    mut y4: *mut ::core::ffi::c_float,
    mut x5: *mut ::core::ffi::c_float,
    mut y5: *mut ::core::ffi::c_float,
    mut x6: *mut ::core::ffi::c_float,
    mut y6: *mut ::core::ffi::c_float,
    mut x7: *mut ::core::ffi::c_float,
    mut y7: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut nRed8: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRed8: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut c1: ::core::ffi::c_float = 0.;
    let mut c2: ::core::ffi::c_float = 0.;
    let mut c3: ::core::ffi::c_float = 0.;
    let mut c4: ::core::ffi::c_float = 0.;
    let mut c5: ::core::ffi::c_float = 0.;
    let mut c6: ::core::ffi::c_float = 0.;
    let mut c7: ::core::ffi::c_float = 0.;
    let mut e: ::core::ffi::c_float = 0.70710678f32;
    let mut s1: ::core::ffi::c_float = 0.;
    let mut s2: ::core::ffi::c_float = 0.;
    let mut s3: ::core::ffi::c_float = 0.;
    let mut s4: ::core::ffi::c_float = 0.;
    let mut s5: ::core::ffi::c_float = 0.;
    let mut s6: ::core::ffi::c_float = 0.;
    let mut s7: ::core::ffi::c_float = 0.;
    let mut t: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut r1: ::core::ffi::c_float = 0.;
    let mut r2: ::core::ffi::c_float = 0.;
    let mut r3: ::core::ffi::c_float = 0.;
    let mut r4: ::core::ffi::c_float = 0.;
    let mut r5: ::core::ffi::c_float = 0.;
    let mut r6: ::core::ffi::c_float = 0.;
    let mut r7: ::core::ffi::c_float = 0.;
    let mut rs0: ::core::ffi::c_float = 0.;
    let mut rs1: ::core::ffi::c_float = 0.;
    let mut rs2: ::core::ffi::c_float = 0.;
    let mut rs3: ::core::ffi::c_float = 0.;
    let mut ru0: ::core::ffi::c_float = 0.;
    let mut ru1: ::core::ffi::c_float = 0.;
    let mut ru2: ::core::ffi::c_float = 0.;
    let mut ru3: ::core::ffi::c_float = 0.;
    let mut i1: ::core::ffi::c_float = 0.;
    let mut i2: ::core::ffi::c_float = 0.;
    let mut i3: ::core::ffi::c_float = 0.;
    let mut i4: ::core::ffi::c_float = 0.;
    let mut i5: ::core::ffi::c_float = 0.;
    let mut i6: ::core::ffi::c_float = 0.;
    let mut i7: ::core::ffi::c_float = 0.;
    let mut is0: ::core::ffi::c_float = 0.;
    let mut is1: ::core::ffi::c_float = 0.;
    let mut is2: ::core::ffi::c_float = 0.;
    let mut is3: ::core::ffi::c_float = 0.;
    let mut iu0: ::core::ffi::c_float = 0.;
    let mut iu1: ::core::ffi::c_float = 0.;
    let mut iu2: ::core::ffi::c_float = 0.;
    let mut iu3: ::core::ffi::c_float = 0.;
    let mut rss0: ::core::ffi::c_float = 0.;
    let mut rss1: ::core::ffi::c_float = 0.;
    let mut rsu0: ::core::ffi::c_float = 0.;
    let mut rsu1: ::core::ffi::c_float = 0.;
    let mut rus0: ::core::ffi::c_float = 0.;
    let mut rus1: ::core::ffi::c_float = 0.;
    let mut ruu0: ::core::ffi::c_float = 0.;
    let mut ruu1: ::core::ffi::c_float = 0.;
    let mut iss0: ::core::ffi::c_float = 0.;
    let mut iss1: ::core::ffi::c_float = 0.;
    let mut isu0: ::core::ffi::c_float = 0.;
    let mut isu1: ::core::ffi::c_float = 0.;
    let mut ius0: ::core::ffi::c_float = 0.;
    let mut ius1: ::core::ffi::c_float = 0.;
    let mut iuu0: ::core::ffi::c_float = 0.;
    let mut iuu1: ::core::ffi::c_float = 0.;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRed8: ::core::ffi::c_float = 0.;
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRed8 = nReduced * 8 as ::core::ffi::c_int;
    fnRed8 = nRed8 as ::core::ffi::c_float;
    sepNRed8 = sep * nRed8;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRed8) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c1 = cos(angle) as ::core::ffi::c_float;
            s1 = sin(angle) as ::core::ffi::c_float;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
            c4 = c2 * c2 - s2 * s2;
            s4 = s2 * c2 + c2 * s2;
            c5 = c4 * c1 - s4 * s1;
            s5 = s4 * c1 + c4 * s1;
            c6 = c4 * c2 - s4 * s2;
            s6 = s4 * c2 + c4 * s2;
            c7 = c4 * c3 - s4 * s3;
            s7 = s4 * c3 + c4 * s3;
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        rs0 = *x0.offset(k as isize) + *x4.offset(k as isize);
                        is0 = *y0.offset(k as isize) + *y4.offset(k as isize);
                        ru0 = *x0.offset(k as isize) - *x4.offset(k as isize);
                        iu0 = *y0.offset(k as isize) - *y4.offset(k as isize);
                        rs1 = *x1.offset(k as isize) + *x5.offset(k as isize);
                        is1 = *y1.offset(k as isize) + *y5.offset(k as isize);
                        ru1 = *x1.offset(k as isize) - *x5.offset(k as isize);
                        iu1 = *y1.offset(k as isize) - *y5.offset(k as isize);
                        rs2 = *x2.offset(k as isize) + *x6.offset(k as isize);
                        is2 = *y2.offset(k as isize) + *y6.offset(k as isize);
                        ru2 = *x2.offset(k as isize) - *x6.offset(k as isize);
                        iu2 = *y2.offset(k as isize) - *y6.offset(k as isize);
                        rs3 = *x3.offset(k as isize) + *x7.offset(k as isize);
                        is3 = *y3.offset(k as isize) + *y7.offset(k as isize);
                        ru3 = *x3.offset(k as isize) - *x7.offset(k as isize);
                        iu3 = *y3.offset(k as isize) - *y7.offset(k as isize);
                        rss0 = rs0 + rs2;
                        iss0 = is0 + is2;
                        rsu0 = rs0 - rs2;
                        isu0 = is0 - is2;
                        rss1 = rs1 + rs3;
                        iss1 = is1 + is3;
                        rsu1 = rs1 - rs3;
                        isu1 = is1 - is3;
                        rus0 = ru0 - iu2;
                        ius0 = iu0 + ru2;
                        ruu0 = ru0 + iu2;
                        iuu0 = iu0 - ru2;
                        rus1 = ru1 - iu3;
                        ius1 = iu1 + ru3;
                        ruu1 = ru1 + iu3;
                        iuu1 = iu1 - ru3;
                        t = (rus1 + ius1) * e;
                        ius1 = (ius1 - rus1) * e;
                        rus1 = t;
                        t = (ruu1 + iuu1) * e;
                        iuu1 = (iuu1 - ruu1) * e;
                        ruu1 = t;
                        *x0.offset(k as isize) = rss0 + rss1;
                        *y0.offset(k as isize) = iss0 + iss1;
                        if zero == 0 {
                            r1 = ruu0 + ruu1;
                            i1 = iuu0 + iuu1;
                            r2 = rsu0 + isu1;
                            i2 = isu0 - rsu1;
                            r3 = rus0 + ius1;
                            i3 = ius0 - rus1;
                            r4 = rss0 - rss1;
                            i4 = iss0 - iss1;
                            r5 = ruu0 - ruu1;
                            i5 = iuu0 - iuu1;
                            r6 = rsu0 - isu1;
                            i6 = isu0 + rsu1;
                            r7 = rus0 - ius1;
                            i7 = ius0 + rus1;
                            *x4.offset(k as isize) = r1 * c1 + i1 * s1;
                            *y4.offset(k as isize) = i1 * c1 - r1 * s1;
                            *x2.offset(k as isize) = r2 * c2 + i2 * s2;
                            *y2.offset(k as isize) = i2 * c2 - r2 * s2;
                            *x6.offset(k as isize) = r3 * c3 + i3 * s3;
                            *y6.offset(k as isize) = i3 * c3 - r3 * s3;
                            *x1.offset(k as isize) = r4 * c4 + i4 * s4;
                            *y1.offset(k as isize) = i4 * c4 - r4 * s4;
                            *x5.offset(k as isize) = r5 * c5 + i5 * s5;
                            *y5.offset(k as isize) = i5 * c5 - r5 * s5;
                            *x3.offset(k as isize) = r6 * c6 + i6 * s6;
                            *y3.offset(k as isize) = i6 * c6 - r6 * s6;
                            *x7.offset(k as isize) = r7 * c7 + i7 * s7;
                            *y7.offset(k as isize) = i7 * c7 - r7 * s7;
                        } else {
                            *x4.offset(k as isize) = ruu0 + ruu1;
                            *y4.offset(k as isize) = iuu0 + iuu1;
                            *x2.offset(k as isize) = rsu0 + isu1;
                            *y2.offset(k as isize) = isu0 - rsu1;
                            *x6.offset(k as isize) = rus0 + ius1;
                            *y6.offset(k as isize) = ius0 - rus1;
                            *x1.offset(k as isize) = rss0 - rss1;
                            *y1.offset(k as isize) = iss0 - iss1;
                            *x5.offset(k as isize) = ruu0 - ruu1;
                            *y5.offset(k as isize) = iuu0 - iuu1;
                            *x3.offset(k as isize) = rsu0 - isu1;
                            *y3.offset(k as isize) = isu0 + rsu1;
                            *x7.offset(k as isize) = rus0 - ius1;
                            *y7.offset(k as isize) = ius0 + rus1;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRed8;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            t = (c1 + s1) * e;
            s1 = (c1 - s1) * e;
            c1 = t;
            t = s2;
            s2 = c2;
            c2 = t;
            t = (-c3 + s3) * e;
            s3 = (c3 + s3) * e;
            c3 = t;
            c4 = -c4;
            t = -(c5 + s5) * e;
            s5 = (-c5 + s5) * e;
            c5 = t;
            t = -s6;
            s6 = -c6;
            c6 = t;
            t = (c7 - s7) * e;
            s7 = -(c7 + s7) * e;
            c7 = t;
            itrip += 1;
        }
        j += 1;
    }
}
pub unsafe extern "C" fn rpcftk(
    mut n: ::core::ffi::c_int,
    mut nReduced: ::core::ffi::c_int,
    mut pFac: ::core::ffi::c_int,
    mut stepAlong: ::core::ffi::c_int,
    mut x: *mut ::core::ffi::c_float,
    mut y: *mut ::core::ffi::c_float,
    mut dim: *mut ::core::ffi::c_int,
) {
    let mut fold: ::core::ffi::c_int = 0;
    let mut zero: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_double = 0.;
    let mut is: ::core::ffi::c_float = 0.;
    let mut iu: ::core::ffi::c_float = 0.;
    let mut rs: ::core::ffi::c_float = 0.;
    let mut ru: ::core::ffi::c_float = 0.;
    let mut t: ::core::ffi::c_float = 0.;
    let mut twopi: ::core::ffi::c_float = 6.2831853f32;
    let mut xt: ::core::ffi::c_float = 0.;
    let mut yt: ::core::ffi::c_float = 0.;
    let mut fu: ::core::ffi::c_float = 0.;
    let mut fp: ::core::ffi::c_float = 0.;
    let mut fjm1: ::core::ffi::c_float = 0.;
    let mut fnRedp: ::core::ffi::c_float = 0.;
    let mut j: ::core::ffi::c_int = 0;
    let mut jj: ::core::ffi::c_int = 0;
    let mut k0: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut nRedOv2p1: ::core::ffi::c_int = 0;
    let mut nRedp: ::core::ffi::c_int = 0;
    let mut pm: ::core::ffi::c_int = 0;
    let mut pp: ::core::ffi::c_int = 0;
    let mut u: ::core::ffi::c_int = 0;
    let mut v: ::core::ffi::c_int = 0;
    let mut k1: ::core::ffi::c_int = 0;
    let mut sepBetween: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut limAlong: ::core::ffi::c_int = 0;
    let mut sepNRedp: ::core::ffi::c_int = 0;
    let mut totFloats: ::core::ffi::c_int = 0;
    let mut extentBetween: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut nSep: ::core::ffi::c_int = 0;
    let mut aa: [[::core::ffi::c_float; 10]; 10] = [[0.; 10]; 10];
    let mut bb: [[::core::ffi::c_float; 10]; 10] = [[0.; 10]; 10];
    let mut a: [::core::ffi::c_float; 19] = [0.; 19];
    let mut b: [::core::ffi::c_float; 19] = [0.; 19];
    let mut c: [::core::ffi::c_float; 19] = [0.; 19];
    let mut sepAlong: [::core::ffi::c_float; 19] = [0.; 19];
    let mut ia: [::core::ffi::c_float; 10] = [0.; 10];
    let mut ib: [::core::ffi::c_float; 10] = [0.; 10];
    let mut ra: [::core::ffi::c_float; 10] = [0.; 10];
    let mut rb: [::core::ffi::c_float; 10] = [0.; 10];
    let mut itrip: ::core::ffi::c_int = 0;
    let mut ntrip: ::core::ffi::c_int = 0;
    totFloats = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    limAlong = *dim.offset(3 as ::core::ffi::c_int as isize);
    extentBetween = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    sepBetween = *dim.offset(5 as ::core::ffi::c_int as isize);
    nSep = n * sep;
    nRedOv2p1 = nReduced / 2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int;
    nRedp = nReduced * pFac;
    fnRedp = nRedp as ::core::ffi::c_float;
    sepNRedp = sep * nRedp;
    pp = pFac / 2 as ::core::ffi::c_int;
    pm = pFac - 1 as ::core::ffi::c_int;
    fp = pFac as ::core::ffi::c_float;
    fu = 0.0f32;
    u = 1 as ::core::ffi::c_int;
    while u <= pp {
        fu = (fu as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fu / fp) as ::core::ffi::c_double;
        jj = pFac - u;
        a[u as usize] = cos(angle) as ::core::ffi::c_float;
        b[u as usize] = sin(angle) as ::core::ffi::c_float;
        a[jj as usize] = a[u as usize];
        b[jj as usize] = -b[u as usize];
        u += 1;
    }
    u = 1 as ::core::ffi::c_int;
    while u <= pp {
        v = 1 as ::core::ffi::c_int;
        while v <= pp {
            jj = u * v - u * v / pFac * pFac;
            aa[v as usize][u as usize] = a[jj as usize];
            bb[v as usize][u as usize] = b[jj as usize];
            v += 1;
        }
        u += 1;
    }
    fjm1 = -1.0f64 as ::core::ffi::c_float;
    j = 1 as ::core::ffi::c_int;
    while j <= nRedOv2p1 {
        fold = (j > 1 as ::core::ffi::c_int
            && 2 as ::core::ffi::c_int * j < nReduced + 2 as ::core::ffi::c_int)
            as ::core::ffi::c_int;
        k0 = (j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
        fjm1 = (fjm1 as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_float;
        angle = (twopi * fjm1 / fnRedp) as ::core::ffi::c_double;
        zero = (angle == 0.0f64) as ::core::ffi::c_int;
        if zero == 0 {
            c[1 as ::core::ffi::c_int as usize] = cos(angle) as ::core::ffi::c_float;
            sepAlong[1 as ::core::ffi::c_int as usize] = sin(angle) as ::core::ffi::c_float;
            u = 2 as ::core::ffi::c_int;
            while u <= pm {
                c[u as usize] = c[(u - 1 as ::core::ffi::c_int) as usize]
                    * c[1 as ::core::ffi::c_int as usize]
                    - sepAlong[(u - 1 as ::core::ffi::c_int) as usize]
                        * sepAlong[1 as ::core::ffi::c_int as usize];
                sepAlong[u as usize] = sepAlong[(u - 1 as ::core::ffi::c_int) as usize]
                    * c[1 as ::core::ffi::c_int as usize]
                    + c[(u - 1 as ::core::ffi::c_int) as usize]
                        * sepAlong[1 as ::core::ffi::c_int as usize];
                u += 1;
            }
        }
        ntrip = if fold != 0 {
            2 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        };
        itrip = 0 as ::core::ffi::c_int;
        while itrip < ntrip {
            kk = k0;
            while kk <= nSep {
                l = kk;
                while l <= totFloats {
                    k1 = l + extentBetween;
                    k = l - 1 as ::core::ffi::c_int;
                    while k < k1 {
                        xt = *x.offset(k as isize);
                        yt = *y.offset(k as isize);
                        rs = *x.offset((k + stepAlong) as isize)
                            + *x.offset((k + stepAlong * pm) as isize);
                        is = *y.offset((k + stepAlong) as isize)
                            + *y.offset((k + stepAlong * pm) as isize);
                        ru = *x.offset((k + stepAlong) as isize)
                            - *x.offset((k + stepAlong * pm) as isize);
                        iu = *y.offset((k + stepAlong) as isize)
                            - *y.offset((k + stepAlong * pm) as isize);
                        u = 1 as ::core::ffi::c_int;
                        while u <= pp {
                            ra[u as usize] =
                                xt + rs * aa[u as usize][1 as ::core::ffi::c_int as usize];
                            ia[u as usize] =
                                yt + is * aa[u as usize][1 as ::core::ffi::c_int as usize];
                            rb[u as usize] = ru * bb[u as usize][1 as ::core::ffi::c_int as usize];
                            ib[u as usize] = iu * bb[u as usize][1 as ::core::ffi::c_int as usize];
                            u += 1;
                        }
                        xt = xt + rs;
                        yt = yt + is;
                        u = 2 as ::core::ffi::c_int;
                        while u <= pp {
                            jj = pFac - u;
                            rs = *x.offset((k + u * stepAlong) as isize)
                                + *x.offset((k + jj * stepAlong) as isize);
                            is = *y.offset((k + u * stepAlong) as isize)
                                + *y.offset((k + jj * stepAlong) as isize);
                            ru = *x.offset((k + u * stepAlong) as isize)
                                - *x.offset((k + jj * stepAlong) as isize);
                            iu = *y.offset((k + u * stepAlong) as isize)
                                - *y.offset((k + jj * stepAlong) as isize);
                            xt = xt + rs;
                            yt = yt + is;
                            v = 1 as ::core::ffi::c_int;
                            while v <= pp {
                                ra[v as usize] = ra[v as usize] + rs * aa[v as usize][u as usize];
                                ia[v as usize] = ia[v as usize] + is * aa[v as usize][u as usize];
                                rb[v as usize] = rb[v as usize] + ru * bb[v as usize][u as usize];
                                ib[v as usize] = ib[v as usize] + iu * bb[v as usize][u as usize];
                                v += 1;
                            }
                            u += 1;
                        }
                        *x.offset(k as isize) = xt;
                        *y.offset(k as isize) = yt;
                        u = 1 as ::core::ffi::c_int;
                        while u <= pp {
                            jj = pFac - u;
                            if zero == 0 {
                                xt = ra[u as usize] + ib[u as usize];
                                yt = ia[u as usize] - rb[u as usize];
                                *x.offset((k + u * stepAlong) as isize) =
                                    xt * c[u as usize] + yt * sepAlong[u as usize];
                                *y.offset((k + u * stepAlong) as isize) =
                                    yt * c[u as usize] - xt * sepAlong[u as usize];
                                xt = ra[u as usize] - ib[u as usize];
                                yt = ia[u as usize] + rb[u as usize];
                                *x.offset((k + jj * stepAlong) as isize) =
                                    xt * c[jj as usize] + yt * sepAlong[jj as usize];
                                *y.offset((k + jj * stepAlong) as isize) =
                                    yt * c[jj as usize] - xt * sepAlong[jj as usize];
                            } else {
                                *x.offset((k + u * stepAlong) as isize) =
                                    ra[u as usize] + ib[u as usize];
                                *y.offset((k + u * stepAlong) as isize) =
                                    ia[u as usize] - rb[u as usize];
                                *x.offset((k + jj * stepAlong) as isize) =
                                    ra[u as usize] - ib[u as usize];
                                *y.offset((k + jj * stepAlong) as isize) =
                                    ia[u as usize] + rb[u as usize];
                            }
                            u += 1;
                        }
                        k += sepBetween;
                    }
                    l += limAlong;
                }
                kk += sepNRedp;
            }
            if fold == 0 {
                break;
            }
            k0 = (nReduced + 1 as ::core::ffi::c_int - j) * sep + 1 as ::core::ffi::c_int;
            u = 1 as ::core::ffi::c_int;
            while u <= pm {
                t = c[u as usize] * a[u as usize] + sepAlong[u as usize] * b[u as usize];
                sepAlong[u as usize] =
                    -sepAlong[u as usize] * a[u as usize] + c[u as usize] * b[u as usize];
                c[u as usize] = t;
                u += 1;
            }
            itrip += 1;
        }
        j += 1;
    }
}
