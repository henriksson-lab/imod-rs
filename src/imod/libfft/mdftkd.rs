//! Translation of `IMOD/libfft/mdftkd.c` -- the multi-dimensional complex
//! Fourier transform kernel driver and its radix kernels.
//!
//! The C source hands each kernel several `float *` that all point into the
//! **same** caller array at different biases (`&x[stepAlong]`,
//! `&x[2 * stepAlong]`, ...), and `cmplft` itself is always called with
//! `y = &x[1]`, so the real and imaginary walkers alias too.  There is no safe
//! Rust spelling for several `&mut [f32]` over one buffer, so the whole array
//! is one `&mut [f32]` and every walker is a `usize` base index into it: the
//! bias the C applies with `&x[i * stepAlong]` is applied at the same place,
//! in `mdftkd`, as `x + i * step_along`.  A kernel's `x0[k]` is therefore
//! `data[x0 + k as usize]`, with the source's index arithmetic unchanged.

/// C `mdftkd`.
pub fn mdftkd(n: i32, factor: &[i32], dim: &[i32], data: &mut [f32], x: usize, y: usize) {
    let mut ind_fac: i32;
    let mut n_reduced: i32;
    let mut p_fac: i32;
    let mut step_along: i32;
    let sep_along: i32;

    sep_along = dim[2];
    ind_fac = 0;
    n_reduced = n;
    while factor[(ind_fac + 1) as usize] != 0 {
        ind_fac = ind_fac + 1;
        p_fac = factor[ind_fac as usize];
        n_reduced = n_reduced / p_fac;
        step_along = n_reduced * sep_along;
        match p_fac {
            1 => {}

            2 => {
                r2cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    dim,
                );
            }

            3 => {
                r3cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    dim,
                );
            }

            4 => {
                r4cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    dim,
                );
            }

            5 => {
                r5cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    x + (4 * step_along) as usize,
                    y + (4 * step_along) as usize,
                    dim,
                );
            }

            8 => {
                r8cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    x + (4 * step_along) as usize,
                    y + (4 * step_along) as usize,
                    x + (5 * step_along) as usize,
                    y + (5 * step_along) as usize,
                    x + (6 * step_along) as usize,
                    y + (6 * step_along) as usize,
                    x + (7 * step_along) as usize,
                    y + (7 * step_along) as usize,
                    dim,
                );
            }

            6 => {
                // `mdftkd.c:57`, `printf("\ntransfer error detected in
                // mdftkd\n\n")`.  `srfp` never emits a factor of 6 -- it
                // splits every composite into primes and regroups only powers
                // of two -- so this arm is unreachable, and the stream it
                // would print on is not exercised by any command.
                // `mdftkd.c:57`.  Through `libc::printf`, not Rust's
                // stdout: `cmplft` and `srfp` write to the C stream, and C
                // stdio is block-buffered under redirection while Rust's is
                // not, so mixing the two reorders a captured file.  The arm
                // is unreachable in practice — `srfp` splits composites into
                // primes and regroups only powers of two, so it never emits a
                // factor of 6 — but the stream is part of the behaviour.
                unsafe {
                    libc::printf(c"\ntransfer error detected in mdftkd\n\n".as_ptr());
                }
                return;
            }

            // `mdftkd.c:60-61`, `case 7:` falls through to `default:`.
            _ => {
                rpcftk(n, n_reduced, p_fac, step_along, data, x, y, dim);
            }
        }
    }
}

/// C `r2cftk`.
///
/// radix 2 multi-dimensional complex fourier transform kernel
pub fn r2cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    dim: &[i32],
) {
    let n_red2: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red2: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c: f32 = 0.0;
    let mut sep_along: f32 = 0.0;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red2: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red2 = n_reduced * 2;
    fn_red2 = n_red2 as f32;
    n_red_ov2p1 = n_reduced / 2 + 1;
    sep_n_red2 = sep * n_red2;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red2) as f64;
        let zero = angle == 0.0;
        if !zero {
            c = angle.cos() as f32;
            sep_along = angle.sin() as f32;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        let rs = data[x0 + k as usize] + data[x1 + k as usize];
                        let is = data[y0 + k as usize] + data[y1 + k as usize];
                        let ru = data[x0 + k as usize] - data[x1 + k as usize];
                        let iu = data[y0 + k as usize] - data[y1 + k as usize];
                        data[x0 + k as usize] = rs;
                        data[y0 + k as usize] = is;
                        if !zero {
                            data[x1 + k as usize] = ru * c + iu * sep_along;
                            data[y1 + k as usize] = iu * c - ru * sep_along;
                        } else {
                            data[x1 + k as usize] = ru;
                            data[y1 + k as usize] = iu;
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_red2;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            c = -c;
        }
    }
}

/// C `r3cftk`.
///
/// radix 3 multi-dimensional complex fourier transform kernel
pub fn r3cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    dim: &[i32],
) {
    let n_red3: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red3: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let a: f32 = -0.5;
    let b: f32 = 0.86602540;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red3: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red3 = n_reduced * 3;
    fn_red3 = n_red3 as f32;
    sep_n_red3 = sep * n_red3;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red3) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        let r0 = data[x0 + k as usize];
                        let i0 = data[y0 + k as usize];
                        let rs = data[x1 + k as usize] + data[x2 + k as usize];
                        let is = data[y1 + k as usize] + data[y2 + k as usize];
                        data[x0 + k as usize] = r0 + rs;
                        data[y0 + k as usize] = i0 + is;
                        let ra = r0 + rs * a;
                        let ia = i0 + is * a;
                        let rb = (data[x1 + k as usize] - data[x2 + k as usize]) * b;
                        let ib = (data[y1 + k as usize] - data[y2 + k as usize]) * b;
                        if !zero {
                            let r1 = ra + ib;
                            let i1 = ia - rb;
                            let r2 = ra - ib;
                            let i2 = ia + rb;
                            data[x1 + k as usize] = r1 * c1 + i1 * s1;
                            data[y1 + k as usize] = i1 * c1 - r1 * s1;
                            data[x2 + k as usize] = r2 * c2 + i2 * s2;
                            data[y2 + k as usize] = i2 * c2 - r2 * s2;
                        } else {
                            data[x1 + k as usize] = ra + ib;
                            data[y1 + k as usize] = ia - rb;
                            data[x2 + k as usize] = ra - ib;
                            data[y2 + k as usize] = ia + rb;
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_red3;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            t = c1 * a + s1 * b;
            s1 = c1 * b - s1 * a;
            c1 = t;
            t = c2 * a - s2 * b;
            s2 = -c2 * b - s2 * a;
            c2 = t;
        }
    }
}

/// C `r4cftk`.
///
/// radix 4 multi-dimensional complex fourier transform kernel
pub fn r4cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    dim: &[i32],
) {
    let n_red4: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red4: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red4: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red4 = n_reduced * 4;
    fn_red4 = n_red4 as f32;
    sep_n_red4 = sep * n_red4;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red4) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        let rs0 = data[x0 + k as usize] + data[x2 + k as usize];
                        let is0 = data[y0 + k as usize] + data[y2 + k as usize];
                        let ru0 = data[x0 + k as usize] - data[x2 + k as usize];
                        let iu0 = data[y0 + k as usize] - data[y2 + k as usize];
                        let rs1 = data[x1 + k as usize] + data[x3 + k as usize];
                        let is1 = data[y1 + k as usize] + data[y3 + k as usize];
                        let ru1 = data[x1 + k as usize] - data[x3 + k as usize];
                        let iu1 = data[y1 + k as usize] - data[y3 + k as usize];
                        data[x0 + k as usize] = rs0 + rs1;
                        data[y0 + k as usize] = is0 + is1;
                        if !zero {
                            let r1 = ru0 + iu1;
                            let i1 = iu0 - ru1;
                            let r2 = rs0 - rs1;
                            let i2 = is0 - is1;
                            let r3 = ru0 - iu1;
                            let i3 = iu0 + ru1;
                            data[x2 + k as usize] = r1 * c1 + i1 * s1;
                            data[y2 + k as usize] = i1 * c1 - r1 * s1;
                            data[x1 + k as usize] = r2 * c2 + i2 * s2;
                            data[y1 + k as usize] = i2 * c2 - r2 * s2;
                            data[x3 + k as usize] = r3 * c3 + i3 * s3;
                            data[y3 + k as usize] = i3 * c3 - r3 * s3;
                        } else {
                            data[x2 + k as usize] = ru0 + iu1;
                            data[y2 + k as usize] = iu0 - ru1;
                            data[x1 + k as usize] = rs0 - rs1;
                            data[y1 + k as usize] = is0 - is1;
                            data[x3 + k as usize] = ru0 - iu1;
                            data[y3 + k as usize] = iu0 + ru1;
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_red4;
            }

            k0 = (n_reduced + 1 - j) * sep + 1;
            t = c1;
            c1 = s1;
            s1 = t;
            c2 = -c2;
            t = c3;
            c3 = -s3;
            s3 = -t;
        }
    }
}

/// C `r5cftk`.
///
/// radix 5 multi-dimensional complex fourier transform kernel
pub fn r5cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    x4: usize,
    y4: usize,
    dim: &[i32],
) {
    let n_red5: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red5: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let a1: f32 = 0.30901699;
    let a2: f32 = -0.80901699;
    let b1: f32 = 0.95105652;
    let b2: f32 = 0.58778525;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut c4: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut s4: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red5: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red5 = n_reduced * 5;
    fn_red5 = n_red5 as f32;
    sep_n_red5 = sep * n_red5;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red5) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
            c4 = c2 * c2 - s2 * s2;
            s4 = s2 * c2 + c2 * s2;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        let r0 = data[x0 + k as usize];
                        let i0 = data[y0 + k as usize];
                        let rs1 = data[x1 + k as usize] + data[x4 + k as usize];
                        let is1 = data[y1 + k as usize] + data[y4 + k as usize];
                        let ru1 = data[x1 + k as usize] - data[x4 + k as usize];
                        let iu1 = data[y1 + k as usize] - data[y4 + k as usize];
                        let rs2 = data[x2 + k as usize] + data[x3 + k as usize];
                        let is2 = data[y2 + k as usize] + data[y3 + k as usize];
                        let ru2 = data[x2 + k as usize] - data[x3 + k as usize];
                        let iu2 = data[y2 + k as usize] - data[y3 + k as usize];
                        data[x0 + k as usize] = r0 + rs1 + rs2;
                        data[y0 + k as usize] = i0 + is1 + is2;
                        let ra1 = r0 + rs1 * a1 + rs2 * a2;
                        let ia1 = i0 + is1 * a1 + is2 * a2;
                        let ra2 = r0 + rs1 * a2 + rs2 * a1;
                        let ia2 = i0 + is1 * a2 + is2 * a1;
                        let rb1 = ru1 * b1 + ru2 * b2;
                        let ib1 = iu1 * b1 + iu2 * b2;
                        let rb2 = ru1 * b2 - ru2 * b1;
                        let ib2 = iu1 * b2 - iu2 * b1;
                        if !zero {
                            let r1 = ra1 + ib1;
                            let i1 = ia1 - rb1;
                            let r2 = ra2 + ib2;
                            let i2 = ia2 - rb2;
                            let r3 = ra2 - ib2;
                            let i3 = ia2 + rb2;
                            let r4 = ra1 - ib1;
                            let i4 = ia1 + rb1;
                            data[x1 + k as usize] = r1 * c1 + i1 * s1;
                            data[y1 + k as usize] = i1 * c1 - r1 * s1;
                            data[x2 + k as usize] = r2 * c2 + i2 * s2;
                            data[y2 + k as usize] = i2 * c2 - r2 * s2;
                            data[x3 + k as usize] = r3 * c3 + i3 * s3;
                            data[y3 + k as usize] = i3 * c3 - r3 * s3;
                            data[x4 + k as usize] = r4 * c4 + i4 * s4;
                            data[y4 + k as usize] = i4 * c4 - r4 * s4;
                        } else {
                            data[x1 + k as usize] = ra1 + ib1;
                            data[y1 + k as usize] = ia1 - rb1;
                            data[x2 + k as usize] = ra2 + ib2;
                            data[y2 + k as usize] = ia2 - rb2;
                            data[x3 + k as usize] = ra2 - ib2;
                            data[y3 + k as usize] = ia2 + rb2;
                            data[x4 + k as usize] = ra1 - ib1;
                            data[y4 + k as usize] = ia1 + rb1;
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_red5;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
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
        }
    }
}

/// C `r8cftk`.
///
/// radix 8 multi-dimensional complex fourier transform kernel
pub fn r8cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    x4: usize,
    y4: usize,
    x5: usize,
    y5: usize,
    x6: usize,
    y6: usize,
    x7: usize,
    y7: usize,
    dim: &[i32],
) {
    let n_red8: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red8: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut c4: f32 = 0.0;
    let mut c5: f32 = 0.0;
    let mut c6: f32 = 0.0;
    let mut c7: f32 = 0.0;
    let e: f32 = 0.70710678;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut s4: f32 = 0.0;
    let mut s5: f32 = 0.0;
    let mut s6: f32 = 0.0;
    let mut s7: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red8: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red8 = n_reduced * 8;
    fn_red8 = n_red8 as f32;
    sep_n_red8 = sep * n_red8;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red8) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
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
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        let rs0 = data[x0 + k as usize] + data[x4 + k as usize];
                        let is0 = data[y0 + k as usize] + data[y4 + k as usize];
                        let ru0 = data[x0 + k as usize] - data[x4 + k as usize];
                        let iu0 = data[y0 + k as usize] - data[y4 + k as usize];
                        let rs1 = data[x1 + k as usize] + data[x5 + k as usize];
                        let is1 = data[y1 + k as usize] + data[y5 + k as usize];
                        let ru1 = data[x1 + k as usize] - data[x5 + k as usize];
                        let iu1 = data[y1 + k as usize] - data[y5 + k as usize];
                        let rs2 = data[x2 + k as usize] + data[x6 + k as usize];
                        let is2 = data[y2 + k as usize] + data[y6 + k as usize];
                        let ru2 = data[x2 + k as usize] - data[x6 + k as usize];
                        let iu2 = data[y2 + k as usize] - data[y6 + k as usize];
                        let rs3 = data[x3 + k as usize] + data[x7 + k as usize];
                        let is3 = data[y3 + k as usize] + data[y7 + k as usize];
                        let ru3 = data[x3 + k as usize] - data[x7 + k as usize];
                        let iu3 = data[y3 + k as usize] - data[y7 + k as usize];
                        let rss0 = rs0 + rs2;
                        let iss0 = is0 + is2;
                        let rsu0 = rs0 - rs2;
                        let isu0 = is0 - is2;
                        let rss1 = rs1 + rs3;
                        let iss1 = is1 + is3;
                        let rsu1 = rs1 - rs3;
                        let isu1 = is1 - is3;
                        let rus0 = ru0 - iu2;
                        let ius0 = iu0 + ru2;
                        let ruu0 = ru0 + iu2;
                        let iuu0 = iu0 - ru2;
                        let mut rus1 = ru1 - iu3;
                        let mut ius1 = iu1 + ru3;
                        let mut ruu1 = ru1 + iu3;
                        let mut iuu1 = iu1 - ru3;
                        t = (rus1 + ius1) * e;
                        ius1 = (ius1 - rus1) * e;
                        rus1 = t;
                        t = (ruu1 + iuu1) * e;
                        iuu1 = (iuu1 - ruu1) * e;
                        ruu1 = t;
                        data[x0 + k as usize] = rss0 + rss1;
                        data[y0 + k as usize] = iss0 + iss1;
                        if !zero {
                            let r1 = ruu0 + ruu1;
                            let i1 = iuu0 + iuu1;
                            let r2 = rsu0 + isu1;
                            let i2 = isu0 - rsu1;
                            let r3 = rus0 + ius1;
                            let i3 = ius0 - rus1;
                            let r4 = rss0 - rss1;
                            let i4 = iss0 - iss1;
                            let r5 = ruu0 - ruu1;
                            let i5 = iuu0 - iuu1;
                            let r6 = rsu0 - isu1;
                            let i6 = isu0 + rsu1;
                            let r7 = rus0 - ius1;
                            let i7 = ius0 + rus1;
                            data[x4 + k as usize] = r1 * c1 + i1 * s1;
                            data[y4 + k as usize] = i1 * c1 - r1 * s1;
                            data[x2 + k as usize] = r2 * c2 + i2 * s2;
                            data[y2 + k as usize] = i2 * c2 - r2 * s2;
                            data[x6 + k as usize] = r3 * c3 + i3 * s3;
                            data[y6 + k as usize] = i3 * c3 - r3 * s3;
                            data[x1 + k as usize] = r4 * c4 + i4 * s4;
                            data[y1 + k as usize] = i4 * c4 - r4 * s4;
                            data[x5 + k as usize] = r5 * c5 + i5 * s5;
                            data[y5 + k as usize] = i5 * c5 - r5 * s5;
                            data[x3 + k as usize] = r6 * c6 + i6 * s6;
                            data[y3 + k as usize] = i6 * c6 - r6 * s6;
                            data[x7 + k as usize] = r7 * c7 + i7 * s7;
                            data[y7 + k as usize] = i7 * c7 - r7 * s7;
                        } else {
                            data[x4 + k as usize] = ruu0 + ruu1;
                            data[y4 + k as usize] = iuu0 + iuu1;
                            data[x2 + k as usize] = rsu0 + isu1;
                            data[y2 + k as usize] = isu0 - rsu1;
                            data[x6 + k as usize] = rus0 + ius1;
                            data[y6 + k as usize] = ius0 - rus1;
                            data[x1 + k as usize] = rss0 - rss1;
                            data[y1 + k as usize] = iss0 - iss1;
                            data[x5 + k as usize] = ruu0 - ruu1;
                            data[y5 + k as usize] = iuu0 - iuu1;
                            data[x3 + k as usize] = rsu0 - isu1;
                            data[y3 + k as usize] = isu0 + rsu1;
                            data[x7 + k as usize] = rus0 - ius1;
                            data[y7 + k as usize] = ius0 + rus1;
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_red8;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
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
        }
    }
}

/// C `rpcftk`.
///
/// radix prime multi-dimensional complex fourier transform kernel
pub fn rpcftk(
    n: i32,
    n_reduced: i32,
    p_fac: i32,
    step_along: i32,
    data: &mut [f32],
    x: usize,
    y: usize,
    dim: &[i32],
) {
    let twopi: f32 = 6.2831853;
    let mut t: f32;
    let mut xt: f32;
    let mut yt: f32;
    let mut fu: f32;
    let fp: f32;
    let mut fjm1: f32;
    let fn_redp: f32;
    let mut jj: i32;
    let n_red_ov2p1: i32;
    let n_redp: i32;
    let pm: i32;
    let pp: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_redp: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;

    // `rpcftk` declares these on the stack without initialising them; every
    // element read below is written first (`pp <= 9` and `pm <= 18` for the
    // largest factor `srfp` allows).
    let mut aa = [[0.0_f32; 10]; 10];
    let mut bb = [[0.0_f32; 10]; 10];
    let mut a = [0.0_f32; 19];
    let mut b = [0.0_f32; 19];
    let mut c = [0.0_f32; 19];
    let mut sep_along = [0.0_f32; 19];
    let mut ia = [0.0_f32; 10];
    let mut ib = [0.0_f32; 10];
    let mut ra = [0.0_f32; 10];
    let mut rb = [0.0_f32; 10];

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red_ov2p1 = n_reduced / 2 + 1;
    n_redp = n_reduced * p_fac;
    fn_redp = n_redp as f32;
    sep_n_redp = sep * n_redp;
    pp = p_fac / 2;
    pm = p_fac - 1;
    fp = p_fac as f32;
    fu = 0.0;
    for u in 1..=pp {
        fu = (fu as f64 + 1.0) as f32;
        let angle = (twopi * fu / fp) as f64;
        jj = p_fac - u;
        a[u as usize] = angle.cos() as f32;
        b[u as usize] = angle.sin() as f32;
        a[jj as usize] = a[u as usize];
        b[jj as usize] = -b[u as usize];
    }
    for u in 1..=pp {
        for v in 1..=pp {
            jj = u * v - u * v / p_fac * p_fac;
            aa[v as usize][u as usize] = a[jj as usize];
            bb[v as usize][u as usize] = b[jj as usize];
        }
    }

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_redp) as f64;
        let zero = angle == 0.0;
        if !zero {
            c[1] = angle.cos() as f32;
            sep_along[1] = angle.sin() as f32;
            for u in 2..=pm {
                c[u as usize] =
                    c[(u - 1) as usize] * c[1] - sep_along[(u - 1) as usize] * sep_along[1];
                sep_along[u as usize] =
                    sep_along[(u - 1) as usize] * c[1] + c[(u - 1) as usize] * sep_along[1];
            }
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    while k < k1 {
                        xt = data[x + k as usize];
                        yt = data[y + k as usize];
                        let mut rs = data[x + (k + step_along) as usize]
                            + data[x + (k + step_along * pm) as usize];
                        let mut is = data[y + (k + step_along) as usize]
                            + data[y + (k + step_along * pm) as usize];
                        let mut ru = data[x + (k + step_along) as usize]
                            - data[x + (k + step_along * pm) as usize];
                        let mut iu = data[y + (k + step_along) as usize]
                            - data[y + (k + step_along * pm) as usize];
                        for u in 1..=pp {
                            ra[u as usize] = xt + rs * aa[u as usize][1];
                            ia[u as usize] = yt + is * aa[u as usize][1];
                            rb[u as usize] = ru * bb[u as usize][1];
                            ib[u as usize] = iu * bb[u as usize][1];
                        }
                        xt = xt + rs;
                        yt = yt + is;

                        for u in 2..=pp {
                            // u numbers from 1 not 0
                            jj = p_fac - u;
                            rs = data[x + (k + u * step_along) as usize]
                                + data[x + (k + jj * step_along) as usize];
                            is = data[y + (k + u * step_along) as usize]
                                + data[y + (k + jj * step_along) as usize];
                            ru = data[x + (k + u * step_along) as usize]
                                - data[x + (k + jj * step_along) as usize];
                            iu = data[y + (k + u * step_along) as usize]
                                - data[y + (k + jj * step_along) as usize];
                            xt = xt + rs;
                            yt = yt + is;
                            for v in 1..=pp {
                                ra[v as usize] = ra[v as usize] + rs * aa[v as usize][u as usize];
                                ia[v as usize] = ia[v as usize] + is * aa[v as usize][u as usize];
                                rb[v as usize] = rb[v as usize] + ru * bb[v as usize][u as usize];
                                ib[v as usize] = ib[v as usize] + iu * bb[v as usize][u as usize];
                            }
                        }
                        data[x + k as usize] = xt;
                        data[y + k as usize] = yt;
                        for u in 1..=pp {
                            jj = p_fac - u;
                            if !zero {
                                xt = ra[u as usize] + ib[u as usize];
                                yt = ia[u as usize] - rb[u as usize];
                                data[x + (k + u * step_along) as usize] =
                                    xt * c[u as usize] + yt * sep_along[u as usize];
                                data[y + (k + u * step_along) as usize] =
                                    yt * c[u as usize] - xt * sep_along[u as usize];
                                xt = ra[u as usize] - ib[u as usize];
                                yt = ia[u as usize] + rb[u as usize];
                                data[x + (k + jj * step_along) as usize] =
                                    xt * c[jj as usize] + yt * sep_along[jj as usize];
                                data[y + (k + jj * step_along) as usize] =
                                    yt * c[jj as usize] - xt * sep_along[jj as usize];
                            } else {
                                data[x + (k + u * step_along) as usize] =
                                    ra[u as usize] + ib[u as usize];
                                data[y + (k + u * step_along) as usize] =
                                    ia[u as usize] - rb[u as usize];
                                data[x + (k + jj * step_along) as usize] =
                                    ra[u as usize] - ib[u as usize];
                                data[y + (k + jj * step_along) as usize] =
                                    ia[u as usize] + rb[u as usize];
                            }
                        }
                        k += sep_between;
                    }
                    l += lim_along;
                }
                kk += sep_n_redp;
            }
            if !fold {
                break;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            for u in 1..=pm {
                t = c[u as usize] * a[u as usize] + sep_along[u as usize] * b[u as usize];
                sep_along[u as usize] =
                    -sep_along[u as usize] * a[u as usize] + c[u as usize] * b[u as usize];
                c[u as usize] = t;
            }
        }
    }
}
