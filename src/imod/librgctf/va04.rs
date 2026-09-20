//! Safe translation of `IMOD/librgctf/va04.cpp`.
//!
//! VA04 is the PFORT direction-set minimizer used by CTFFIND.  The original
//! is a mechanically translated Fortran routine with one-based arrays and
//! labelled jumps.  This version retains its control flow and arithmetic but
//! exposes ordinary slices and a Rust objective callback.

/// Original `va04a_`.
///
/// `e` and `x` must each contain `n` values.  The callback is invoked with
/// the current candidate values and this routine writes the best result back
/// to `x` and `f`.  Returns zero, as the source does on every exit path.
pub fn va04a<F>(
    n: usize,
    e: &[f32],
    escale: f32,
    _num_function_calls: &mut i32,
    mut target_function: F,
    f: &mut f32,
    iprint: i32,
    icon: i32,
    maxit: i32,
    x: &mut [f32],
) -> i32
where
    F: FnMut(&[f32]) -> f32,
{
    if n == 0 || e.len() < n || x.len() < n {
        return 0;
    }
    let maxx = 100_i32.saturating_mul(maxit);
    let mut icnt = 0_i32;
    let mut xs = x[..n].to_vec();
    let mut w = vec![0.0_f64; n * (n + 3)];
    let ddmag_initial = escale as f64 * 0.1;
    let scer = 0.05_f64 / escale as f64;
    let mut ddmag = ddmag_initial;
    let jj = n * n + n;
    let mut jjj = jj + n;
    let mut k = n;
    let mut nfcc = 1_i32;
    let mut ind = 1_i32;
    let mut inn = 1_i32;
    for i in 0..n {
        for j in 0..n {
            w[k] = 0.0;
            if i == j {
                w[k] = e[i].abs() as f64;
                w[i] = escale as f64;
            }
            k += 1;
        }
    }
    let mut iterc = 1_i32;
    let mut isgrad = 2_i32;
    icnt += 1;
    if icnt > maxx {
        *f = target_function(&xs);
        x[..n].copy_from_slice(&xs);
        return 0;
    }
    *f = target_function(&x[..n]);
    let mut fkeep = (*f as f64).abs() * 2.0;
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut d = 0.0_f64;
    let mut da = 0.0_f64;
    let mut db = 0.0_f64;
    let mut dc = 0.0_f64;
    let mut dd = 0.0_f64;
    let mut di = 0.0_f64;
    let mut dl = 0.0_f64;
    let mut fa = 0.0_f64;
    let mut fb = 0.0_f64;
    let mut fc = 0.0_f64;
    let mut fi = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut fhold = 0.0_f64;
    let mut fprev = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut dacc = 0.0_f64;
    let mut dmag = 0.0_f64;
    let mut dmax = 0.0_f64;
    let mut ddmax = 0.0_f64;
    let mut aaa = 0.0_f64;
    let mut idirn = 0_usize;
    let mut iline = 0_usize;
    let mut is = 0_i32;
    let mut itone = 0_i32;
    let mut jil = 0_usize;
    let mut ixp = 0_usize;
    let mut state = 5_i32;

    loop {
        match state {
            5 => {
                itone = 1;
                fp = *f as f64;
                sum = 0.0;
                ixp = jj;
                for value in &x[..n] {
                    w[ixp] = *value as f64;
                    ixp += 1;
                }
                idirn = n;
                iline = 0;
                state = 7;
            }
            7 => {
                dmax = w[iline];
                dacc = dmax * scer;
                dmag = ddmag.min(dmax * 0.1).max(dacc * 20.0);
                ddmax = dmag * 10.0;
                state = if itone == 3 { 71 } else { 70 };
            }
            70 => {
                dl = 0.0;
                d = dmag;
                fprev = *f as f64;
                is = 5;
                fa = *f as f64;
                da = dl;
                state = 8;
            }
            8 => {
                dd = d - dl;
                dl = d;
                state = 58;
            }
            58 => {
                k = idirn;
                for value in &mut x[..n] {
                    *value += (dd * w[k]) as f32;
                    k += 1;
                }
                icnt += 1;
                if icnt > maxx {
                    x[..n].copy_from_slice(&xs);
                    *f = target_function(&x[..n]);
                    return 0;
                }
                *f = target_function(&x[..n]);
                nfcc += 1;
                state = match is {
                    1 => 10,
                    2 => 11,
                    3 => 12,
                    4 => 13,
                    5 => 14,
                    _ => 96,
                };
            }
            14 => {
                if (*f as f64) < fa {
                    state = 15;
                } else if (*f as f64) == fa {
                    state = 16;
                } else {
                    state = 24;
                }
            }
            16 => {
                if d.abs() <= dmax {
                    d += d;
                    state = 8;
                } else {
                    state = 20;
                }
            }
            15 => {
                fb = *f as f64;
                db = d;
                state = 21;
            }
            24 => {
                fb = fa;
                db = da;
                fa = *f as f64;
                da = d;
                state = 21;
            }
            21 => {
                state = if isgrad == 1 { 83 } else { 23 };
            }
            23 => {
                d = db + db - da;
                is = 1;
                state = 8;
            }
            83 => {
                d = (da + db - (fa - fb) / (da - db)) * 0.5;
                is = 4;
                state = if (da - d) * (d - db) >= 0.0 { 8 } else { 25 };
            }
            25 => {
                state = if (d - db).abs() <= ddmax { 8 } else { 26 };
            }
            26 => {
                d = db + ddmax.copysign(db - da);
                is = 1;
                ddmax += ddmax;
                ddmag += ddmag;
                state = if ddmax <= dmax { 8 } else { 27 };
            }
            27 => {
                ddmax = dmax;
                state = 8;
            }
            13 => {
                state = if (*f as f64) >= fa { 23 } else { 28 };
            }
            28 => {
                fc = fb;
                dc = db;
                state = 29;
            }
            29 => {
                fb = *f as f64;
                db = d;
                state = 30;
            }
            12 => {
                state = if (*f as f64) <= fb { 28 } else { 31 };
            }
            31 => {
                fa = *f as f64;
                da = d;
                state = 30;
            }
            11 => {
                state = if (*f as f64) >= fb { 10 } else { 32 };
            }
            32 => {
                fa = fb;
                da = db;
                state = 29;
            }
            71 => {
                dl = 1.0;
                ddmax = 5.0;
                fa = fp;
                da = -1.0;
                fb = fhold;
                db = 0.0;
                d = 1.0;
                state = 10;
            }
            10 => {
                fc = *f as f64;
                dc = d;
                state = 30;
            }
            30 => {
                a = (db - dc) * (fa - fc);
                b = (dc - da) * (fb - fc);
                state = if (a + b) * (da - dc) <= 0.0 { 33 } else { 34 };
            }
            33 => {
                fa = fb;
                da = db;
                fb = fc;
                db = dc;
                state = 26;
            }
            34 => {
                d = (a * (db + dc) + b * (da + dc)) * 0.5 / (a + b);
                di = db;
                fi = fb;
                if fb > fc {
                    di = dc;
                    fi = fc;
                }
                state = if itone == 3 { 85 } else { 86 };
            }
            85 => {
                itone = 2;
                state = 45;
            }
            86 => {
                state = if (d - di).abs() <= dacc {
                    41
                } else if (d - di).abs() <= d.abs() * 0.03 {
                    41
                } else {
                    45
                };
            }
            45 => {
                state = if (da - dc) * (dc - d) >= 0.0 { 46 } else { 47 };
            }
            46 => {
                fa = fb;
                da = db;
                fb = fc;
                db = dc;
                state = 25;
            }
            47 => {
                is = if (db - d) * (d - dc) >= 0.0 { 2 } else { 3 };
                state = 8;
            }
            41 => {
                *f = fi as f32;
                d = di - dl;
                dd = ((dc - db) * (dc - da) * (da - db) / (a + b)).sqrt();
                for value in &mut x[..n] {
                    *value += (d * w[idirn]) as f32;
                    w[idirn] *= dd;
                    idirn += 1;
                }
                if dd == 0.0 {
                    dd = 1.0e-10;
                }
                w[iline] /= dd;
                iline += 1;
                state = if iprint == 1 { 51 } else { 50 };
            }
            50 => {
                state = if iprint == 1 {
                    51
                } else if iprint == 2 {
                    53
                } else {
                    51
                };
            }
            51 => {
                if itone == 1 {
                    state = 55;
                } else {
                    state = 38;
                }
            }
            55 => {
                if fprev - *f as f64 - sum >= 0.0 {
                    sum = fprev - *f as f64;
                    jil = iline;
                }
                state = if idirn < jj { 7 } else { 84 };
            }
            84 => {
                state = if ind == 1 { 92 } else { 72 };
            }
            92 => {
                fhold = *f as f64;
                is = 6;
                ixp = jj;
                for value in &x[..n] {
                    w[ixp] = *value as f64 - w[ixp];
                    ixp += 1;
                }
                dd = 1.0;
                state = 58;
            }
            96 => {
                state = if ind == 1 { 112 } else { 87 };
            }
            112 => {
                state = if fp - (*f as f64) <= 0.0 { 37 } else { 91 };
            }
            91 => {
                d = (fp + *f as f64 - fhold * 2.0) * 2.0 / (fp - *f as f64).powi(2);
                state = if d * (fp - fhold - sum).powi(2) - sum >= 0.0 {
                    37
                } else {
                    87
                };
            }
            87 => {
                let j = (jil + 1) * n;
                if j < jj {
                    for i in j..jj {
                        w[i - n] = w[i];
                    }
                    for i in jil..n {
                        w[i - 1] = w[i];
                    }
                }
                idirn -= n;
                itone = 3;
                k = idirn;
                ixp = jj;
                aaa = 0.0;
                for i in 0..n {
                    w[k] = w[ixp];
                    aaa = aaa.max((w[k] / e[i] as f64).abs());
                    k += 1;
                    ixp += 1;
                }
                ddmag = 1.0;
                if aaa == 0.0 {
                    aaa = 1.0e-10;
                }
                w[n - 1] = escale as f64 / aaa;
                iline = n - 1;
                state = 7;
            }
            37 => {
                ixp = jj;
                aaa = 0.0;
                *f = fhold as f32;
                for i in 0..n {
                    x[i] -= w[ixp] as f32;
                    aaa = aaa.max((w[ixp] / e[i] as f64).abs());
                    ixp += 1;
                }
                state = 72;
            }
            38 => {
                aaa *= di + 1.0;
                state = if ind == 1 { 72 } else { 106 };
            }
            72 => {
                state = if iprint >= 2 { 50 } else { 53 };
            }
            53 => {
                state = if ind == 1 { 109 } else { 88 };
            }
            109 => {
                state = if aaa <= 0.1 { 89 } else { 76 };
            }
            89 => {
                state = if icon == 1 { 20 } else { 116 };
            }
            116 => {
                ind = 2;
                state = if inn == 1 { 100 } else { 101 };
            }
            100 => {
                inn = 2;
                k = jjj;
                for i in 0..n {
                    w[k] = x[i] as f64;
                    x[i] += e[i] * 10.0;
                    k += 1;
                }
                fkeep = *f as f64;
                icnt += 1;
                if icnt > maxx {
                    x[..n].copy_from_slice(&xs);
                    *f = target_function(&x[..n]);
                    return 0;
                }
                *f = target_function(&x[..n]);
                nfcc += 1;
                ddmag = 0.0;
                state = 108;
            }
            76 => {
                state = if (*f as f64) >= fp { 78 } else { 35 };
            }
            78 => {
                state = 20;
            }
            88 => {
                ind = 1;
                state = 35;
            }
            35 => {
                let tmp = fp - *f as f64;
                ddmag = if tmp > 0.0 { tmp.sqrt() * 0.4 } else { 0.0 };
                isgrad = 1;
                state = 108;
            }
            108 => {
                iterc += 1;
                state = if iterc <= maxit { 5 } else { 81 };
            }
            81 => {
                state = if (*f as f64) <= fkeep { 20 } else { 110 };
            }
            110 => {
                *f = fkeep as f32;
                for value in &mut x[..n] {
                    *value = w[jjj] as f32;
                    jjj += 1;
                }
                state = 20;
            }
            101 => {
                jil = 1;
                fp = fkeep;
                if (*f as f64) < fkeep {
                    state = 105;
                } else if (*f as f64) == fkeep {
                    state = 78;
                } else {
                    state = 104;
                }
            }
            104 => {
                jil = 2;
                fp = *f as f64;
                *f = fkeep as f32;
                state = 105;
            }
            105 => {
                ixp = jj;
                for i in 0..n {
                    k = ixp + n;
                    if jil == 1 {
                        w[ixp] = w[k];
                    } else {
                        w[ixp] = x[i] as f64;
                        x[i] = w[k] as f32;
                    }
                    ixp += 1;
                }
                jil = 2;
                state = 92;
            }
            106 => {
                state = if aaa <= 0.1 { 20 } else { 107 };
            }
            107 => {
                inn = 1;
                state = 35;
            }
            _ => return 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn va04a_minimizes_a_two_parameter_quadratic() {
        let mut x = [7.0, -5.0];
        let mut f = 0.0;
        let mut calls = 0;
        assert_eq!(
            va04a(
                2,
                &[0.25, 0.25],
                1.0,
                &mut calls,
                |v| (v[0] - 2.0).powi(2) + (v[1] + 3.0).powi(2),
                &mut f,
                0,
                1,
                1,
                &mut x
            ),
            0
        );
        assert!(f < 1.0e-4, "f={f}, x={x:?}");
        assert!(
            (x[0] - 2.0).abs() < 0.02 && (x[1] + 3.0).abs() < 0.02,
            "x={x:?}"
        );
    }
}
