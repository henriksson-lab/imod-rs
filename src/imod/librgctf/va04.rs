//! Translation of `IMOD/librgctf/va04.cpp`.
//!
//! VA04 is the PFORT direction-set minimizer CTFFIND uses for its conjugate
//! gradient refinement.  The vendored file is f2c output from Fortran 66, so
//! the arrays are one-based (`--e; --x;` shifts the parameters) and the
//! control flow is a graph of `goto` labels; both are kept here, with the label
//! numbers as the `state` values and `w[(k - 1) as usize]` where the C writes
//! `w[k - 1]`.
//!
//! Two properties of the C that are easy to lose:
//!
//! * every local except `icnt`, `maxx` and the `r__` temporaries is declared
//!   **`static`**, so its value survives from one call to the next.  `jil`,
//!   `aaa` and `di` are read on paths that do not necessarily write them first,
//!   so the state lives in a thread-local and is copied in and out.
//! * the constants are written `(float).1`, `(float).03`, `(float).4`,
//!   `(float)1e-10` and `(float).05` in double expressions, so the value used
//!   is the *float* literal widened — `0.1f` is 0.100000001490116…, not 0.1.

use std::cell::Cell;

/// `(float).1` widened to double.
const F_0_1: f64 = 0.1f32 as f64;
/// `(float).03` widened to double.
const F_0_03: f64 = 0.03f32 as f64;
/// `(float).4` widened to double.
const F_0_4: f64 = 0.4f32 as f64;
/// `(float)1e-10` widened to double.
const F_1E_10: f64 = 1e-10f32 as f64;

/// The `static` locals of `va04a_` (`va04.cpp:80-96`).
#[derive(Clone, Copy)]
struct Va04Statics {
    a: f64,
    b: f64,
    d: f64,
    xs: [f64; 12],
    k: i32,
    w: [f64; 155],
    da: f64,
    db: f64,
    fa: f64,
    dd: f64,
    fb: f64,
    fc: f64,
    dc: f64,
    di: f64,
    fi: f64,
    dl: f64,
    jj: i32,
    fp: f64,
    is: i32,
    aaa: f64,
    ind: i32,
    jjj: i32,
    jil: i32,
    inn: i32,
    ixp: i32,
    sum: f64,
    dacc: f64,
    dmag: f64,
    nfcc: i32,
    dmax: f64,
    scer: f64,
    ddmag: f64,
    fkeep: f64,
    fhold: f64,
    ddmax: f64,
    iline: i32,
    idirn: i32,
    iterc: i32,
    itone: i32,
    fprev: f64,
    isgrad: i32,
}

impl Va04Statics {
    const fn zeroed() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            d: 0.0,
            xs: [0.0; 12],
            k: 0,
            w: [0.0; 155],
            da: 0.0,
            db: 0.0,
            fa: 0.0,
            dd: 0.0,
            fb: 0.0,
            fc: 0.0,
            dc: 0.0,
            di: 0.0,
            fi: 0.0,
            dl: 0.0,
            jj: 0,
            fp: 0.0,
            is: 0,
            aaa: 0.0,
            ind: 0,
            jjj: 0,
            jil: 0,
            inn: 0,
            ixp: 0,
            sum: 0.0,
            dacc: 0.0,
            dmag: 0.0,
            nfcc: 0,
            dmax: 0.0,
            scer: 0.0,
            ddmag: 0.0,
            fkeep: 0.0,
            fhold: 0.0,
            ddmax: 0.0,
            iline: 0,
            idirn: 0,
            iterc: 0,
            itone: 0,
            fprev: 0.0,
            isgrad: 0,
        }
    }
}

thread_local! {
    static VA04_STATICS: Cell<Va04Statics> = const { Cell::new(Va04Statics::zeroed()) };
}

/// C `va04a_` (`va04.cpp:61`).
///
/// The C++ pair of a `float (*)(void *, float [])` and a `void *parameters` is
/// one Rust closure.  `num_function_calls` is in the signature and the source
/// never writes through it.  Returns 0, as every exit path of the C does.
#[allow(clippy::too_many_arguments)]
pub fn va04a<F>(
    n: i32,
    e: &[f32],
    escale: f32,
    num_function_calls: &mut i32,
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
    let _ = num_function_calls;
    let maxx: i32 = 100 * maxit;
    let mut icnt: i32 = 0;
    let mut s = VA04_STATICS.with(|cell| cell.get());

    let n_usize = n as usize;

    // e and x are one-based below, matching the C's `--e; --x;`.
    macro_rules! e1 {
        ($i:expr) => {
            e[($i - 1) as usize]
        };
    }
    macro_rules! x1 {
        ($i:expr) => {
            x[($i - 1) as usize]
        };
    }
    macro_rules! w0 {
        ($i:expr) => {
            s.w[($i - 1) as usize]
        };
    }

    s.ddmag = f64::from(escale * 0.1f32);
    s.scer = f64::from(0.05f32 / escale);
    s.jj = n * n + n;
    s.jjj = s.jj + n;
    s.k = n + 1;
    s.nfcc = 1;
    s.ind = 1;
    s.inn = 1;
    for i in 1..=n {
        s.xs[i as usize] = f64::from(x1!(i));
    }
    for i in 1..=n {
        for j in 1..=n {
            w0!(s.k) = 0.;
            if i - j == 0 {
                w0!(s.k) = f64::from(e1!(i)).abs();
                w0!(i) = f64::from(escale);
            }
            s.k += 1;
        }
    }
    s.iterc = 1;
    s.isgrad = 2;
    icnt += 1;
    if icnt > maxx {
        return finish_999(&mut s, n, x, f, &mut target_function);
    }
    *f = target_function(&x[..n_usize]);
    s.fkeep = f64::from(f.abs()) + f64::from(f.abs());

    let mut state = 5;
    loop {
        match state {
            5 => {
                s.itone = 1;
                s.fp = f64::from(*f);
                s.sum = 0.;
                s.ixp = s.jj;
                for i in 1..=n {
                    s.ixp += 1;
                    w0!(s.ixp) = f64::from(x1!(i));
                }
                s.idirn = n + 1;
                s.iline = 1;
                state = 7;
            }
            7 => {
                s.dmax = w0!(s.iline);
                s.dacc = s.dmax * s.scer;
                s.dmag = s.ddmag.min(s.dmax * F_0_1);
                s.dmag = s.dmag.max(s.dacc * 20.);
                s.ddmax = s.dmag * 10.;
                state = match s.itone {
                    1 | 2 => 70,
                    _ => 71,
                };
            }
            70 => {
                s.dl = 0.;
                s.d = s.dmag;
                s.fprev = f64::from(*f);
                s.is = 5;
                s.fa = f64::from(*f);
                s.da = s.dl;
                state = 8;
            }
            8 => {
                s.dd = s.d - s.dl;
                s.dl = s.d;
                state = 58;
            }
            58 => {
                s.k = s.idirn;
                for i in 1..=n {
                    x1!(i) = (f64::from(x1!(i)) + s.dd * w0!(s.k)) as f32;
                    s.k += 1;
                }
                icnt += 1;
                if icnt > maxx {
                    return finish_999(&mut s, n, x, f, &mut target_function);
                }
                *f = target_function(&x[..n_usize]);

                s.nfcc += 1;
                state = match s.is {
                    1 => 10,
                    2 => 11,
                    3 => 12,
                    4 => 13,
                    5 => 14,
                    _ => 96,
                };
            }
            14 => {
                let r = f64::from(*f) - s.fa;
                state = if r < 0. {
                    15
                } else if r == 0. {
                    16
                } else {
                    24
                };
            }
            16 => {
                state = if s.d.abs() - s.dmax <= 0. { 17 } else { 18 };
            }
            17 => {
                s.d += s.d;
                state = 8;
            }
            18 => {
                // The source's "maximum change does not alter function"
                // message is commented out.
                state = 20;
            }
            15 => {
                s.fb = f64::from(*f);
                s.db = s.d;
                state = 21;
            }
            24 => {
                s.fb = s.fa;
                s.db = s.da;
                s.fa = f64::from(*f);
                s.da = s.d;
                state = 21;
            }
            21 => {
                state = match s.isgrad {
                    1 => 83,
                    _ => 23,
                };
            }
            23 => {
                s.d = s.db + s.db - s.da;
                s.is = 1;
                state = 8;
            }
            83 => {
                s.d = (s.da + s.db - (s.fa - s.fb) / (s.da - s.db)) * 0.5;
                s.is = 4;
                state = if (s.da - s.d) * (s.d - s.db) >= 0. {
                    8
                } else {
                    25
                };
            }
            25 => {
                s.is = 1;
                state = if (s.d - s.db).abs() - s.ddmax <= 0. {
                    8
                } else {
                    26
                };
            }
            26 => {
                s.d = s.db + s.ddmax.copysign(s.db - s.da);
                s.is = 1;
                s.ddmax += s.ddmax;
                s.ddmag += s.ddmag;
                state = if s.ddmax - s.dmax <= 0. { 8 } else { 27 };
            }
            27 => {
                s.ddmax = s.dmax;
                state = 8;
            }
            13 => {
                state = if f64::from(*f) - s.fa >= 0. { 23 } else { 28 };
            }
            28 => {
                s.fc = s.fb;
                s.dc = s.db;
                state = 29;
            }
            29 => {
                s.fb = f64::from(*f);
                s.db = s.d;
                state = 30;
            }
            12 => {
                state = if f64::from(*f) - s.fb <= 0. { 28 } else { 31 };
            }
            31 => {
                s.fa = f64::from(*f);
                s.da = s.d;
                state = 30;
            }
            11 => {
                state = if f64::from(*f) - s.fb >= 0. { 10 } else { 32 };
            }
            32 => {
                s.fa = s.fb;
                s.da = s.db;
                state = 29;
            }
            71 => {
                s.dl = 1.;
                s.ddmax = 5.;
                s.fa = s.fp;
                s.da = -1.;
                s.fb = s.fhold;
                s.db = 0.;
                s.d = 1.;
                state = 10;
            }
            10 => {
                s.fc = f64::from(*f);
                s.dc = s.d;
                state = 30;
            }
            30 => {
                s.a = (s.db - s.dc) * (s.fa - s.fc);
                s.b = (s.dc - s.da) * (s.fb - s.fc);
                state = if (s.a + s.b) * (s.da - s.dc) <= 0. {
                    33
                } else {
                    34
                };
            }
            33 => {
                s.fa = s.fb;
                s.da = s.db;
                s.fb = s.fc;
                s.db = s.dc;
                state = 26;
            }
            34 => {
                s.d = (s.a * (s.db + s.dc) + s.b * (s.da + s.dc)) * 0.5 / (s.a + s.b);
                s.di = s.db;
                s.fi = s.fb;
                if s.fb - s.fc <= 0. {
                    state = 44;
                } else {
                    state = 43;
                }
            }
            43 => {
                s.di = s.dc;
                s.fi = s.fc;
                state = 44;
            }
            44 => {
                state = match s.itone {
                    1 | 2 => 86,
                    _ => 85,
                };
            }
            85 => {
                s.itone = 2;
                state = 45;
            }
            86 => {
                state = if (s.d - s.di).abs() - s.dacc <= 0. {
                    41
                } else {
                    93
                };
            }
            93 => {
                state = if (s.d - s.di).abs() - s.d.abs() * F_0_03 <= 0. {
                    41
                } else {
                    45
                };
            }
            45 => {
                state = if (s.da - s.dc) * (s.dc - s.d) >= 0. {
                    46
                } else {
                    47
                };
            }
            46 => {
                s.fa = s.fb;
                s.da = s.db;
                s.fb = s.fc;
                s.db = s.dc;
                state = 25;
            }
            47 => {
                s.is = 2;
                if (s.db - s.d) * (s.d - s.dc) >= 0. {
                    state = 8;
                } else {
                    state = 48;
                }
            }
            48 => {
                s.is = 3;
                state = 8;
            }
            41 => {
                *f = s.fi as f32;
                s.d = s.di - s.dl;
                s.dd = ((s.dc - s.db) * (s.dc - s.da) * (s.da - s.db) / (s.a + s.b)).sqrt();
                for i in 1..=n {
                    x1!(i) = (f64::from(x1!(i)) + s.d * w0!(s.idirn)) as f32;
                    w0!(s.idirn) = s.dd * w0!(s.idirn);
                    s.idirn += 1;
                }
                if s.dd == 0. {
                    s.dd = F_1E_10;
                }
                w0!(s.iline) /= s.dd;
                s.iline += 1;
                state = if iprint - 1 != 0 { 51 } else { 50 };
            }
            50 => {
                state = match iprint {
                    1 => 51,
                    2 => 53,
                    // A `switch` with no matching case falls through to L51.
                    _ => 51,
                };
            }
            51 => {
                state = match s.itone {
                    1 => 55,
                    _ => 38,
                };
            }
            55 => {
                state = if s.fprev - f64::from(*f) - s.sum >= 0. {
                    95
                } else {
                    94
                };
            }
            95 => {
                s.sum = s.fprev - f64::from(*f);
                s.jil = s.iline;
                state = 94;
            }
            94 => {
                state = if s.idirn - s.jj <= 0 { 7 } else { 84 };
            }
            84 => {
                state = match s.ind {
                    1 => 92,
                    _ => 72,
                };
            }
            92 => {
                s.fhold = f64::from(*f);
                s.is = 6;
                s.ixp = s.jj;
                for i in 1..=n {
                    s.ixp += 1;
                    w0!(s.ixp) = f64::from(x1!(i)) - w0!(s.ixp);
                }
                s.dd = 1.;
                state = 58;
            }
            96 => {
                state = match s.ind {
                    1 => 112,
                    _ => 87,
                };
            }
            112 => {
                state = if s.fp - f64::from(*f) <= 0. { 37 } else { 91 };
            }
            91 => {
                let r1 = s.fp - f64::from(*f);
                s.d = (s.fp + f64::from(*f) - s.fhold * 2.) * 2. / (r1 * r1);
                let r2 = s.fp - s.fhold - s.sum;
                state = if s.d * (r2 * r2) - s.sum >= 0. {
                    37
                } else {
                    87
                };
            }
            87 => {
                let j = s.jil * n + 1;
                state = if j - s.jj <= 0 { 60 } else { 61 };
                if state == 60 {
                    let mut i = j;
                    while i <= s.jj {
                        s.k = i - n;
                        w0!(s.k) = w0!(i);
                        i += 1;
                    }
                    // The C writes `w[i - 2]` from `i = jil`, which is
                    // `w[-1]` when `jil` is 1 — an out-of-bounds store into
                    // whatever static precedes `w`.  That byte is not
                    // reproducible, so the copy starts at the first in-range
                    // index instead.
                    let mut i = s.jil.max(2);
                    while i <= n {
                        s.w[(i - 2) as usize] = w0!(i);
                        i += 1;
                    }
                    state = 61;
                }
            }
            61 => {
                s.idirn -= n;
                s.itone = 3;
                s.k = s.idirn;
                s.ixp = s.jj;
                s.aaa = 0.;
                for i in 1..=n {
                    s.ixp += 1;
                    w0!(s.k) = w0!(s.ixp);
                    let r1 = (w0!(s.k) / f64::from(e1!(i))).abs();
                    if s.aaa - r1 < 0. {
                        s.aaa = r1;
                    }
                    s.k += 1;
                }
                s.ddmag = 1.;
                if s.aaa == 0. {
                    s.aaa = F_1E_10;
                }
                w0!(n) = f64::from(escale) / s.aaa;
                s.iline = n;
                state = 7;
            }
            37 => {
                s.ixp = s.jj;
                s.aaa = 0.;
                *f = s.fhold as f32;
                for i in 1..=n {
                    s.ixp += 1;
                    x1!(i) = (f64::from(x1!(i)) - w0!(s.ixp)) as f32;
                    if s.aaa * f64::from(e1!(i)).abs() - w0!(s.ixp).abs() < 0. {
                        s.aaa = (w0!(s.ixp) / f64::from(e1!(i))).abs();
                    }
                }
                state = 72;
            }
            38 => {
                s.aaa *= s.di + 1.;
                state = match s.ind {
                    1 => 72,
                    _ => 106,
                };
            }
            72 => {
                state = if iprint - 2 >= 0 { 50 } else { 53 };
            }
            53 => {
                state = match s.ind {
                    1 => 109,
                    _ => 88,
                };
            }
            109 => {
                state = if s.aaa - F_0_1 <= 0. { 89 } else { 76 };
            }
            89 => {
                state = match icon {
                    1 => 20,
                    _ => 116,
                };
            }
            116 => {
                s.ind = 2;
                state = match s.inn {
                    1 => 100,
                    _ => 101,
                };
            }
            100 => {
                s.inn = 2;
                s.k = s.jjj;
                for i in 1..=n {
                    s.k += 1;
                    w0!(s.k) = f64::from(x1!(i));
                    x1!(i) += e1!(i) * 10.0f32;
                }
                s.fkeep = f64::from(*f);
                icnt += 1;
                if icnt > maxx {
                    return finish_999(&mut s, n, x, f, &mut target_function);
                }
                *f = target_function(&x[..n_usize]);
                s.nfcc += 1;
                s.ddmag = 0.;
                state = 108;
            }
            76 => {
                state = if f64::from(*f) - s.fp >= 0. { 78 } else { 35 };
            }
            78 => {
                // The source's "accuracy limited by errors in F" message is
                // commented out.
                state = 20;
            }
            88 => {
                s.ind = 1;
                state = 35;
            }
            35 => {
                let tmp = s.fp - f64::from(*f);
                if tmp > 0. {
                    s.ddmag = tmp.sqrt() * F_0_4;
                } else {
                    s.ddmag = 0.;
                }
                s.isgrad = 1;
                state = 108;
            }
            108 => {
                s.iterc += 1;
                state = if s.iterc - maxit <= 0 { 5 } else { 81 };
            }
            81 => {
                state = if f64::from(*f) - s.fkeep <= 0. {
                    20
                } else {
                    110
                };
            }
            110 => {
                *f = s.fkeep as f32;
                for i in 1..=n {
                    s.jjj += 1;
                    x1!(i) = w0!(s.jjj) as f32;
                }
                state = 20;
            }
            101 => {
                s.jil = 1;
                s.fp = s.fkeep;
                let r = f64::from(*f) - s.fkeep;
                state = if r < 0. {
                    105
                } else if r == 0. {
                    78
                } else {
                    104
                };
            }
            104 => {
                s.jil = 2;
                s.fp = f64::from(*f);
                *f = s.fkeep as f32;
                state = 105;
            }
            105 => {
                s.ixp = s.jj;
                for i in 1..=n {
                    s.ixp += 1;
                    s.k = s.ixp + n;
                    if s.jil == 1 {
                        w0!(s.ixp) = w0!(s.k);
                    } else {
                        w0!(s.ixp) = f64::from(x1!(i));
                        x1!(i) = w0!(s.k) as f32;
                    }
                }
                s.jil = 2;
                state = 92;
            }
            106 => {
                state = if s.aaa - F_0_1 <= 0. { 20 } else { 107 };
            }
            107 => {
                s.inn = 1;
                state = 35;
            }
            _ => {
                // L20: return 0;
                VA04_STATICS.with(|cell| cell.set(s));
                return 0;
            }
        }
    }
}

/// The `L999` endless-loop safety catch of `va04a_` (`va04.cpp:697`).
fn finish_999<F>(
    s: &mut Va04Statics,
    n: i32,
    x: &mut [f32],
    f: &mut f32,
    target_function: &mut F,
) -> i32
where
    F: FnMut(&[f32]) -> f32,
{
    for i in 1..=n {
        x[(i - 1) as usize] = s.xs[i as usize] as f32;
    }
    *f = target_function(&x[..n as usize]);
    VA04_STATICS.with(|cell| cell.set(*s));
    0
}

#[cfg(test)]
mod tests {
    use super::va04a;

    #[test]
    fn va04a_minimizes_a_two_parameter_quadratic() {
        let mut x = [7.0f32, -5.0];
        let mut f = 0.0f32;
        let mut calls = 0;
        assert_eq!(
            va04a(
                2,
                &[0.25, 0.25],
                100.0,
                &mut calls,
                |v| (v[0] - 2.0).powi(2) + (v[1] + 3.0).powi(2),
                &mut f,
                0,
                1,
                50,
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
