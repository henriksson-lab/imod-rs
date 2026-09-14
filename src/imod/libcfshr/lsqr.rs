//! Translation of `IMOD/libcfshr/lsqr.c` and its local `lsqr.h` interface.

use core::ffi::c_void;
use std::io::Write;

use super::b3dutil::{CArg, ImodFile, c_format};
use super::lsqrblas::{cblas_dcopy, cblas_dnrm2, cblas_dscal};

/// Matrix product callback from `lsqr.h`.
pub type Aprod = unsafe extern "C" fn(i32, i32, i32, *mut f64, *mut f64, *mut c_void);

/// Original `d2norm` (`lsqr.c:40`).
fn d2norm(a: f64, b: f64) -> f64 {
    let scale = a.abs() + b.abs();
    if scale == 0. {
        0.
    } else {
        scale * ((a / scale) * (a / scale) + (b / scale) * (b / scale)).sqrt()
    }
}

/// Original `dload` (`lsqr.c:53`).
unsafe fn dload(n: i32, alpha: f64, x: *mut f64) {
    for i in 0..n {
        unsafe { *x.add(i as usize) = alpha };
    }
}

/// Original `lsqr` (`lsqr.c:63`).
///
/// `aprod` must add `A * x` to `y` for mode 1, and `A.transpose() * y` to
/// `x` for mode 2.  The optional `nout` parameter is retained for interface
/// compatibility.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lsqr(
    m: i32,
    n: i32,
    aprod: Aprod,
    damp: f64,
    user_work: *mut c_void,
    u: *mut f64,
    v: *mut f64,
    w: *mut f64,
    x: *mut f64,
    se: *mut f64,
    atol: f64,
    btol: f64,
    conlim: f64,
    itnlim: i32,
    nout: Option<&mut ImodFile>,
    istop_out: *mut i32,
    itn_out: *mut i32,
    anorm_out: *mut f64,
    acond_out: *mut f64,
    rnorm_out: *mut f64,
    arnorm_out: *mut f64,
    xnorm_out: *mut f64,
) {
    let damped = damp > 0.;
    let wantse = !se.is_null();
    let mut istop = 0;
    let mut itn = 0;
    let mut nstop = 0;
    let mut maxdx = 0;
    let ctol = if conlim > 0. { 1. / conlim } else { 0. };
    let mut anorm = 0.;
    let mut acond = 0.;
    let mut dnorm = 0.;
    let mut dxmax = 0.;
    let mut res2 = 0.;
    let mut psi = 0.;
    let mut xnorm = 0.;
    let mut xnorm1 = 0.;
    let mut cs2 = -1.;
    let mut sn2 = 0.;
    let mut z = 0.;

    let mut nout = nout;
    if let Some(nout) = nout.as_deref_mut() {
        let _ = nout.write_all(
            c_format(
                " %s        Least-squares solution of  Ax = b\n The matrix  A  has %7d rows  and %7d columns\n damp   = %-22.2e    wantse = %10i\n atol   = %-22.2e    conlim = %10.2e\n btol   = %-22.2e    itnlim = %10d\n\n",
                &[
                    CArg::Str("Enter LSQR.  "),
                    CArg::Int(m as i64),
                    CArg::Int(n as i64),
                    CArg::Dbl(damp),
                    CArg::Int(wantse as i64),
                    CArg::Dbl(atol),
                    CArg::Dbl(conlim),
                    CArg::Dbl(btol),
                    CArg::Int(itnlim as i64),
                ],
            )
            .as_bytes(),
        );
    }

    unsafe {
        dload(n, 0., v);
        dload(n, 0., x);
    }
    if wantse {
        unsafe { dload(n, 0., se) };
    }
    let mut alpha = 0.;
    let mut beta = unsafe { cblas_dnrm2(m, core::slice::from_raw_parts(u, m as usize), 1) };
    if beta > 0. {
        unsafe {
            cblas_dscal(
                m,
                1. / beta,
                core::slice::from_raw_parts_mut(u, m as usize),
                1,
            );
            aprod(2, m, n, v, u, user_work);
        }
        alpha = unsafe { cblas_dnrm2(n, core::slice::from_raw_parts(v, n as usize), 1) };
    }
    if alpha > 0. {
        unsafe {
            cblas_dscal(
                n,
                1. / alpha,
                core::slice::from_raw_parts_mut(v, n as usize),
                1,
            );
            cblas_dcopy(
                n,
                core::slice::from_raw_parts(v, n as usize),
                1,
                core::slice::from_raw_parts_mut(w, n as usize),
                1,
            );
        }
    }
    let mut arnorm = alpha * beta;
    let bnorm = beta;
    let mut rnorm = beta;
    let mut test2 = 0.;
    if arnorm != 0. {
        let mut rhobar = alpha;
        let mut phibar = beta;
        if let Some(nout) = nout.as_deref_mut() {
            let _ = nout.write_all(
                c_format(
                    if damped {
                        "    Itn       x(1)           Function     Compatible    LS      Norm Abar   Cond Abar\n"
                    } else {
                        "    Itn       x(1)           Function     Compatible    LS      Norm A   Cond A\n"
                    },
                    &[],
                )
                .as_bytes(),
            );
            let _ = nout.write_all(
                c_format(
                    " %6d %16.9e %16.9e %9.2e %9.2e\n",
                    &[
                        CArg::Int(itn as i64),
                        CArg::Dbl(unsafe { *x }),
                        CArg::Dbl(rnorm),
                        CArg::Dbl(1.),
                        CArg::Dbl(alpha / beta),
                    ],
                )
                .as_bytes(),
            );
            let _ = nout.write_all(b"\n");
        }
        loop {
            itn += 1;
            unsafe {
                cblas_dscal(m, -alpha, core::slice::from_raw_parts_mut(u, m as usize), 1);
                aprod(1, m, n, v, u, user_work);
            }
            beta = unsafe { cblas_dnrm2(m, core::slice::from_raw_parts(u, m as usize), 1) };
            let temp = d2norm(d2norm(alpha, beta), damp);
            anorm = d2norm(anorm, temp);
            if beta > 0. {
                unsafe {
                    cblas_dscal(
                        m,
                        1. / beta,
                        core::slice::from_raw_parts_mut(u, m as usize),
                        1,
                    );
                    cblas_dscal(n, -beta, core::slice::from_raw_parts_mut(v, n as usize), 1);
                    aprod(2, m, n, v, u, user_work);
                }
                alpha = unsafe { cblas_dnrm2(n, core::slice::from_raw_parts(v, n as usize), 1) };
                if alpha > 0. {
                    unsafe {
                        cblas_dscal(
                            n,
                            1. / alpha,
                            core::slice::from_raw_parts_mut(v, n as usize),
                            1,
                        )
                    };
                }
            }
            let mut rhbar1 = rhobar;
            if damped {
                rhbar1 = d2norm(rhobar, damp);
                let cs1 = rhobar / rhbar1;
                let sn1 = damp / rhbar1;
                psi = sn1 * phibar;
                phibar = cs1 * phibar;
            }
            let rho = d2norm(rhbar1, beta);
            let cs = rhbar1 / rho;
            let sn = beta / rho;
            let theta = sn * alpha;
            rhobar = -cs * alpha;
            let phi = cs * phibar;
            phibar = sn * phibar;
            let tau = sn * phi;
            let t1 = phi / rho;
            let t2 = -theta / rho;
            let t3 = 1. / rho;
            let mut dknorm = 0.;
            for i in 0..n as usize {
                let old_w = unsafe { *w.add(i) };
                unsafe {
                    *x.add(i) += t1 * old_w;
                    *w.add(i) = t2 * old_w + *v.add(i);
                }
                let value = (t3 * old_w) * (t3 * old_w);
                if wantse {
                    unsafe {
                        *se.add(i) += value;
                    }
                }
                dknorm += value;
            }
            dknorm = dknorm.sqrt();
            dnorm = d2norm(dnorm, dknorm);
            let dxk = (phi * dknorm).abs();
            if dxmax < dxk {
                dxmax = dxk;
                maxdx = itn;
            }
            let delta = sn2 * rho;
            let gambar = -cs2 * rho;
            let rhs = phi - delta * z;
            let zbar = rhs / gambar;
            xnorm = d2norm(xnorm1, zbar);
            let gamma = d2norm(gambar, theta);
            cs2 = gambar / gamma;
            sn2 = theta / gamma;
            z = rhs / gamma;
            xnorm1 = d2norm(xnorm1, z);
            acond = anorm * dnorm;
            res2 = d2norm(res2, psi);
            rnorm = d2norm(res2, phibar);
            arnorm = alpha * tau.abs();
            let _alfopt = (rnorm / (dnorm * xnorm)).sqrt();
            let test1 = rnorm / bnorm;
            test2 = if rnorm > 0. {
                arnorm / (anorm * rnorm)
            } else {
                0.
            };
            let test3 = 1. / acond;
            let t1 = test1 / (1. + anorm * xnorm / bnorm);
            let rtol = btol + atol * anorm * xnorm / bnorm;
            if itn >= itnlim {
                istop = 5;
            }
            if 1. + test3 <= 1. {
                istop = 4;
            }
            if 1. + test2 <= 1. {
                istop = 2;
            }
            if 1. + t1 <= 1. {
                istop = 1;
            }
            if test3 <= ctol {
                istop = 4;
            }
            if test2 <= atol {
                istop = 2;
            }
            if test1 <= rtol {
                istop = 1;
            }
            if nout.is_some()
                && (n <= 40
                    || itn <= 10
                    || itn >= itnlim - 10
                    || itn % 10 == 0
                    || test3 <= 2. * ctol
                    || test2 <= 10. * atol
                    || test1 <= 10. * rtol
                    || istop != 0)
            {
                if let Some(nout) = nout.as_deref_mut() {
                    let _ = nout.write_all(
                        c_format(
                            " %6d %16.9e %16.9e %9.2e %9.2e %8.1e %8.1e\n",
                            &[
                                CArg::Int(itn as i64),
                                CArg::Dbl(unsafe { *x }),
                                CArg::Dbl(rnorm),
                                CArg::Dbl(test1),
                                CArg::Dbl(test2),
                                CArg::Dbl(anorm),
                                CArg::Dbl(acond),
                            ],
                        )
                        .as_bytes(),
                    );
                    if itn % 10 == 0 {
                        let _ = nout.write_all(b"\n");
                    }
                }
            }
            if istop == 0 {
                nstop = 0;
            } else {
                nstop += 1;
                if nstop < 1 && itn < itnlim {
                    istop = 0;
                }
            }
            if istop != 0 {
                break;
            }
        }
        if wantse {
            let mut t = 1.;
            if m > n {
                t = (m - n) as f64;
            }
            if damped {
                t = m as f64;
            }
            t = rnorm / t.sqrt();
            for i in 0..n as usize {
                unsafe {
                    *se.add(i) = t * (*se.add(i)).sqrt();
                }
            }
        }
    }
    if damped && istop == 2 {
        istop = 3;
    }
    if let Some(nout) = nout.as_deref_mut() {
        let message = match istop {
            0 => "The exact solution is  x = 0",
            1 => "A solution to Ax = b was found, given atol, btol",
            2 => "A least-squares solution was found, given atol",
            3 => "A damped least-squares solution was found, given atol",
            4 => "Cond(Abar) seems to be too large, given conlim",
            _ => "The iteration limit was reached",
        };
        let _ = nout.write_all(
            c_format(
                "\n %s       istop  = %-10d      itn    = %-10d\n %s       anorm  = %11.5e     acond  = %11.5e\n %s       vnorm  = %11.5e     xnorm  = %11.5e\n %s       rnorm  = %11.5e     arnorm = %11.5e\n",
                &[
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Int(istop as i64),
                    CArg::Int(itn as i64),
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Dbl(anorm),
                    CArg::Dbl(acond),
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Dbl(bnorm),
                    CArg::Dbl(xnorm),
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Dbl(rnorm),
                    CArg::Dbl(arnorm),
                ],
            )
            .as_bytes(),
        );
        let _ = nout.write_all(
            c_format(
                " %s       max dx = %7.1e occured at itn %-9d\n %s              = %7.1e*xnorm\n",
                &[
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Dbl(dxmax),
                    CArg::Int(maxdx as i64),
                    CArg::Str("Exit  LSQR.  "),
                    CArg::Dbl(dxmax / (xnorm + 1.0e-20)),
                ],
            )
            .as_bytes(),
        );
        let _ = nout.write_all(
            c_format(
                " %s       %s\n",
                &[CArg::Str("Exit  LSQR.  "), CArg::Str(message)],
            )
            .as_bytes(),
        );
    }
    unsafe {
        *istop_out = istop;
        *itn_out = itn;
        *anorm_out = anorm;
        *acond_out = acond;
        *rnorm_out = rnorm;
        *arnorm_out = test2;
        *xnorm_out = xnorm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn diagonal_aprod(
        mode: i32,
        _m: i32,
        _n: i32,
        x: *mut f64,
        y: *mut f64,
        _: *mut c_void,
    ) {
        let diagonal = [2., 3.];
        for i in 0..2 {
            unsafe {
                if mode == 1 {
                    *y.add(i) += diagonal[i] * *x.add(i);
                } else {
                    *x.add(i) += diagonal[i] * *y.add(i);
                }
            }
        }
    }

    #[test]
    fn solves_a_compatible_diagonal_system() {
        let mut u = [4., 9.];
        let mut v = [0.; 2];
        let mut w = [0.; 2];
        let mut x = [0.; 2];
        let (mut stop, mut iterations) = (0, 0);
        let (mut anorm, mut acond, mut rnorm, mut arnorm, mut xnorm) = (0., 0., 0., 0., 0.);
        unsafe {
            lsqr(
                2,
                2,
                diagonal_aprod,
                0.,
                core::ptr::null_mut(),
                u.as_mut_ptr(),
                v.as_mut_ptr(),
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                core::ptr::null_mut(),
                1.0e-12,
                1.0e-12,
                1.0e12,
                20,
                None,
                &mut stop,
                &mut iterations,
                &mut anorm,
                &mut acond,
                &mut rnorm,
                &mut arnorm,
                &mut xnorm,
            );
        }
        assert_eq!(stop, 1);
        assert!(iterations <= 2);
        assert!((x[0] - 2.).abs() < 1.0e-10 && (x[1] - 3.).abs() < 1.0e-10);
        assert!(rnorm < 1.0e-10 && anorm > 0. && acond >= 1. && xnorm > 0.);
    }

    #[test]
    fn reports_zero_solution_for_zero_rhs() {
        let mut u = [0., 0.];
        let mut v = [9.; 2];
        let mut w = [9.; 2];
        let mut x = [9.; 2];
        let (mut stop, mut iterations) = (9, 9);
        let (mut anorm, mut acond, mut rnorm, mut arnorm, mut xnorm) = (9., 9., 9., 9., 9.);
        unsafe {
            lsqr(
                2,
                2,
                diagonal_aprod,
                0.,
                core::ptr::null_mut(),
                u.as_mut_ptr(),
                v.as_mut_ptr(),
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                core::ptr::null_mut(),
                1.0e-12,
                1.0e-12,
                1.0e12,
                20,
                None,
                &mut stop,
                &mut iterations,
                &mut anorm,
                &mut acond,
                &mut rnorm,
                &mut arnorm,
                &mut xnorm,
            );
        }
        assert_eq!((stop, iterations), (0, 0));
        assert_eq!(x, [0., 0.]);
    }

    #[test]
    fn writes_source_style_progress_and_final_report() {
        let mut output = ImodFile::tmpfile().unwrap();
        let mut u = [4., 9.];
        let mut v = [0.; 2];
        let mut w = [0.; 2];
        let mut x = [0.; 2];
        let (mut stop, mut iterations) = (0, 0);
        let (mut anorm, mut acond, mut rnorm, mut arnorm, mut xnorm) = (0., 0., 0., 0., 0.);
        unsafe {
            lsqr(
                2,
                2,
                diagonal_aprod,
                0.,
                core::ptr::null_mut(),
                u.as_mut_ptr(),
                v.as_mut_ptr(),
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                core::ptr::null_mut(),
                1.0e-12,
                1.0e-12,
                1.0e12,
                20,
                Some(&mut output),
                &mut stop,
                &mut iterations,
                &mut anorm,
                &mut acond,
                &mut rnorm,
                &mut arnorm,
                &mut xnorm,
            );
            use std::io::{Read, Seek, SeekFrom};
            let _ = output.flush();
            let _ = output.seek(SeekFrom::Start(0));
            let mut bytes = Vec::new();
            let _ = output.read_to_end(&mut bytes);
            let report = core::str::from_utf8(&bytes).unwrap();
            assert!(
                report.starts_with(" Enter LSQR.          Least-squares solution of  Ax = b\n")
            );
            assert!(report.contains(
                "    Itn       x(1)           Function     Compatible    LS      Norm A   Cond A\n"
            ));
            assert!(report.contains(" Exit  LSQR.         istop  = 1"));
            assert!(report.ends_with(
                " Exit  LSQR.         A solution to Ax = b was found, given atol, btol\n"
            ));
        }
    }
}
