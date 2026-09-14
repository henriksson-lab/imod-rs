//! Translation of `IMOD/libcfshr/lsqr.c` and its local `lsqr.h` interface.

use core::ffi::c_void;
use std::io::Write;

use super::b3dutil::ImodFile;
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

/// Original `lsqr` (`lsqr.c:63`).
///
/// `aprod` must add `A * x` to `y` for mode 1, and `A.transpose() * y` to
/// `x` for mode 2.  The optional `nout` parameter is retained for interface
/// compatibility.
#[allow(clippy::too_many_arguments)]
pub fn lsqr<F>(
    m: usize,
    n: usize,
    mut aprod: F,
    damp: f64,
    u: &mut [f64],
    v: &mut [f64],
    w: &mut [f64],
    x: &mut [f64],
    mut se: Option<&mut [f64]>,
    atol: f64,
    btol: f64,
    conlim: f64,
    itnlim: i32,
    nout: Option<&mut ImodFile>,
    istop_out: &mut i32,
    itn_out: &mut i32,
    anorm_out: &mut f64,
    acond_out: &mut f64,
    rnorm_out: &mut f64,
    arnorm_out: &mut f64,
    xnorm_out: &mut f64,
) where
    F: FnMut(i32, &mut [f64], &mut [f64]),
{
    assert_eq!(u.len(), m);
    assert_eq!(v.len(), n);
    assert_eq!(w.len(), n);
    assert_eq!(x.len(), n);
    if let Some(se) = se.as_ref() {
        assert_eq!(se.len(), n);
    }
    let damped = damp > 0.;
    let wantse = se.is_some();
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
        let _ = write!(
            nout,
            " Enter LSQR.          Least-squares solution of  Ax = b\n The matrix  A  has {m:7} rows  and {n:7} columns\n damp   = {damp:<22.2e}    wantse = {:10}\n atol   = {atol:<22.2e}    conlim = {conlim:10.2e}\n btol   = {btol:<22.2e}    itnlim = {itnlim:10}\n\n",
            wantse as i32
        );
    }

    v.fill(0.);
    x.fill(0.);
    if let Some(se) = se.as_deref_mut() {
        se.fill(0.);
    }
    let mut alpha = 0.;
    let mut beta = cblas_dnrm2(m as i32, u, 1);
    if beta > 0. {
        cblas_dscal(m as i32, 1. / beta, u, 1);
        aprod(2, v, u);
        alpha = cblas_dnrm2(n as i32, v, 1);
    }
    if alpha > 0. {
        cblas_dscal(n as i32, 1. / alpha, v, 1);
        cblas_dcopy(n as i32, v, 1, w, 1);
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
                if damped {
                    b"    Itn       x(1)           Function     Compatible    LS      Norm Abar   Cond Abar\n"
                } else {
                    b"    Itn       x(1)           Function     Compatible    LS      Norm A   Cond A\n"
                },
            );
            let _ = writeln!(
                nout,
                " {itn:6} {:16.9e} {rnorm:16.9e} {:9.2e} {:9.2e}",
                x[0],
                1.,
                alpha / beta
            );
            let _ = nout.write_all(b"\n");
        }
        loop {
            itn += 1;
            cblas_dscal(m as i32, -alpha, u, 1);
            aprod(1, v, u);
            beta = cblas_dnrm2(m as i32, u, 1);
            let temp = d2norm(d2norm(alpha, beta), damp);
            anorm = d2norm(anorm, temp);
            if beta > 0. {
                cblas_dscal(m as i32, 1. / beta, u, 1);
                cblas_dscal(n as i32, -beta, v, 1);
                aprod(2, v, u);
                alpha = cblas_dnrm2(n as i32, v, 1);
                if alpha > 0. {
                    cblas_dscal(n as i32, 1. / alpha, v, 1);
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
            for i in 0..n {
                let old_w = w[i];
                x[i] += t1 * old_w;
                w[i] = t2 * old_w + v[i];
                let value = (t3 * old_w) * (t3 * old_w);
                if let Some(se) = se.as_deref_mut() {
                    se[i] += value;
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
                    let _ = writeln!(
                        nout,
                        " {itn:6} {:16.9e} {rnorm:16.9e} {test1:9.2e} {test2:9.2e} {anorm:8.1e} {acond:8.1e}",
                        x[0]
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
            for value in se.as_deref_mut().expect("standard errors requested") {
                *value = t * value.sqrt();
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
        let _ = write!(
            nout,
            "\n Exit  LSQR.         istop  = {istop:<10}      itn    = {itn:<10}\n Exit  LSQR.         anorm  = {anorm:11.5e}     acond  = {acond:11.5e}\n Exit  LSQR.         vnorm  = {bnorm:11.5e}     xnorm  = {xnorm:11.5e}\n Exit  LSQR.         rnorm  = {rnorm:11.5e}     arnorm = {arnorm:11.5e}\n"
        );
        let _ = write!(
            nout,
            " Exit  LSQR.         max dx = {dxmax:7.1e} occured at itn {maxdx:<9}\n Exit  LSQR.                = {:7.1e}*xnorm\n",
            dxmax / (xnorm + 1.0e-20)
        );
        let _ = writeln!(nout, " Exit  LSQR.         {message}");
    }
    *istop_out = istop;
    *itn_out = itn;
    *anorm_out = anorm;
    *acond_out = acond;
    *rnorm_out = rnorm;
    *arnorm_out = test2;
    *xnorm_out = xnorm;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal_aprod(mode: i32, x: &mut [f64], y: &mut [f64]) {
        let diagonal = [2., 3.];
        for i in 0..2 {
            if mode == 1 {
                y[i] += diagonal[i] * x[i];
            } else {
                x[i] += diagonal[i] * y[i];
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
        lsqr(
            2,
            2,
            diagonal_aprod,
            0.,
            &mut u,
            &mut v,
            &mut w,
            &mut x,
            None,
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
        lsqr(
            2,
            2,
            diagonal_aprod,
            0.,
            &mut u,
            &mut v,
            &mut w,
            &mut x,
            None,
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
        lsqr(
            2,
            2,
            diagonal_aprod,
            0.,
            &mut u,
            &mut v,
            &mut w,
            &mut x,
            None,
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
        assert!(report.starts_with(" Enter LSQR.          Least-squares solution of  Ax = b\n"));
        assert!(report.contains(
            "    Itn       x(1)           Function     Compatible    LS      Norm A   Cond A\n"
        ));
        assert!(report.contains(" Exit  LSQR.         istop  = 1"));
        assert!(
            report.ends_with(
                " Exit  LSQR.         A solution to Ax = b was found, given atol, btol\n"
            )
        );
    }
}
