//! Translation of `IMOD/flib/subrs/graphics/pssymbol.f90`: symbol drawing
//! on the PostScript plot.

use super::psf::{pscircle, psquadrangle, pstriangle, pswritetext};
use super::psplotpak::{
    N_THICK, SYM_SCALE, UNITS_PER_INCH, ps_move_abs, ps_setup_thick, ps_vect_abs,
};
use crate::imod::flib::subrs::compat::gfortran_rt::nint_r4;

/// `data index/.../` (`pssymbol.f90:8`).
const INDEX: [i32; 21] = [
    1, 1, 5, 5, 9, 9, 12, 16, -1, -1, 20, 20, 23, 29, 43, -2, -1, -1, 41, 43, 41,
];
/// `data ifFill /.../` (`pssymbol.f90:11`).
const IF_FILL: [i32; 21] = [
    0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
];
/// `data ifCircle /.../` (`pssymbol.f90:12`).
const IF_CIRCLE: [i32; 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
];
/// `real*4 vec(3, 44)` (`pssymbol.f90:14-25`), in storage order: square,
/// diamond, triangle, X, cross, down triangle, U, S, -, |.
const VEC: [f32; 132] = [
    -0.5, -0.5, -1., 0.5, -0.5, 0., 0.5, 0.5, 0., -0.5, 0.5, 1., //
    -0.5, 0., -1., 0., -0.625, 0., 0.5, 0., 0., 0., 0.625, 1., //
    -0.5, -0.289, -1., 0.5, -0.289, 0., 0., 0.577, 1., //
    -0.4, -0.4, -1., 0.4, 0.4, 0., -0.4, 0.4, -1., 0.4, -0.4, 1., //
    -0.5, 0., -1., 0.5, 0., 0., 0., -0.5, -1., 0., 0.5, 1., //
    -0.5, 0.289, -1., 0.5, 0.289, 0., 0., -0.577, 1., //
    -0.38, 0.5, -1., -0.38, -0.38, 0., -0.26, -0.5, 0., 0.26, -0.5, 0., 0.38, -0.38, 0., //
    0.38, 0.5, 1., //
    0.38, 0.38, -1., 0.26, 0.5, 0., -0.26, 0.5, 0., -0.38, 0.38, 0., -0.38, 0.12, 0., //
    -0.26, 0., 0., 0.26, 0., 0., 0.38, -0.12, 0., 0.38, -0.38, 0., 0.26, -0.5, 0., //
    -0.26, -0.5, 0., -0.38, -0.38, 1., //
    -0.5, 0., -1., 0.5, 0., 1., 0., -0.5, -1., 0., 0.5, 1.,
];

/// `vec(k, ind)`, 1-based.
fn vec(k: usize, ind: i32) -> f32 {
    VEC[(ind as usize - 1) * 3 + (k - 1)]
}

/// Original `psSymbol` (`pssymbol.f90:5`): draws symbol `itype` at `x, y`
/// (inches); a negative type writes the number.
///
/// Fixed in translation (BUGS.md, `psSymbol`): a number of five or more
/// digits overflows the source's `character*4` and its substring; here the
/// number is written in full.
pub fn ps_symbol(x: f32, y: f32, itype: i32) {
    let ring_frac: f32 = 0.45;
    if itype == 0 || itype > 21 {
        return;
    }
    let sym_scale = SYM_SCALE.get();
    if itype < 0 {
        let iscale = nint_r4(107. * sym_scale);
        let dumm2 = (-itype).to_string();
        pswritetext(x + 0.016, y - 0.018, dumm2.as_bytes(), iscale, 0, 0);
        return;
    }
    let t = itype as usize - 1;
    let mut ind = INDEX[t];
    let mut scale_vec = sym_scale;
    if IF_FILL[t] != 0 {
        scale_vec = sym_scale + (N_THICK.get() - 1) as f32 / UNITS_PER_INCH.get();
    }
    let mut ind_vec = 0usize;
    let mut xvec = [0f32; 4];
    let mut yvec = [0f32; 4];
    // `go to (20, 20, 20, 20, 20, 20, 10, 10, 30, 30, 20, 20, 10, 10, 10, 50,
    // 40, 40, 10, 10, 10), itype`
    let target = [
        20, 20, 20, 20, 20, 20, 10, 10, 30, 30, 20, 20, 10, 10, 10, 50, 40, 40, 10, 10, 10,
    ][t];
    let circle = |scale_vec: f32| pscircle(x, y, scale_vec / 2., IF_FILL[t]);
    match target {
        10 => {
            //
            // arbitrary path
            //
            loop {
                let xx = x + sym_scale * vec(1, ind);
                let yy = y + sym_scale * vec(2, ind);
                let instruct = nint_r4(vec(3, ind));
                if instruct < 0 {
                    ps_move_abs(xx, yy);
                }
                if instruct >= 0 {
                    ps_vect_abs(xx, yy);
                }
                if instruct > 0 {
                    if IF_CIRCLE[t] != 0 {
                        circle(scale_vec);
                    }
                    return;
                }
                ind += 1;
            }
        }
        20 => {
            //
            // closed path
            //
            loop {
                ind_vec += 1;
                xvec[ind_vec - 1] = x + scale_vec * vec(1, ind);
                yvec[ind_vec - 1] = y + scale_vec * vec(2, ind);
                let instruct = nint_r4(vec(3, ind));
                ind += 1;
                if instruct > 0 {
                    break;
                }
            }
            if ind_vec == 3 {
                pstriangle(&xvec, &yvec, IF_FILL[t]);
            }
            if ind_vec == 4 {
                psquadrangle(&xvec, &yvec, IF_FILL[t]);
            }
            if IF_CIRCLE[t] != 0 {
                circle(scale_vec);
            }
        }
        30 => {
            //
            // circle, open or filled
            //
            circle(scale_vec);
        }
        40 => {
            //
            // Dot
            //
            pscircle(x, y, 2.max(N_THICK.get()) as f32 / UNITS_PER_INCH.get(), 1);
            if IF_CIRCLE[t] != 0 {
                circle(scale_vec);
            }
        }
        _ => {
            //
            // Ring, select a thickness to cover given fraction
            //
            let nthk_save = N_THICK.get();
            let ring_scale = 0.5 * scale_vec;
            N_THICK.set(nint_r4(ring_frac * ring_scale * UNITS_PER_INCH.get()) + 1);
            ps_setup_thick(N_THICK.get());
            pscircle(x, y, (1. - ring_frac / 2.) * ring_scale, 0);
            ps_setup_thick(nthk_save);
        }
    }
}

/// Original `psSymSize` (`pssymbol.f90:95`).
pub fn ps_sym_size(size: f32) {
    SYM_SCALE.set(size);
}
