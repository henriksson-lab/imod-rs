//! Translation of `IMOD/libcfshr/gcvspl.c` (f2c output of the GCVSPL
//! smoothing-spline package, with Woltring's C entry points on top).
//!
//! The f2c source biases each array pointer by its Fortran lower bound
//! (`e -= e_offset`, `--wk`) and then indexes it with the original 1-based
//! subscript.  A Rust slice cannot start before its allocation, so the bias is
//! applied at the point of use instead — every `e[i]` in the C is `e[(i -
//! e_offset) as usize]` here, with the same `e_offset` computed in the same
//! place.  The loop bounds and subscript expressions are otherwise untouched.
//!
//! f2c also gives every local the `static` storage class.  Only `m2`, `nm1`
//! and `el` in `gcvspl_` are genuine Fortran SAVEs (they carry a DATA
//! initialiser and an `md < 0` call compares against them); the rest are
//! assigned before use on every call and are plain locals here.
#![allow(unused_mut, unused_assignments, clippy::too_many_arguments)]

use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};

/// f2c constant `c_b6`, the `eps` passed to `splc_`.
const C_B6: f64 = 1e-15;

/// SAVEd `m2` of `gcvspl_` (`gcvspl.c:296`).
static M2: AtomicI32 = AtomicI32::new(0);
/// SAVEd `nm1` of `gcvspl_` (`gcvspl.c:297`).
static NM1: AtomicI32 = AtomicI32::new(0);
/// SAVEd `el` of `gcvspl_` (`gcvspl.c:298`), held as `f64` bits.
static EL: AtomicU64 = AtomicU64::new(0);

/// Original `gcvspl` (`gcvspl.c:35`), the C-callable entry point.
pub fn gcvspl(
    x: &[f64],
    y: &[f64],
    y_dim: i32,
    wgtx: &[f64],
    wgty: &[f64],
    m_order: i32,
    num_val: i32,
    num_ycol: i32,
    mode: i32,
    val: f64,
    coeff: &mut [f64],
    coeff_dim: i32,
    work: &mut [f64],
    ier: &mut i32,
) -> i32 {
    gcvspl_(
        x, y, y_dim, wgtx, wgty, m_order, num_val, num_ycol, mode, val, coeff, coeff_dim, work, ier,
    )
}

/// Original `splder` (`gcvspl.c:43`), the C-callable entry point.
pub fn splder(
    deriv_order: i32,
    m_order: i32,
    num_val: i32,
    t_val: f64,
    x: &[f64],
    coeff: &[f64],
    near_ind: &mut i32,
    work: &mut [f64],
) -> f64 {
    splder_(
        deriv_order,
        m_order,
        num_val,
        t_val,
        x,
        coeff,
        near_ind,
        work,
    )
}

/// Original `gcvspl_` (`gcvspl.c:290`).
pub fn gcvspl_(
    x: &[f64],
    y: &[f64],
    ny: i32,
    wx: &[f64],
    wy: &[f64],
    m: i32,
    n: i32,
    k: i32,
    md: i32,
    val: f64,
    c__: &mut [f64],
    nc: i32,
    wk: &mut [f64],
    ier: &mut i32,
) -> i32 {
    let mut current_block: u64;
    /* `m2`, `nm1` and `el` carry the DATA-initialised SAVE attribute
    (`gcvspl.c:295-299`): they persist between calls and an `md < 0` call
    checks them against the previous one. */
    let mut m2: i32 = M2.load(Ordering::Relaxed);
    let mut nm1: i32 = NM1.load(Ordering::Relaxed);
    let mut el: f64 = f64::from_bits(EL.load(Ordering::Relaxed));
    let mut y_dim1: i32 = 0;
    let mut y_offset: i32 = 0;
    let mut c_dim1: i32 = 0;
    let mut c_offset: i32 = 0;
    let mut i__1: i32 = 0;
    let mut i__: i32 = 0;
    let mut j: i32 = 0;
    let mut r1: f64 = 0.;
    let mut r2: f64 = 0.;
    let mut r3: f64 = 0.;
    let mut r4: f64 = 0.;
    let mut ib: i32 = 0;
    let mut gf1: f64 = 0.;
    let mut gf2: f64 = 0.;
    let mut gf3: f64 = 0.;
    let mut gf4: f64 = 0.;
    let mut iwe: i32 = 0;
    let mut err: f64 = 0.;
    let mut nm2m1: i32 = 0;
    let mut nm2p1: i32 = 0;
    let mut alpha: f64 = 0.;
    y_dim1 = ny;
    y_offset = 1 as i32 + y_dim1;
    c_dim1 = nc;
    c_offset = 1 as i32 + c_dim1;
    *ier = 0 as i32;
    if (if md >= 0 as i32 { md } else { -md }) > 4 as i32
        || md == 0 as i32
        || (if md >= 0 as i32 { md } else { -md }) == 1 as i32 && val < 0.0f64
        || (if md >= 0 as i32 { md } else { -md }) == 3 as i32 && val < 0.0f64
        || (if md >= 0 as i32 { md } else { -md }) == 4 as i32
            && (val < 0.0f64 || val > (n - m) as f64)
    {
        *ier = 3 as i32;
        return 0 as i32;
    }
    if md > 0 as i32 {
        m2 = m << 1 as i32;
        nm1 = (n - 1 as i32) as i32;
        M2.store(m2, Ordering::Relaxed);
        NM1.store(nm1, Ordering::Relaxed);
    } else if m2 != m << 1 as i32 || nm1 != n - 1 as i32 {
        *ier = 3 as i32;
        return 0 as i32;
    }
    if m <= 0 as i32 || n < m2 {
        *ier = 1 as i32;
        return 0 as i32;
    }
    if wx[(1 as i32 - 1) as usize] <= 0.0f64 {
        *ier = 2 as i32;
    }
    i__1 = n;
    i__ = 2 as i32;
    while i__ <= i__1 {
        if wx[(i__ - 1) as usize] <= 0.0f64
            || x[((i__ as i32 - 1 as i32) - 1) as usize] >= x[(i__ - 1) as usize]
        {
            *ier = 2 as i32;
        }
        if *ier != 0 as i32 {
            return 0 as i32;
        }
        i__ += 1;
    }
    i__1 = k;
    j = 1 as i32;
    while j <= i__1 {
        if wy[(j - 1) as usize] <= 0.0f64 {
            *ier = 2 as i32;
        }
        if *ier != 0 as i32 {
            return 0 as i32;
        }
        j += 1;
    }
    nm2p1 = (n * (m2 as i32 + 1 as i32)) as i32;
    nm2m1 = (n * (m2 as i32 - 1 as i32)) as i32;
    ib = (nm2p1 as i32 + 7 as i32) as i32;
    iwe = ib + nm2m1;
    /* The C hands `splc_` four spans of the one work array: STAT at wk[1],
    BWE at wk[7], B at wk[ib] and WE at wk[iwe], with lengths 6, N*(2M+1),
    N*(2M-1) and N*(2M+1).  They are disjoint, so the same partition is made
    once here with `split_at_mut` at exactly those boundaries. */
    let (wk_stat, wk_rest) = wk.split_at_mut(6);
    let (wk_bwe, wk_rest) = wk_rest.split_at_mut((ib - 1 - 6) as usize);
    let (wk_b, wk_we) = wk_rest.split_at_mut((iwe - ib) as usize);
    if md > 0 as i32 {
        basis_(m, n, x, wk_b, &mut r1, wk_bwe);
        prep_(m, n, x, wx, wk_we, &mut el);
        el /= r1 as f64;
        EL.store(el.to_bits(), Ordering::Relaxed);
    }
    if (if md >= 0 as i32 { md } else { -md }) != 1 as i32 {
        if md < -(1 as i32) {
            r1 = wk_stat[(4 as i32 - 1) as usize];
        } else {
            r1 = 1.0f64 / el;
        }
        r2 = (r1 as f64 * 2.0f64) as f64;
        gf2 = splc_(
            m, n, k, y, ny, wx, wy, md, val, r2, C_B6, c__, nc, wk_stat, wk_b, wk_we, el, wk_bwe,
        );
        loop {
            gf1 = splc_(
                m, n, k, y, ny, wx, wy, md, val, r1, C_B6, c__, nc, wk_stat, wk_b, wk_we, el,
                wk_bwe,
            );
            if gf1 > gf2 {
                r3 = (r2 as f64 * 2.0f64) as f64;
                current_block = 8394302012487765337;
                break;
            } else {
                if wk_stat[(4 as i32 - 1) as usize] <= 0.0f64 {
                    current_block = 3190149595005962170;
                    break;
                }
                r2 = r1;
                gf2 = gf1;
                r1 /= 2.0f64;
            }
        }
        match current_block {
            3190149595005962170 => {}
            _ => {
                loop {
                    gf3 = splc_(
                        m, n, k, y, ny, wx, wy, md, val, r3, C_B6, c__, nc, wk_stat, wk_b, wk_we,
                        el, wk_bwe,
                    );
                    if gf3 > gf2 {
                        r2 = r3;
                        gf2 = gf3;
                        alpha = ((r2 as f64 - r1 as f64) / 1.618033983f64) as f64;
                        r4 = r1 + alpha;
                        r3 = r2 - alpha;
                        gf3 = splc_(
                            m, n, k, y, ny, wx, wy, md, val, r3, C_B6, c__, nc, wk_stat, wk_b,
                            wk_we, el, wk_bwe,
                        );
                        gf4 = splc_(
                            m, n, k, y, ny, wx, wy, md, val, r4, C_B6, c__, nc, wk_stat, wk_b,
                            wk_we, el, wk_bwe,
                        );
                        current_block = 5432794640721371522;
                        break;
                    } else {
                        if wk_stat[(4 as i32 - 1) as usize] >= 999999999999999.88f64 {
                            current_block = 3190149595005962170;
                            break;
                        }
                        r2 = r3;
                        gf2 = gf3;
                        r3 *= 2.0f64;
                    }
                }
                match current_block {
                    3190149595005962170 => {}
                    _ => {
                        loop {
                            if gf3 <= gf4 {
                                r2 = r4;
                                gf2 = gf4;
                                err = (r2 - r1) / (r1 + r2);
                                if err as f64 * err as f64 + 1.0f64 == 1.0f64 || err <= 1e-6f64 {
                                    break;
                                }
                                r4 = r3;
                                gf4 = gf3;
                                alpha /= 1.618033983f64;
                                r3 = r2 - alpha;
                                gf3 = splc_(
                                    m, n, k, y, ny, wx, wy, md, val, r3, C_B6, c__, nc, wk_stat,
                                    wk_b, wk_we, el, wk_bwe,
                                );
                            } else {
                                r1 = r3;
                                gf1 = gf3;
                                err = (r2 - r1) / (r1 + r2);
                                if err as f64 * err as f64 + 1.0f64 == 1.0f64 || err <= 1e-6f64 {
                                    break;
                                }
                                r3 = r4;
                                gf3 = gf4;
                                alpha /= 1.618033983f64;
                                r4 = r1 + alpha;
                                gf4 = splc_(
                                    m, n, k, y, ny, wx, wy, md, val, r4, C_B6, c__, nc, wk_stat,
                                    wk_b, wk_we, el, wk_bwe,
                                );
                            }
                        }
                        r1 = ((r1 as f64 + r2 as f64) * 0.5f64) as f64;
                    }
                }
            }
        }
    } else {
        r1 = val;
    }
    gf1 = splc_(
        m, n, k, y, ny, wx, wy, md, val, r1, C_B6, c__, nc, wk_stat, wk_b, wk_we, el, wk_bwe,
    );
    return 0 as i32;
}

/// Original `basis_` (`gcvspl.c:566`).
pub fn basis_(m_0: i32, n_0: i32, x_0: &[f64], b: &mut [f64], bl: &mut f64, q: &mut [f64]) -> i32 {
    let mut b_dim1: i32 = 0;
    let mut b_offset: i32 = 0;
    let mut q_offset: i32 = 0;
    let mut i__1_0: i32 = 0;
    let mut i__2: i32 = 0;
    let mut i__3: i32 = 0;
    let mut i__4: i32 = 0;
    let mut d__1: f64 = 0.;
    let mut i___0: i32 = 0;
    let mut j_0: i32 = 0;
    let mut k_0: i32 = 0;
    let mut l: i32 = 0;
    let mut u: f64 = 0.;
    let mut v: f64 = 0.;
    let mut y_0: f64 = 0.;
    let mut j1: i32 = 0;
    let mut j2: i32 = 0;
    let mut m2_0: i32 = 0;
    let mut ir: i32 = 0;
    let mut mm1: i32 = 0;
    let mut mp1: i32 = 0;
    let mut arg: f64 = 0.;
    let mut nmip1: i32 = 0;
    q_offset = 1 as i32 - m_0;
    b_dim1 = (m_0 - 1 as i32 - (1 as i32 - m_0) + 1 as i32) as i32;
    b_offset = 1 as i32 - m_0 + b_dim1;
    if m_0 == 1 as i32 {
        i__1_0 = n_0;
        i___0 = 1 as i32;
        while i___0 <= i__1_0 {
            b[((i___0 * b_dim1) - b_offset) as usize] = 1.0f64 as f64;
            i___0 += 1;
        }
        *bl = 1.0f64 as f64;
        return 0 as i32;
    }
    mm1 = (m_0 - 1 as i32) as i32;
    mp1 = (m_0 + 1 as i32) as i32;
    m2_0 = m_0 << 1 as i32;
    i__1_0 = n_0;
    l = 1 as i32;
    while l <= i__1_0 {
        i__2 = m_0;
        j_0 = -mm1;
        while j_0 <= i__2 {
            q[(j_0 - q_offset) as usize] = 0.0f64 as f64;
            j_0 += 1;
        }
        q[(mm1 - q_offset) as usize] = 1.0f64 as f64;
        if l != 1 as i32 && l != n_0 {
            q[(mm1 - q_offset) as usize] = 1.0f64
                / (x_0[((l as i32 + 1 as i32) - 1) as usize]
                    - x_0[((l as i32 - 1 as i32) - 1) as usize]);
        }
        arg = x_0[(l - 1) as usize];
        i__2 = m2_0;
        i___0 = 3 as i32;
        while i___0 <= i__2 {
            ir = mp1 - i___0;
            v = q[(ir - q_offset) as usize];
            if l < i___0 {
                i__3 = i___0;
                j_0 = (l as i32 + 1 as i32) as i32;
                while j_0 <= i__3 {
                    u = v;
                    v = q[((ir as i32 + 1 as i32) - q_offset) as usize];
                    q[(ir - q_offset) as usize] = u + (x_0[(j_0 - 1) as usize] - arg) * v;
                    ir += 1;
                    j_0 += 1;
                }
            }
            i__3 = (l as i32 - i___0 as i32 + 1 as i32) as i32;
            j1 = (if i__3 >= 1 as i32 {
                i__3 as i32
            } else {
                1 as i32
            }) as i32;
            i__3 = (l as i32 - 1 as i32) as i32;
            i__4 = n_0 - i___0;
            j2 = (if i__3 <= i__4 {
                i__3 as i32
            } else {
                i__4 as i32
            }) as i32;
            if j1 <= j2 {
                if i___0 < m2_0 {
                    i__3 = j2;
                    j_0 = j1;
                    while j_0 <= i__3 {
                        y_0 = x_0[((i___0 + j_0) - 1) as usize];
                        u = v;
                        v = q[((ir as i32 + 1 as i32) - q_offset) as usize];
                        q[(ir - q_offset) as usize] =
                            u + (v - u) * (y_0 - arg) / (y_0 - x_0[(j_0 - 1) as usize]);
                        ir += 1;
                        j_0 += 1;
                    }
                } else {
                    i__3 = j2;
                    j_0 = j1;
                    while j_0 <= i__3 {
                        u = v;
                        v = q[((ir as i32 + 1 as i32) - q_offset) as usize];
                        q[(ir - q_offset) as usize] = (arg - x_0[(j_0 - 1) as usize]) * u
                            + (x_0[((i___0 + j_0) - 1) as usize] - arg) * v;
                        ir += 1;
                        j_0 += 1;
                    }
                }
            }
            nmip1 = (n_0 - i___0 as i32 + 1 as i32) as i32;
            if nmip1 < l {
                i__3 = (l as i32 - 1 as i32) as i32;
                j_0 = nmip1;
                while j_0 <= i__3 {
                    u = v;
                    v = q[((ir as i32 + 1 as i32) - q_offset) as usize];
                    q[(ir - q_offset) as usize] = (arg - x_0[(j_0 - 1) as usize]) * u + v;
                    ir += 1;
                    j_0 += 1;
                }
            }
            i___0 += 1;
        }
        i__2 = mm1;
        j_0 = -mm1;
        while j_0 <= i__2 {
            b[((j_0 + l * b_dim1) - b_offset) as usize] = q[(j_0 - q_offset) as usize];
            j_0 += 1;
        }
        l += 1;
    }
    i__1_0 = mm1;
    i___0 = 1 as i32;
    while i___0 <= i__1_0 {
        i__2 = mm1;
        k_0 = i___0;
        while k_0 <= i__2 {
            b[((-k_0 + i___0 * b_dim1) - b_offset) as usize] = 0.0f64 as f64;
            b[((k_0 + (n_0 + 1 as i32 - i___0) * b_dim1) - b_offset) as usize] = 0.0f64 as f64;
            k_0 += 1;
        }
        i___0 += 1;
    }
    *bl = 0.0f64 as f64;
    i__1_0 = n_0;
    i___0 = 1 as i32;
    while i___0 <= i__1_0 {
        i__2 = mm1;
        k_0 = -mm1;
        while k_0 <= i__2 {
            d__1 = b[((k_0 + i___0 * b_dim1) - b_offset) as usize];
            *bl += (if d__1 >= 0 as i32 as f64 {
                d__1 as f64
            } else {
                -(d__1 as f64)
            });
            k_0 += 1;
        }
        i___0 += 1;
    }
    *bl /= n_0 as f64;
    return 0 as i32;
}

/// Original `prep_` (`gcvspl.c:772`).
pub fn prep_(m_0: i32, n_0: i32, x_0: &[f64], w: &[f64], we: &mut [f64], el_0: &mut f64) -> i32 {
    let mut i__1_0: i32 = 0;
    let mut i__2: i32 = 0;
    let mut i__3: i32 = 0;
    let mut d__1: f64 = 0.;
    let mut f: f64 = 0.;
    let mut i___0: i32 = 0;
    let mut j_0: i32 = 0;
    let mut k_0: i32 = 0;
    let mut l: i32 = 0;
    let mut y_0: f64 = 0.;
    let mut f1: f64 = 0.;
    let mut i1: i32 = 0;
    let mut i2: i32 = 0;
    let mut m2_0: i32 = 0;
    let mut ff: f64 = 0.;
    let mut jj: i32 = 0;
    let mut jm: i32 = 0;
    let mut kl: i32 = 0;
    let mut nm: i32 = 0;
    let mut ku: i32 = 0;
    let mut wi: f64 = 0.;
    let mut n2m: i32 = 0;
    let mut mp1: i32 = 0;
    let mut i2m1: i32 = 0;
    let mut inc: i32 = 0;
    let mut i1p1: i32 = 0;
    let mut m2m1: i32 = 0;
    let mut m2p1: i32 = 0;
    m2_0 = m_0 << 1 as i32;
    mp1 = (m_0 + 1 as i32) as i32;
    m2m1 = (m2_0 as i32 - 1 as i32) as i32;
    m2p1 = (m2_0 as i32 + 1 as i32) as i32;
    nm = n_0 - m_0;
    f1 = -1.0f64 as f64;
    if m_0 != 1 as i32 {
        i__1_0 = m_0;
        i___0 = 2 as i32;
        while i___0 <= i__1_0 {
            f1 = (-(f1 as f64) * i___0 as f64) as f64;
            i___0 += 1;
        }
        i__1_0 = m2m1;
        i___0 = mp1;
        while i___0 <= i__1_0 {
            f1 *= i___0 as f64;
            i___0 += 1;
        }
    }
    i1 = 1 as i32;
    i2 = m_0;
    jm = mp1;
    i__1_0 = n_0;
    j_0 = 1 as i32;
    while j_0 <= i__1_0 {
        inc = m2p1;
        if j_0 > nm {
            f1 = -f1;
            f = f1;
        } else if j_0 < mp1 {
            inc = 1 as i32;
            f = f1;
        } else {
            f = f1 * (x_0[((j_0 + m_0) - 1) as usize] - x_0[((j_0 - m_0) - 1) as usize]);
        }
        if j_0 > mp1 {
            i1 += 1;
        }
        if i2 < n_0 {
            i2 += 1;
        }
        jj = jm;
        ff = f;
        y_0 = x_0[(i1 - 1) as usize];
        i1p1 = (i1 as i32 + 1 as i32) as i32;
        i__2 = i2;
        i___0 = i1p1;
        while i___0 <= i__2 {
            ff /= (y_0 - x_0[(i___0 - 1) as usize]) as f64;
            i___0 += 1;
        }
        we[(jj - 1) as usize] = ff;
        jj += m2_0 as i32;
        i2m1 = (i2 as i32 - 1 as i32) as i32;
        if i1p1 <= i2m1 {
            i__2 = i2m1;
            l = i1p1;
            while l <= i__2 {
                ff = f;
                y_0 = x_0[(l - 1) as usize];
                i__3 = (l as i32 - 1 as i32) as i32;
                i___0 = i1;
                while i___0 <= i__3 {
                    ff /= (y_0 - x_0[(i___0 - 1) as usize]) as f64;
                    i___0 += 1;
                }
                i__3 = i2;
                i___0 = (l as i32 + 1 as i32) as i32;
                while i___0 <= i__3 {
                    ff /= (y_0 - x_0[(i___0 - 1) as usize]) as f64;
                    i___0 += 1;
                }
                we[(jj - 1) as usize] = ff;
                jj += m2_0 as i32;
                l += 1;
            }
        }
        ff = f;
        y_0 = x_0[(i2 - 1) as usize];
        i__2 = i2m1;
        i___0 = i1;
        while i___0 <= i__2 {
            ff /= (y_0 - x_0[(i___0 - 1) as usize]) as f64;
            i___0 += 1;
        }
        we[(jj - 1) as usize] = ff;
        jj += m2_0 as i32;
        jm += inc as i32;
        j_0 += 1;
    }
    kl = 1 as i32;
    n2m = (m2p1 as i32 * n_0 + 1 as i32) as i32;
    i__1_0 = m_0;
    i___0 = 1 as i32;
    while i___0 <= i__1_0 {
        ku = kl + m_0 - i___0;
        i__2 = ku;
        k_0 = kl;
        while k_0 <= i__2 {
            we[(k_0 - 1) as usize] = 0.0f64 as f64;
            we[((n2m - k_0) - 1) as usize] = 0.0f64 as f64;
            k_0 += 1;
        }
        kl += m2p1 as i32;
        i___0 += 1;
    }
    jj = 0 as i32;
    *el_0 = 0.0f64 as f64;
    i__1_0 = n_0;
    i___0 = 1 as i32;
    while i___0 <= i__1_0 {
        wi = w[(i___0 - 1) as usize];
        i__2 = m2p1;
        j_0 = 1 as i32;
        while j_0 <= i__2 {
            jj += 1;
            we[(jj - 1) as usize] /= wi as f64;
            d__1 = we[(jj - 1) as usize];
            *el_0 += (if d__1 >= 0 as i32 as f64 {
                d__1 as f64
            } else {
                -(d__1 as f64)
            });
            j_0 += 1;
        }
        i___0 += 1;
    }
    *el_0 /= n_0 as f64;
    return 0 as i32;
}

/// Original `splc_` (`gcvspl.c:1023`).
pub fn splc_(
    m_0: i32,
    n_0: i32,
    k_0: i32,
    y_0: &[f64],
    ny_0: i32,
    wx_0: &[f64],
    wy_0: &[f64],
    mode: i32,
    val_0: f64,
    p: f64,
    eps: f64,
    c___0: &mut [f64],
    nc_0: i32,
    stat: &mut [f64],
    b: &[f64],
    we: &[f64],
    el_0: f64,
    bwe: &mut [f64],
) -> f64 {
    let mut y_dim1_0: i32 = 0;
    let mut y_offset_0: i32 = 0;
    let mut c_dim1_0: i32 = 0;
    let mut c_offset_0: i32 = 0;
    let mut b_dim1: i32 = 0;
    let mut b_offset: i32 = 0;
    let mut we_dim1: i32 = 0;
    let mut we_offset: i32 = 0;
    let mut bwe_dim1: i32 = 0;
    let mut bwe_offset: i32 = 0;
    let mut i__1_0: i32 = 0;
    let mut i__2: i32 = 0;
    let mut i__3: i32 = 0;
    let mut i__4: i32 = 0;
    let mut ret_val: f64 = 0.;
    let mut d__1: f64 = 0.;
    let mut i___0: i32 = 0;
    let mut j_0: i32 = 0;
    let mut l: i32 = 0;
    let mut dp: f64 = 0.;
    let mut km: i32 = 0;
    let mut dt: f64 = 0.;
    let mut kp: i32 = 0;
    let mut pel: f64 = 0.;
    let mut esn: f64 = 0.;
    let mut trn: f64 = 0.;
    bwe_dim1 = (m_0 - -m_0 + 1 as i32) as i32;
    bwe_offset = -m_0 + bwe_dim1;
    we_dim1 = (m_0 - -m_0 + 1 as i32) as i32;
    we_offset = -m_0 + we_dim1;
    b_dim1 = (m_0 - 1 as i32 - (1 as i32 - m_0) + 1 as i32) as i32;
    b_offset = 1 as i32 - m_0 + b_dim1;
    y_dim1_0 = ny_0;
    y_offset_0 = 1 as i32 + y_dim1_0;
    c_dim1_0 = nc_0;
    c_offset_0 = 1 as i32 + c_dim1_0;
    dp = p;
    stat[(4 as i32 - 1) as usize] = p;
    pel = p * el_0;
    if pel < eps {
        dp = eps / el_0;
        stat[(4 as i32 - 1) as usize] = 0.0f64 as f64;
    }
    if pel * eps > 1.0f64 {
        dp = 1.0f64 / (el_0 * eps);
        stat[(4 as i32 - 1) as usize] = dp;
    }
    i__1_0 = n_0;
    i___0 = 1 as i32;
    while i___0 <= i__1_0 {
        i__2 = m_0;
        i__3 = (i___0 as i32 - 1 as i32) as i32;
        km = -if i__2 <= i__3 {
            i__2 as i32
        } else {
            i__3 as i32
        } as i32;
        i__2 = m_0;
        i__3 = n_0 - i___0;
        kp = (if i__2 <= i__3 {
            i__2 as i32
        } else {
            i__3 as i32
        }) as i32;
        i__2 = kp;
        l = km;
        while l <= i__2 {
            if (if l >= 0 as i32 { l as i32 } else { -(l as i32) }) == m_0 {
                bwe[((l + i___0 * bwe_dim1) - bwe_offset) as usize] =
                    dp * we[((l + i___0 * we_dim1) - we_offset) as usize];
            } else {
                bwe[((l + i___0 * bwe_dim1) - bwe_offset) as usize] = b
                    [((l + i___0 * b_dim1) - b_offset) as usize]
                    + dp * we[((l + i___0 * we_dim1) - we_offset) as usize];
            }
            l += 1;
        }
        i___0 += 1;
    }
    bandet_(bwe, m_0, n_0);
    bansol_(bwe, y_0, ny_0, c___0, nc_0, m_0, n_0, k_0);
    stat[(3 as i32 - 1) as usize] = trinv_(we, bwe, m_0, n_0) * dp;
    trn = (stat[(3 as i32 - 1) as usize] as f64 / n_0 as f64) as f64;
    esn = 0.0f64 as f64;
    i__1_0 = k_0;
    j_0 = 1 as i32;
    while j_0 <= i__1_0 {
        i__2 = n_0;
        i___0 = 1 as i32;
        while i___0 <= i__2 {
            dt = -y_0[((i___0 + j_0 * y_dim1_0) - y_offset_0) as usize];
            i__3 = (m_0 - 1 as i32) as i32;
            i__4 = (i___0 as i32 - 1 as i32) as i32;
            km = -if i__3 <= i__4 {
                i__3 as i32
            } else {
                i__4 as i32
            } as i32;
            i__3 = (m_0 - 1 as i32) as i32;
            i__4 = n_0 - i___0;
            kp = (if i__3 <= i__4 {
                i__3 as i32
            } else {
                i__4 as i32
            }) as i32;
            i__3 = kp;
            l = km;
            while l <= i__3 {
                dt += (b[((l + i___0 * b_dim1) - b_offset) as usize]
                    * c___0[((i___0 + l + j_0 * c_dim1_0) - c_offset_0) as usize])
                    as f64;
                l += 1;
            }
            esn += (dt * dt * wx_0[(i___0 - 1) as usize] * wy_0[(j_0 - 1) as usize]) as f64;
            i___0 += 1;
        }
        j_0 += 1;
    }
    esn /= (n_0 * k_0) as f64;
    stat[(6 as i32 - 1) as usize] = esn / trn;
    stat[(1 as i32 - 1) as usize] = stat[(6 as i32 - 1) as usize] / trn;
    stat[(2 as i32 - 1) as usize] = esn;
    if (if mode >= 0 as i32 { mode } else { -mode }) != 3 as i32 {
        stat[(5 as i32 - 1) as usize] = stat[(6 as i32 - 1) as usize] - esn;
        if (if mode >= 0 as i32 { mode } else { -mode }) == 1 as i32 {
            ret_val = 0.0f64 as f64;
        }
        if (if mode >= 0 as i32 { mode } else { -mode }) == 2 as i32 {
            ret_val = stat[(1 as i32 - 1) as usize];
        }
        if (if mode >= 0 as i32 { mode } else { -mode }) == 4 as i32 {
            d__1 = stat[(3 as i32 - 1) as usize] - val_0;
            ret_val = (if d__1 >= 0 as i32 as f64 {
                d__1 as f64
            } else {
                -(d__1 as f64)
            }) as f64;
        }
    } else {
        stat[(5 as i32 - 1) as usize] =
            (esn as f64 - val_0 * (trn as f64 * 2.0f64 - 1.0f64)) as f64;
        ret_val = stat[(5 as i32 - 1) as usize];
    }
    return ret_val;
}

/// Original `bandet_` (`gcvspl.c:1210`).
pub fn bandet_(e: &mut [f64], m_1: i32, n_1: i32) -> i32 {
    let mut e_dim1: i32 = 0;
    let mut e_offset: i32 = 0;
    let mut i__1_1: i32 = 0;
    let mut i__2_0: i32 = 0;
    let mut i__3_0: i32 = 0;
    let mut i__4_0: i32 = 0;
    let mut i___1: i32 = 0;
    let mut k_1: i32 = 0;
    let mut l_0: i32 = 0;
    let mut di: f64 = 0.;
    let mut dl: f64 = 0.;
    let mut mi: i32 = 0;
    let mut km_0: i32 = 0;
    let mut lm: i32 = 0;
    let mut du: f64 = 0.;
    e_dim1 = (m_1 - -m_1 + 1 as i32) as i32;
    e_offset = -m_1 + e_dim1;
    if m_1 <= 0 as i32 {
        return 0 as i32;
    }
    i__1_1 = n_1;
    i___1 = 1 as i32;
    while i___1 <= i__1_1 {
        di = e[((i___1 * e_dim1) - e_offset) as usize];
        i__2_0 = m_1;
        i__3_0 = (i___1 as i32 - 1 as i32) as i32;
        mi = (if i__2_0 <= i__3_0 {
            i__2_0 as i32
        } else {
            i__3_0 as i32
        }) as i32;
        if mi >= 1 as i32 {
            i__2_0 = mi;
            k_1 = 1 as i32;
            while k_1 <= i__2_0 {
                di -= (e[((-k_1 + i___1 * e_dim1) - e_offset) as usize]
                    * e[((k_1 + (i___1 - k_1) * e_dim1) - e_offset) as usize])
                    as f64;
                k_1 += 1;
            }
            e[((i___1 * e_dim1) - e_offset) as usize] = di;
        }
        i__2_0 = m_1;
        i__3_0 = n_1 - i___1;
        lm = (if i__2_0 <= i__3_0 {
            i__2_0 as i32
        } else {
            i__3_0 as i32
        }) as i32;
        if lm >= 1 as i32 {
            i__2_0 = lm;
            l_0 = 1 as i32;
            while l_0 <= i__2_0 {
                dl = e[((-l_0 + (i___1 + l_0) * e_dim1) - e_offset) as usize];
                i__3_0 = m_1 - l_0;
                i__4_0 = (i___1 as i32 - 1 as i32) as i32;
                km_0 = (if i__3_0 <= i__4_0 {
                    i__3_0 as i32
                } else {
                    i__4_0 as i32
                }) as i32;
                if km_0 >= 1 as i32 {
                    du = e[((l_0 + i___1 * e_dim1) - e_offset) as usize];
                    i__3_0 = km_0;
                    k_1 = 1 as i32;
                    while k_1 <= i__3_0 {
                        du -= (e[((-k_1 + i___1 * e_dim1) - e_offset) as usize]
                            * e[((l_0 + k_1 + (i___1 - k_1) * e_dim1) - e_offset) as usize])
                            as f64;
                        dl -= (e[((-l_0 - k_1 + (l_0 + i___1) * e_dim1) - e_offset) as usize]
                            * e[((k_1 + (i___1 - k_1) * e_dim1) - e_offset) as usize])
                            as f64;
                        k_1 += 1;
                    }
                    e[((l_0 + i___1 * e_dim1) - e_offset) as usize] = du;
                }
                e[((-l_0 + (i___1 + l_0) * e_dim1) - e_offset) as usize] = dl / di;
                l_0 += 1;
            }
        }
        i___1 += 1;
    }
    return 0 as i32;
}

/// Original `bansol_` (`gcvspl.c:1328`).
pub fn bansol_(
    e: &[f64],
    y_1: &[f64],
    ny_1: i32,
    c___1: &mut [f64],
    nc_1: i32,
    m_1: i32,
    n_1: i32,
    k_1: i32,
) -> i32 {
    let mut e_dim1: i32 = 0;
    let mut e_offset: i32 = 0;
    let mut y_dim1_1: i32 = 0;
    let mut y_offset_1: i32 = 0;
    let mut c_dim1_1: i32 = 0;
    let mut c_offset_1: i32 = 0;
    let mut i__1_1: i32 = 0;
    let mut i__2_0: i32 = 0;
    let mut i__3_0: i32 = 0;
    let mut i__4_0: i32 = 0;
    let mut d__: f64 = 0.;
    let mut i___1: i32 = 0;
    let mut j_1: i32 = 0;
    let mut l_0: i32 = 0;
    let mut mi: i32 = 0;
    let mut nm1_0: i32 = 0;
    e_dim1 = (m_1 - -m_1 + 1 as i32) as i32;
    e_offset = -m_1 + e_dim1;
    c_dim1_1 = nc_1;
    c_offset_1 = 1 as i32 + c_dim1_1;
    y_dim1_1 = ny_1;
    y_offset_1 = 1 as i32 + y_dim1_1;
    nm1_0 = (n_1 - 1 as i32) as i32;
    i__1_1 = (m_1 - 1 as i32) as i32;
    if i__1_1 < 0 as i32 {
        i__1_1 = n_1;
        i___1 = 1 as i32;
        while i___1 <= i__1_1 {
            i__2_0 = k_1;
            j_1 = 1 as i32;
            while j_1 <= i__2_0 {
                c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize] = y_1
                    [((i___1 + j_1 * y_dim1_1) - y_offset_1) as usize]
                    / e[((i___1 * e_dim1) - e_offset) as usize];
                j_1 += 1;
            }
            i___1 += 1;
        }
        return 0 as i32;
    } else if i__1_1 == 0 as i32 {
        i__1_1 = k_1;
        j_1 = 1 as i32;
        while j_1 <= i__1_1 {
            c___1[((j_1 as i32 * c_dim1_1 as i32 + 1 as i32) - c_offset_1) as usize] =
                y_1[((j_1 as i32 * y_dim1_1 as i32 + 1 as i32) - y_offset_1) as usize];
            i__2_0 = n_1;
            i___1 = 2 as i32;
            while i___1 <= i__2_0 {
                c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize] = y_1
                    [((i___1 + j_1 * y_dim1_1) - y_offset_1) as usize]
                    - e[((i___1 as i32 * e_dim1 as i32 - 1 as i32) - e_offset) as usize]
                        * c___1[((i___1 - 1 as i32 + j_1 * c_dim1_1) - c_offset_1) as usize];
                i___1 += 1;
            }
            c___1[((n_1 + j_1 * c_dim1_1) - c_offset_1) as usize] /=
                e[((n_1 * e_dim1) - e_offset) as usize] as f64;
            i___1 = nm1_0;
            while i___1 >= 1 as i32 {
                c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize] = (c___1
                    [((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize]
                    - e[((i___1 as i32 * e_dim1 as i32 + 1 as i32) - e_offset) as usize]
                        * c___1[((i___1 + 1 as i32 + j_1 * c_dim1_1) - c_offset_1) as usize])
                    / e[((i___1 * e_dim1) - e_offset) as usize];
                i___1 -= 1;
            }
            j_1 += 1;
        }
        return 0 as i32;
    } else {
        i__1_1 = k_1;
        j_1 = 1 as i32;
        while j_1 <= i__1_1 {
            c___1[((j_1 as i32 * c_dim1_1 as i32 + 1 as i32) - c_offset_1) as usize] =
                y_1[((j_1 as i32 * y_dim1_1 as i32 + 1 as i32) - y_offset_1) as usize];
            i__2_0 = n_1;
            i___1 = 2 as i32;
            while i___1 <= i__2_0 {
                i__3_0 = m_1;
                i__4_0 = (i___1 as i32 - 1 as i32) as i32;
                mi = (if i__3_0 <= i__4_0 {
                    i__3_0 as i32
                } else {
                    i__4_0 as i32
                }) as i32;
                d__ = y_1[((i___1 + j_1 * y_dim1_1) - y_offset_1) as usize];
                i__3_0 = mi;
                l_0 = 1 as i32;
                while l_0 <= i__3_0 {
                    d__ -= (e[((-l_0 + i___1 * e_dim1) - e_offset) as usize]
                        * c___1[((i___1 - l_0 + j_1 * c_dim1_1) - c_offset_1) as usize])
                        as f64;
                    l_0 += 1;
                }
                c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize] = d__;
                i___1 += 1;
            }
            c___1[((n_1 + j_1 * c_dim1_1) - c_offset_1) as usize] /=
                e[((n_1 * e_dim1) - e_offset) as usize] as f64;
            i___1 = nm1_0;
            while i___1 >= 1 as i32 {
                i__2_0 = m_1;
                i__3_0 = n_1 - i___1;
                mi = (if i__2_0 <= i__3_0 {
                    i__2_0 as i32
                } else {
                    i__3_0 as i32
                }) as i32;
                d__ = c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize];
                i__2_0 = mi;
                l_0 = 1 as i32;
                while l_0 <= i__2_0 {
                    d__ -= (e[((l_0 + i___1 * e_dim1) - e_offset) as usize]
                        * c___1[((i___1 + l_0 + j_1 * c_dim1_1) - c_offset_1) as usize])
                        as f64;
                    l_0 += 1;
                }
                c___1[((i___1 + j_1 * c_dim1_1) - c_offset_1) as usize] =
                    d__ / e[((i___1 * e_dim1) - e_offset) as usize];
                i___1 -= 1;
            }
            j_1 += 1;
        }
        return 0 as i32;
    };
}

/// Original `trinv_` (`gcvspl.c:1490`).
pub fn trinv_(b_0: &[f64], e: &mut [f64], m_1: i32, n_1: i32) -> f64 {
    let mut b_dim1_0: i32 = 0;
    let mut b_offset_0: i32 = 0;
    let mut e_dim1: i32 = 0;
    let mut e_offset: i32 = 0;
    let mut i__1_1: i32 = 0;
    let mut i__2_0: i32 = 0;
    let mut i__3_0: i32 = 0;
    let mut ret_val_0: f64 = 0.;
    let mut i___1: i32 = 0;
    let mut j_1: i32 = 0;
    let mut k_1: i32 = 0;
    let mut dd: f64 = 0.;
    let mut dl: f64 = 0.;
    let mut mi: i32 = 0;
    let mut du: f64 = 0.;
    let mut mn: i32 = 0;
    let mut mp: i32 = 0;
    e_dim1 = (m_1 - -m_1 + 1 as i32) as i32;
    e_offset = -m_1 + e_dim1;
    b_dim1_0 = (m_1 - -m_1 + 1 as i32) as i32;
    b_offset_0 = -m_1 + b_dim1_0;
    e[((n_1 * e_dim1) - e_offset) as usize] = 1.0f64 / e[((n_1 * e_dim1) - e_offset) as usize];
    i___1 = (n_1 - 1 as i32) as i32;
    while i___1 >= 1 as i32 {
        i__1_1 = m_1;
        i__2_0 = n_1 - i___1;
        mi = (if i__1_1 <= i__2_0 {
            i__1_1 as i32
        } else {
            i__2_0 as i32
        }) as i32;
        dd = 1.0f64 / e[((i___1 * e_dim1) - e_offset) as usize];
        i__1_1 = mi;
        k_1 = 1 as i32;
        while k_1 <= i__1_1 {
            e[((k_1 + n_1 * e_dim1) - e_offset) as usize] =
                e[((k_1 + i___1 * e_dim1) - e_offset) as usize] * dd;
            e[((-k_1 + e_dim1) - e_offset) as usize] =
                e[((-k_1 + (k_1 + i___1) * e_dim1) - e_offset) as usize];
            k_1 += 1;
        }
        dd += dd as f64;
        j_1 = mi;
        while j_1 >= 1 as i32 {
            du = 0.0f64 as f64;
            dl = 0.0f64 as f64;
            i__1_1 = mi;
            k_1 = 1 as i32;
            while k_1 <= i__1_1 {
                du -= (e[((k_1 + n_1 * e_dim1) - e_offset) as usize]
                    * e[((j_1 - k_1 + (i___1 + k_1) * e_dim1) - e_offset) as usize])
                    as f64;
                dl -= (e[((-k_1 + e_dim1) - e_offset) as usize]
                    * e[((k_1 - j_1 + (i___1 + j_1) * e_dim1) - e_offset) as usize])
                    as f64;
                k_1 += 1;
            }
            e[((j_1 + i___1 * e_dim1) - e_offset) as usize] = du;
            e[((-j_1 + (j_1 + i___1) * e_dim1) - e_offset) as usize] = dl;
            dd -= (e[((j_1 + n_1 * e_dim1) - e_offset) as usize] * dl
                + e[((-j_1 + e_dim1) - e_offset) as usize] * du) as f64;
            j_1 -= 1;
        }
        e[((i___1 * e_dim1) - e_offset) as usize] = (dd as f64 * 0.5f64) as f64;
        i___1 -= 1;
    }
    dd = 0.0f64 as f64;
    i__1_1 = n_1;
    i___1 = 1 as i32;
    while i___1 <= i__1_1 {
        i__2_0 = m_1;
        i__3_0 = (i___1 as i32 - 1 as i32) as i32;
        mn = -if i__2_0 <= i__3_0 {
            i__2_0 as i32
        } else {
            i__3_0 as i32
        } as i32;
        i__2_0 = m_1;
        i__3_0 = n_1 - i___1;
        mp = (if i__2_0 <= i__3_0 {
            i__2_0 as i32
        } else {
            i__3_0 as i32
        }) as i32;
        i__2_0 = mp;
        k_1 = mn;
        while k_1 <= i__2_0 {
            dd += (b_0[((k_1 + i___1 * b_dim1_0) - b_offset_0) as usize]
                * e[((-k_1 + (k_1 + i___1) * e_dim1) - e_offset) as usize])
                as f64;
            k_1 += 1;
        }
        i___1 += 1;
    }
    ret_val_0 = dd;
    i__1_1 = m_1;
    k_1 = 1 as i32;
    while k_1 <= i__1_1 {
        e[((k_1 + n_1 * e_dim1) - e_offset) as usize] = 0.0f64 as f64;
        e[((-k_1 + e_dim1) - e_offset) as usize] = 0.0f64 as f64;
        k_1 += 1;
    }
    return ret_val_0;
}

/// Original `splder_` (`gcvspl.c:1647`).
pub fn splder_(
    ider: i32,
    m: i32,
    n: i32,
    t: f64,
    x: &[f64],
    c__: &[f64],
    l: &mut i32,
    q: &mut [f64],
) -> f64 {
    let mut i__1: i32 = 0;
    let mut i__2: i32 = 0;
    let mut ret_val: f64 = 0.;
    let mut i__: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut z__: f64 = 0.;
    let mut i1: i32 = 0;
    let mut j1: i32 = 0;
    let mut k1: i32 = 0;
    let mut j2: i32 = 0;
    let mut m2: i32 = 0;
    let mut ii: i32 = 0;
    let mut jj: i32 = 0;
    let mut ki: i32 = 0;
    let mut jl: i32 = 0;
    let mut lk: i32 = 0;
    let mut mi: i32 = 0;
    let mut nk: i32 = 0;
    let mut lm: i32 = 0;
    let mut ml: i32 = 0;
    let mut jm: i32 = 0;
    let mut ir: i32 = 0;
    let mut ju: i32 = 0;
    let mut tt: f64 = 0.;
    let mut lk1: i32 = 0;
    let mut mp1: i32 = 0;
    let mut m2m1: i32 = 0;
    let mut jin: i32 = 0;
    let mut nki: i32 = 0;
    let mut npm: i32 = 0;
    let mut lk1i: i32 = 0;
    let mut nki1: i32 = 0;
    let mut lk1i1: i32 = 0;
    let mut xjki: f64 = 0.;
    m2 = m << 1 as i32;
    k = m2 - ider;
    if k < 1 as i32 {
        ret_val = 0.0f64 as f64;
        return ret_val as f64;
    }
    search_(n, x, t, l);
    tt = t;
    mp1 = (m + 1 as i32) as i32;
    npm = n + m;
    m2m1 = (m2 as i32 - 1 as i32) as i32;
    k1 = (k as i32 - 1 as i32) as i32;
    nk = n - k;
    lk = *l - k;
    lk1 = (lk as i32 + 1 as i32) as i32;
    lm = *l - m;
    jl = (*l + 1 as i32) as i32;
    ju = *l + m2;
    ii = n - m2;
    ml = -*l;
    i__1 = ju;
    j = jl;
    while j <= i__1 {
        if j >= mp1 && j <= npm {
            q[((j + ml) - 1) as usize] = c__[((j - m) - 1) as usize];
        } else {
            q[((j + ml) - 1) as usize] = 0.0f64 as f64;
        }
        j += 1;
    }
    if ider > 0 as i32 {
        jl -= m2 as i32;
        ml += m2 as i32;
        i__1 = ider;
        i__ = 1 as i32;
        while i__ <= i__1 {
            jl += 1;
            ii += 1;
            j1 = (if 1 as i32 >= jl { 1 as i32 } else { jl as i32 }) as i32;
            j2 = (if *l <= ii { *l } else { ii as i32 }) as i32;
            mi = m2 - i__;
            j = (j2 as i32 + 1 as i32) as i32;
            if j1 <= j2 {
                i__2 = j2;
                jin = j1;
                while jin <= i__2 {
                    j -= 1;
                    jm = ml + j;
                    q[(jm - 1) as usize] = (q[(jm - 1) as usize]
                        - q[((jm as i32 - 1 as i32) - 1) as usize])
                        / (x[((j + mi) - 1) as usize] - x[(j - 1) as usize]);
                    jin += 1;
                }
            }
            if !(jl >= 1 as i32) {
                i1 = (i__ as i32 + 1 as i32) as i32;
                j = (ml as i32 + 1 as i32) as i32;
                if i1 <= ml {
                    i__2 = ml;
                    jin = i1;
                    while jin <= i__2 {
                        j -= 1;
                        q[(j - 1) as usize] = -q[((j as i32 - 1 as i32) - 1) as usize];
                        jin += 1;
                    }
                }
            }
            i__ += 1;
        }
        i__1 = k;
        j = 1 as i32;
        while j <= i__1 {
            q[(j - 1) as usize] = q[((j + ider) - 1) as usize];
            j += 1;
        }
    }
    if k1 >= 1 as i32 {
        i__1 = k1;
        i__ = 1 as i32;
        while i__ <= i__1 {
            nki = nk + i__;
            ir = k;
            jj = *l;
            ki = k - i__;
            nki1 = (nki as i32 + 1 as i32) as i32;
            if *l >= nki1 {
                i__2 = *l;
                j = nki1;
                while j <= i__2 {
                    q[(ir - 1) as usize] = q[((ir as i32 - 1 as i32) - 1) as usize]
                        + (tt - x[(jj - 1) as usize]) * q[(ir - 1) as usize];
                    jj -= 1;
                    ir -= 1;
                    j += 1;
                }
            }
            lk1i = lk1 + i__;
            j1 = (if 1 as i32 >= lk1i {
                1 as i32
            } else {
                lk1i as i32
            }) as i32;
            j2 = (if *l <= nki { *l } else { nki as i32 }) as i32;
            if j1 <= j2 {
                i__2 = j2;
                j = j1;
                while j <= i__2 {
                    xjki = x[((jj + ki) - 1) as usize];
                    z__ = q[(ir - 1) as usize];
                    q[(ir - 1) as usize] = z__
                        + (xjki - tt) * (q[((ir as i32 - 1 as i32) - 1) as usize] - z__)
                            / (xjki - x[(jj - 1) as usize]);
                    ir -= 1;
                    jj -= 1;
                    j += 1;
                }
            }
            if lk1i <= 0 as i32 {
                jj = ki;
                lk1i1 = 1 as i32 - lk1i;
                i__2 = lk1i1;
                j = 1 as i32;
                while j <= i__2 {
                    q[(ir - 1) as usize] += ((x[(jj - 1) as usize] - tt)
                        * q[((ir as i32 - 1 as i32) - 1) as usize])
                        as f64;
                    jj -= 1;
                    ir -= 1;
                    j += 1;
                }
            }
            i__ += 1;
        }
    }
    z__ = q[(k - 1) as usize];
    if ider > 0 as i32 {
        i__1 = m2m1;
        j = k;
        while j <= i__1 {
            z__ *= j as f64;
            j += 1;
        }
    }
    ret_val = z__;
    return ret_val as f64;
}

/// Original `search_` (`gcvspl.c:1877`).
pub fn search_(n_0: i32, x_0: &[f64], t_0: f64, l_0: &mut i32) -> i32 {
    let mut current_block: u64;
    let mut il: i32 = 0;
    let mut iu: i32 = 0;
    if t_0 < x_0[(1 as i32 - 1) as usize] {
        *l_0 = 0 as i32;
        return 0 as i32;
    }
    if t_0 >= x_0[(n_0 - 1) as usize] {
        *l_0 = n_0;
        return 0 as i32;
    }
    *l_0 = (if *l_0 >= 1 as i32 { *l_0 } else { 1 as i32 }) as i32;
    if *l_0 >= n_0 {
        *l_0 = (n_0 - 1 as i32) as i32;
    }
    if t_0 >= x_0[(*l_0 - 1) as usize] {
        if t_0 < x_0[((*l_0 + 1 as i32) - 1) as usize] {
            return 0 as i32;
        }
        *l_0 += 1;
        if t_0 < x_0[((*l_0 + 1 as i32) - 1) as usize] {
            return 0 as i32;
        }
        il = (*l_0 + 1 as i32) as i32;
        iu = n_0;
        current_block = 3202994239531656598;
    } else {
        *l_0 -= 1;
        if t_0 >= x_0[(*l_0 - 1) as usize] {
            return 0 as i32;
        }
        il = 1 as i32;
        current_block = 13523560774170872681;
    }
    loop {
        match current_block {
            13523560774170872681 => {
                iu = *l_0;
                current_block = 3202994239531656598;
            }
            _ => {
                *l_0 = ((il as i32 + iu as i32) / 2 as i32) as i32;
                if iu - il <= 1 as i32 {
                    return 0 as i32;
                }
                if t_0 < x_0[(*l_0 - 1) as usize] {
                    current_block = 13523560774170872681;
                    continue;
                }
                il = *l_0;
                current_block = 3202994239531656598;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_order_one_interpolation_round_trips_through_splder() {
        let x = [0_f64, 1.];
        let y = [2_f64, 4.];
        let wx = [1_f64, 1.];
        let wy = [1_f64];
        let mut coefficients = [0_f64; 2];
        let mut work = [0_f64; 20];
        let mut error = -1;
        assert_eq!(
            gcvspl(
                &x,
                &y,
                2,
                &wx,
                &wy,
                1,
                2,
                1,
                1,
                0.,
                &mut coefficients,
                2,
                &mut work,
                &mut error
            ),
            0
        );
        assert_eq!(error, 0);
        let mut near = 0;
        let mut derivative_work = [0_f64; 2];
        let value = splder(
            0,
            1,
            2,
            0.25,
            &x,
            &coefficients,
            &mut near,
            &mut derivative_work,
        );
        assert!(
            (value - 2.5).abs() < 1.0e-12,
            "value {value}, coefficients {coefficients:?}"
        );
    }

    #[test]
    fn invalid_order_sets_source_error_one() {
        let x = [0_f64, 1.];
        let y = [0_f64; 2];
        let wx = [1_f64; 2];
        let wy = [1_f64];
        let mut c = [0_f64; 2];
        let mut work = [0_f64; 20];
        let mut error = 0;
        gcvspl(
            &x, &y, 2, &wx, &wy, 0, 2, 1, 1, 0., &mut c, 2, &mut work, &mut error,
        );
        assert_eq!(error, 1);
    }
}
