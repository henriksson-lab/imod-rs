//! Translation of `IMOD/raptor/suitesparse/cs_chol.c`.

use super::cs::{Cs, Csn, Css, cs_csc};
use super::cs_ereach::cs_ereach;
use super::cs_malloc::cs_malloc;
use super::cs_symperm::cs_symperm;
use super::cs_util::{cs_ndone, cs_spalloc};

/// `cs_chol(A, S)`: L = chol (A, [pinv parent cp]), pinv is optional.
pub fn cs_chol(a: &Cs, s: &Css) -> Option<Csn> {
    if !cs_csc(a) {
        return None;
    }
    let (cp, parent) = match (s.cp.as_ref(), s.parent.as_ref()) {
        (Some(cp), Some(parent)) => (cp, parent),
        _ => return None,
    };
    let n = a.n;
    let nu = n as usize;
    let mut nn = Csn::default();
    // c = [c (n) | s (n)]
    let mut cw: Vec<i32> = cs_malloc(2 * n);
    let mut x: Vec<f64> = cs_malloc(n);
    let pinv = s.pinv.as_deref();
    // C is a copy E=A(p,p), or an alias for A
    let e;
    let c: &Cs = match pinv {
        Some(_) => {
            e = cs_symperm(a, pinv, 1)?;
            &e
        }
        None => a,
    };
    let (c_, s_) = cw.split_at_mut(nu);
    let cpp = &c.p;
    let ci = &c.i;
    let cx = c.x.as_ref().expect("cs_chol values");
    let mut l = cs_spalloc(n, n, cp[nu], 1, 0); // allocate result
    for k in 0..nu {
        c_[k] = cp[k];
        l.p[k] = c_[k];
    }
    for k in 0..n {
        let ku = k as usize;
        // --- Nonzero pattern of L(k,:) ---
        let mut top = cs_ereach(c, k, parent, s_, c_); // find pattern of L(k,:)
        x[ku] = 0.0; // x (0:k) is now zero
        for p in cpp[ku]..cpp[ku + 1] {
            // x = full(triu(C(:,k)))
            let p = p as usize;
            if ci[p] <= k {
                x[ci[p] as usize] = cx[p];
            }
        }
        let mut d = x[ku]; // d = C(k,k)
        x[ku] = 0.0; // clear x for k+1st iteration
        // --- Triangular solve ---
        let lx = l.x.as_mut().unwrap();
        while top < n {
            // solve L(0:k-1,0:k-1) * x = C(:,k)
            let i = s_[top as usize] as usize; // s [top..n-1] is pattern of L(k,:)
            let lki = x[i] / lx[l.p[i] as usize]; // L(k,i) = x (i) / L(i,i)
            x[i] = 0.0; // clear x for k+1st iteration
            let (lo, hi) = ((l.p[i] + 1) as usize, c_[i] as usize);
            for (&li, &lxp) in l.i[lo..hi].iter().zip(&lx[lo..hi]) {
                debug_assert!((li as u32) < n as u32);
                // SAFETY: `li` is a row index this function stored in L
                // (`l.i[p] = k` with `0 <= k < n`), and `x.len() == n`
                // (`cs_malloc(n)` above).
                unsafe { *x.get_unchecked_mut(li as usize) -= lxp * lki };
            }
            d -= lki * lki; // d = d - L(k,i)*L(k,i)
            let p = c_[i] as usize;
            c_[i] += 1;
            l.i[p] = k; // store L(k,i) in column i
            lx[p] = lki;
            top += 1;
        }
        // --- Compute L(k,k) ---
        if d <= 0.0 {
            nn.l = Some(l);
            return cs_ndone(nn, 0); // not pos def
        }
        let p = c_[ku] as usize;
        c_[ku] += 1;
        l.i[p] = k; // store L(k,k) = sqrt (d) in column k
        lx[p] = d.sqrt();
    }
    l.p[nu] = cp[nu]; // finalize L
    nn.l = Some(l);
    cs_ndone(nn, 1) // success: free E,s,x; return N
}
