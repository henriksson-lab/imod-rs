//! Translation of `IMOD/raptor/suitesparse/cs_amd.c`.
//!
//! The workspace `W` (eight arrays of `n+1` ints: `len`, `nv`, `next`,
//! `head`, `elen`, `degree`, `w`, `hhead`) is eight `Vec`s; `last` uses the
//! result array `P` as workspace, as in C.

use super::cs::{Cs, cs_csc, cs_flip, cs_max, cs_min};
use super::cs_add::cs_add;
use super::cs_fkeep::cs_fkeep;
use super::cs_malloc::cs_malloc;
use super::cs_multiply::cs_multiply;
use super::cs_tdfs::cs_tdfs;
use super::cs_transpose::cs_transpose;
use super::cs_util::{cs_idone, cs_sprealloc};

/// `cs_wclear(mark, lemax, w, n)` (static): clear w.
fn cs_wclear(mut mark: i32, lemax: i32, w: &mut [i32], n: i32) -> i32 {
    if mark < 2 || mark.wrapping_add(lemax) < 0 {
        for k in 0..n as usize {
            if w[k] != 0 {
                w[k] = 1;
            }
        }
        mark = 2;
    }
    mark // at this point, w [0..n-1] < mark holds
}

/// `cs_diag(i, j, aij, other)` (static): keep off-diagonal entries; drop
/// diagonal entries.
fn cs_diag(i: i32, j: i32, _aij: f64) -> i32 {
    (i != j) as i32
}

/// `cs_amd(order, A)`: p = amd(A+A') if symmetric is true, or amd(A'A)
/// otherwise.  order 0:natural, 1:Chol, 2:LU, 3:QR.
pub fn cs_amd(order: i32, a: &Cs) -> Option<Vec<i32>> {
    let mut lemax = 0i32;
    let mut mindeg = 0i32;
    let mut nel = 0i32;
    // --- Construct matrix C ---
    if !cs_csc(a) || order <= 0 || order > 3 {
        return None;
    }
    let mut at = cs_transpose(a, 0)?; // compute A'
    let m = a.m;
    let n = a.n;
    // `CS_MAX (16, 10 * sqrt ((double) n))` is a double, truncated on
    // assignment to the int `dense`.
    let dense_d = if 16.0 > 10.0 * (n as f64).sqrt() {
        16.0
    } else {
        10.0 * (n as f64).sqrt()
    };
    let mut dense = dense_d as i32; // find dense threshold
    dense = cs_min(n - 2, dense);
    let c = if order == 1 && n == m {
        cs_add(a, &at, 0.0, 0.0) // C = A+A'
    } else if order == 2 {
        // drop dense columns from AT
        let mut p2 = 0i32;
        for j in 0..m as usize {
            let mut p = at.p[j]; // column j of AT starts here
            at.p[j] = p2; // new column j starts here
            if at.p[j + 1] - p > dense {
                continue; // skip dense col j
            }
            while p < at.p[j + 1] {
                at.i[p2 as usize] = at.i[p as usize];
                p2 += 1;
                p += 1;
            }
        }
        at.p[m as usize] = p2; // finalize AT
        let a2 = cs_transpose(&at, 0); // A2 = AT'
        match a2 {
            Some(a2) => cs_multiply(&at, &a2), // C=A'*A with no dense rows
            None => None,
        }
    } else {
        cs_multiply(&at, a) // C=A'*A
    };
    drop(at);
    let mut c = c?;
    cs_fkeep(&mut c, &cs_diag); // drop diagonal entries
    let mut cnz = c.p[n as usize];
    let mut pp: Vec<i32> = cs_malloc(n + 1); // allocate result
    let nu1 = (n + 1) as usize;
    let mut len: Vec<i32> = vec![0; nu1];
    let mut nv: Vec<i32> = vec![0; nu1];
    let mut next: Vec<i32> = vec![0; nu1];
    let mut head: Vec<i32> = vec![0; nu1];
    let mut elen: Vec<i32> = vec![0; nu1];
    let mut degree: Vec<i32> = vec![0; nu1];
    let mut w: Vec<i32> = vec![0; nu1];
    let mut hhead: Vec<i32> = vec![0; nu1];
    let t = cnz + cnz / 5 + 2 * n; // add elbow room to C
    if cs_sprealloc(&mut c, t) == 0 {
        return cs_idone(pp, 0);
    }
    let last = &mut pp; // use P as workspace for last
    // --- Initialize quotient graph ---
    for k in 0..n as usize {
        len[k] = c.p[k + 1] - c.p[k];
    }
    len[n as usize] = 0;
    let nzmax = c.nzmax;
    let cp = &mut c.p;
    let ci = &mut c.i;
    for i in 0..=n as usize {
        head[i] = -1; // degree list i is empty
        last[i] = -1;
        next[i] = -1;
        hhead[i] = -1; // hash list i is empty
        nv[i] = 1; // node i is just one node
        w[i] = 1; // node i is alive
        elen[i] = 0; // Ek of node i is empty
        degree[i] = len[i]; // degree of node i
    }
    let mut mark = cs_wclear(0, 0, &mut w, n); // clear w
    elen[n as usize] = -2; // n is a dead element
    cp[n as usize] = -1; // n is a root of assembly tree
    w[n as usize] = 0; // n is a dead element
    // --- Initialize degree lists ---
    for i in 0..n {
        let iu = i as usize;
        let d = degree[iu];
        if d == 0 {
            // node i is empty
            elen[iu] = -2; // element i is dead
            nel += 1;
            cp[iu] = -1; // i is a root of assembly tree
            w[iu] = 0;
        } else if d > dense {
            // node i is dense
            nv[iu] = 0; // absorb i into element n
            elen[iu] = -1; // node i is dead
            nel += 1;
            cp[iu] = cs_flip(n);
            nv[n as usize] += 1;
        } else {
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = i;
            }
            next[iu] = head[d as usize]; // put node i in degree list d
            head[d as usize] = i;
        }
    }
    while nel < n {
        // while (selecting pivots) do
        // --- Select node of minimum approximate degree ---
        let mut k = -1i32;
        while mindeg < n && {
            k = head[mindeg as usize];
            k == -1
        } {
            mindeg += 1;
        }
        let ku = k as usize;
        if next[ku] != -1 {
            last[next[ku] as usize] = -1;
        }
        head[mindeg as usize] = next[ku]; // remove k from degree list
        let elenk = elen[ku]; // elenk = |Ek|
        let mut nvk = nv[ku]; // # of nodes k represents
        nel += nvk; // nv[k] nodes of A eliminated
        // --- Garbage collection ---
        if elenk > 0 && cnz + mindeg >= nzmax {
            for j in 0..n as usize {
                let p = cp[j];
                if p >= 0 {
                    // j is a live node or element
                    cp[j] = ci[p as usize]; // save first entry of object
                    ci[p as usize] = cs_flip(j as i32); // first entry is now CS_FLIP(j)
                }
            }
            let mut q = 0i32;
            let mut p = 0i32;
            while p < cnz {
                // scan all of memory
                let j = cs_flip(ci[p as usize]);
                p += 1;
                if j >= 0 {
                    // found object j
                    ci[q as usize] = cp[j as usize]; // restore first entry of object
                    cp[j as usize] = q; // new pointer to object j
                    q += 1;
                    for _k3 in 0..len[j as usize] - 1 {
                        ci[q as usize] = ci[p as usize];
                        q += 1;
                        p += 1;
                    }
                }
            }
            cnz = q; // Ci [cnz...nzmax-1] now free
        }
        // --- Construct new element ---
        let mut dk = 0i32;
        nv[ku] = -nvk; // flag k as in Lk
        let mut p = cp[ku];
        let pk1 = if elenk == 0 { p } else { cnz }; // do in place if elen[k] == 0
        let mut pk2 = pk1;
        for k1 in 1..=elenk + 1 {
            let e;
            let mut pj;
            let ln;
            if k1 > elenk {
                e = k; // search the nodes in k
                pj = p; // list of nodes starts at Ci[pj]
                ln = len[ku] - elenk; // length of list of nodes in k
            } else {
                e = ci[p as usize]; // search the nodes in e
                p += 1;
                pj = cp[e as usize];
                ln = len[e as usize]; // length of list of nodes in e
            }
            for _k2 in 1..=ln {
                let i = ci[pj as usize];
                pj += 1;
                let iu = i as usize;
                let nvi = nv[iu];
                if nvi <= 0 {
                    continue; // node i dead, or seen
                }
                dk += nvi; // degree[Lk] += size of node i
                nv[iu] = -nvi; // negate nv[i] to denote i in Lk
                ci[pk2 as usize] = i; // place i in Lk
                pk2 += 1;
                if next[iu] != -1 {
                    last[next[iu] as usize] = last[iu];
                }
                if last[iu] != -1 {
                    // remove i from degree list
                    next[last[iu] as usize] = next[iu];
                } else {
                    head[degree[iu] as usize] = next[iu];
                }
            }
            if e != k {
                cp[e as usize] = cs_flip(k); // absorb e into k
                w[e as usize] = 0; // e is now a dead element
            }
        }
        if elenk != 0 {
            cnz = pk2; // Ci [cnz...nzmax] is free
        }
        degree[ku] = dk; // external degree of k - |Lk\i|
        cp[ku] = pk1; // element k is in Ci[pk1..pk2-1]
        len[ku] = pk2 - pk1;
        elen[ku] = -2; // k is now an element
        // --- Find set differences ---
        mark = cs_wclear(mark, lemax, &mut w, n); // clear w if necessary
        for pk in pk1..pk2 {
            // scan 1: find |Le\Lk|
            let i = ci[pk as usize] as usize;
            let eln = elen[i];
            if eln <= 0 {
                continue; // skip if elen[i] empty
            }
            let nvi = -nv[i]; // nv [i] was negated
            let wnvi = mark - nvi;
            let mut p = cp[i];
            while p <= cp[i] + eln - 1 {
                // scan Ei
                let e = ci[p as usize] as usize;
                if w[e] >= mark {
                    w[e] -= nvi; // decrement |Le\Lk|
                } else if w[e] != 0 {
                    // ensure e is a live element
                    w[e] = degree[e] + wnvi; // 1st time e seen in scan 1
                }
                p += 1;
            }
        }
        // --- Degree update ---
        for pk in pk1..pk2 {
            // scan2: degree update
            let i = ci[pk as usize] as usize; // consider node i in Lk
            let p1 = cp[i];
            let p2 = p1 + elen[i] - 1;
            let mut pn = p1;
            let mut h: u32 = 0;
            let mut d = 0i32;
            let mut p = p1;
            while p <= p2 {
                // scan Ei
                let e = ci[p as usize];
                if w[e as usize] != 0 {
                    // e is an unabsorbed element
                    let dext = w[e as usize] - mark; // dext = |Le\Lk|
                    if dext > 0 {
                        d += dext; // sum up the set differences
                        ci[pn as usize] = e; // keep e in Ei
                        pn += 1;
                        h = h.wrapping_add(e as u32); // compute the hash of node i
                    } else {
                        cp[e as usize] = cs_flip(k); // aggressive absorb. e->k
                        w[e as usize] = 0; // e is a dead element
                    }
                }
                p += 1;
            }
            elen[i] = pn - p1 + 1; // elen[i] = |Ei|
            let p3 = pn;
            let p4 = p1 + len[i];
            let mut p = p2 + 1;
            while p < p4 {
                // prune edges in Ai
                let j = ci[p as usize];
                p += 1;
                let nvj = nv[j as usize];
                if nvj <= 0 {
                    continue; // node j dead or in Lk
                }
                d += nvj; // degree(i) += |j|
                ci[pn as usize] = j; // place j in node list of i
                pn += 1;
                h = h.wrapping_add(j as u32); // compute hash for node i
            }
            if d == 0 {
                // check for mass elimination
                cp[i] = cs_flip(k); // absorb i into k
                let nvi = -nv[i];
                dk -= nvi; // |Lk| -= |i|
                nvk += nvi; // |k| += nv[i]
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1; // node i is dead
            } else {
                degree[i] = cs_min(degree[i], d); // update degree(i)
                ci[pn as usize] = ci[p3 as usize]; // move first node to end
                ci[p3 as usize] = ci[p1 as usize]; // move 1st el. to end of Ei
                ci[p1 as usize] = k; // add k as 1st element in of Ei
                len[i] = pn - p1 + 1; // new len of adj. list of node i
                h %= n as u32; // finalize hash of i
                next[i] = hhead[h as usize]; // place i in hash bucket
                hhead[h as usize] = i as i32;
                last[i] = h as i32; // save hash of i in last[i]
            }
        } // scan2 is done
        degree[ku] = dk; // finalize |Lk|
        lemax = cs_max(lemax, dk);
        mark = cs_wclear(mark + lemax, lemax, &mut w, n); // clear w
        // --- Supernode detection ---
        for pk in pk1..pk2 {
            let i = ci[pk as usize];
            if nv[i as usize] >= 0 {
                continue; // skip if i is dead
            }
            let h = last[i as usize] as u32; // scan hash bucket of node i
            let mut i = hhead[h as usize];
            hhead[h as usize] = -1; // hash bucket will be empty
            while i != -1 && next[i as usize] != -1 {
                let iu = i as usize;
                let ln = len[iu];
                let eln = elen[iu];
                let mut p = cp[iu] + 1;
                while p <= cp[iu] + ln - 1 {
                    w[ci[p as usize] as usize] = mark;
                    p += 1;
                }
                let mut jlast = i;
                let mut j = next[iu];
                while j != -1 {
                    // compare i with all j
                    let ju = j as usize;
                    let mut ok = (len[ju] == ln) && (elen[ju] == eln);
                    let mut p = cp[ju] + 1;
                    while ok && p <= cp[ju] + ln - 1 {
                        if w[ci[p as usize] as usize] != mark {
                            ok = false; // compare i and j
                        }
                        p += 1;
                    }
                    if ok {
                        // i and j are identical
                        cp[ju] = cs_flip(i); // absorb j into i
                        nv[iu] += nv[ju];
                        nv[ju] = 0;
                        elen[ju] = -1; // node j is dead
                        j = next[ju]; // delete j from hash bucket
                        next[jlast as usize] = j;
                    } else {
                        jlast = j; // j and i are different
                        j = next[ju];
                    }
                }
                i = next[iu];
                mark += 1;
            }
        }
        // --- Finalize new element ---
        let mut p = pk1;
        for pk in pk1..pk2 {
            // finalize Lk
            let i = ci[pk as usize];
            let iu = i as usize;
            let nvi = -nv[iu];
            if nvi <= 0 {
                continue; // skip if i is dead
            }
            nv[iu] = nvi; // restore nv[i]
            let mut d = degree[iu] + dk - nvi; // compute external degree(i)
            d = cs_min(d, n - nel - nvi);
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = i;
            }
            next[iu] = head[d as usize]; // put i back in degree list
            last[iu] = -1;
            head[d as usize] = i;
            mindeg = cs_min(mindeg, d); // find new minimum degree
            degree[iu] = d;
            ci[p as usize] = i; // place i in Lk
            p += 1;
        }
        nv[ku] = nvk; // # nodes absorbed into k
        len[ku] = p - pk1;
        if len[ku] == 0 {
            // length of adj list of element k
            cp[ku] = -1; // k is a root of the tree
            w[ku] = 0; // k is now a dead element
        }
        if elenk != 0 {
            cnz = p; // free unused space in Lk
        }
    }
    // --- Postordering ---
    for i in 0..n as usize {
        cp[i] = cs_flip(cp[i]); // fix assembly tree
    }
    for j in 0..=n as usize {
        head[j] = -1;
    }
    let mut j = n;
    while j >= 0 {
        // place unordered nodes in lists
        let ju = j as usize;
        if nv[ju] <= 0 {
            next[ju] = head[cp[ju] as usize]; // place j in list of its parent
            head[cp[ju] as usize] = j;
        }
        j -= 1;
    }
    let mut e = n;
    while e >= 0 {
        // place elements in lists
        let eu = e as usize;
        if nv[eu] > 0 && cp[eu] != -1 {
            next[eu] = head[cp[eu] as usize]; // place e in list of its parent
            head[cp[eu] as usize] = e;
        }
        e -= 1;
    }
    let mut k = 0i32;
    for i in 0..=n {
        // postorder the assembly tree
        if cp[i as usize] == -1 {
            k = cs_tdfs(i, k, &mut head, &next, last, &mut w);
        }
    }
    cs_idone(pp, 1)
}
