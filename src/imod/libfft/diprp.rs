//! c2rust parity baseline of `IMOD/libfft/diprp.c`.
#![allow(non_snake_case, unused_mut, unsafe_op_in_unsafe_fn)]

pub unsafe extern "C" fn diprp(
    mut pts: ::core::ffi::c_int,
    mut sym: *mut ::core::ffi::c_int,
    mut psym: ::core::ffi::c_int,
    mut unsym: *mut ::core::ffi::c_int,
    mut dim: *mut ::core::ffi::c_int,
    mut x: *mut ::core::ffi::c_float,
    mut y: *mut ::core::ffi::c_float,
) {
    let mut t: ::core::ffi::c_float = 0.;
    let mut onemod: ::core::ffi::c_int = 0;
    let mut modulo: [::core::ffi::c_int; 15] = [0; 15];
    let mut dk: ::core::ffi::c_int = 0;
    let mut jj: ::core::ffi::c_int = 0;
    let mut kk: ::core::ffi::c_int = 0;
    let mut lk: ::core::ffi::c_int = 0;
    let mut mods: ::core::ffi::c_int = 0;
    let mut mult: ::core::ffi::c_int = 0;
    let mut nest: ::core::ffi::c_int = 0;
    let mut punsym: ::core::ffi::c_int = 0;
    let mut test: ::core::ffi::c_int = 0;
    let mut nt: ::core::ffi::c_int = 0;
    let mut sep: ::core::ffi::c_int = 0;
    let mut delta: ::core::ffi::c_int = 0;
    let mut p: ::core::ffi::c_int = 0;
    let mut p0: ::core::ffi::c_int = 0;
    let mut p1: ::core::ffi::c_int = 0;
    let mut p2: ::core::ffi::c_int = 0;
    let mut p3: ::core::ffi::c_int = 0;
    let mut p4: ::core::ffi::c_int = 0;
    let mut p5: ::core::ffi::c_int = 0;
    let mut size: ::core::ffi::c_int = 0;
    let mut s: [::core::ffi::c_int; 15] = [0; 15];
    let mut u: [::core::ffi::c_int; 15] = [0; 15];
    let mut a: ::core::ffi::c_int = 0;
    let mut b: ::core::ffi::c_int = 0;
    let mut c: ::core::ffi::c_int = 0;
    let mut d: ::core::ffi::c_int = 0;
    let mut e: ::core::ffi::c_int = 0;
    let mut f: ::core::ffi::c_int = 0;
    let mut g: ::core::ffi::c_int = 0;
    let mut h: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut k: ::core::ffi::c_int = 0;
    let mut l: ::core::ffi::c_int = 0;
    let mut m: ::core::ffi::c_int = 0;
    let mut n: ::core::ffi::c_int = 0;
    nest = 14 as ::core::ffi::c_int;
    nt = *dim.offset(1 as ::core::ffi::c_int as isize);
    sep = *dim.offset(2 as ::core::ffi::c_int as isize);
    p2 = *dim.offset(3 as ::core::ffi::c_int as isize);
    size = *dim.offset(4 as ::core::ffi::c_int as isize) - 1 as ::core::ffi::c_int;
    p4 = *dim.offset(5 as ::core::ffi::c_int as isize);
    if *sym.offset(1 as ::core::ffi::c_int as isize) != 0 as ::core::ffi::c_int {
        j = 1 as ::core::ffi::c_int;
        while j <= nest {
            u[j as usize] = 1 as ::core::ffi::c_int;
            s[j as usize] = 1 as ::core::ffi::c_int;
            j += 1;
        }
        n = pts;
        j = 1 as ::core::ffi::c_int;
        while j <= nest {
            if *sym.offset(j as isize) == 0 as ::core::ffi::c_int {
                break;
            }
            jj = nest + 1 as ::core::ffi::c_int - j;
            u[jj as usize] = n;
            s[jj as usize] = n / *sym.offset(j as isize);
            n = n / *sym.offset(j as isize);
            j += 1;
        }
        jj = 0 as ::core::ffi::c_int;
        a = 1 as ::core::ffi::c_int;
        while a <= u[1 as ::core::ffi::c_int as usize] {
            b = a;
            while b <= u[2 as ::core::ffi::c_int as usize] {
                c = b;
                while c <= u[3 as ::core::ffi::c_int as usize] {
                    d = c;
                    while d <= u[4 as ::core::ffi::c_int as usize] {
                        e = d;
                        while e <= u[5 as ::core::ffi::c_int as usize] {
                            f = e;
                            while f <= u[6 as ::core::ffi::c_int as usize] {
                                g = f;
                                while g <= u[7 as ::core::ffi::c_int as usize] {
                                    h = g;
                                    while h <= u[8 as ::core::ffi::c_int as usize] {
                                        i = h;
                                        while i <= u[9 as ::core::ffi::c_int as usize] {
                                            j = i;
                                            while j <= u[10 as ::core::ffi::c_int as usize] {
                                                k = j;
                                                while k <= u[11 as ::core::ffi::c_int as usize] {
                                                    l = k;
                                                    while l <= u[12 as ::core::ffi::c_int as usize]
                                                    {
                                                        m = l;
                                                        while m
                                                            <= u[13 as ::core::ffi::c_int as usize]
                                                        {
                                                            n = m;
                                                            while n
                                                                <= u[14 as ::core::ffi::c_int
                                                                    as usize]
                                                            {
                                                                jj = jj + 1 as ::core::ffi::c_int;
                                                                if !(jj >= n) {
                                                                    delta = (n - jj) * sep;
                                                                    p1 = (jj
                                                                        - 1 as ::core::ffi::c_int)
                                                                        * sep
                                                                        + 1 as ::core::ffi::c_int;
                                                                    p0 = p1;
                                                                    while p0 <= nt {
                                                                        p3 = p0 + size;
                                                                        p = p0 - 1
                                                                            as ::core::ffi::c_int;
                                                                        while p < p3 {
                                                                            p5 = p + delta;
                                                                            t = *x
                                                                                .offset(p as isize);
                                                                            *x.offset(p as isize) =
                                                                                *x.offset(
                                                                                    p5 as isize,
                                                                                );
                                                                            *x.offset(
                                                                                p5 as isize,
                                                                            ) = t;
                                                                            t = *y
                                                                                .offset(p as isize);
                                                                            *y.offset(p as isize) =
                                                                                *y.offset(
                                                                                    p5 as isize,
                                                                                );
                                                                            *y.offset(
                                                                                p5 as isize,
                                                                            ) = t;
                                                                            p += p4;
                                                                        }
                                                                        p0 += p2;
                                                                    }
                                                                }
                                                                n += s[14 as ::core::ffi::c_int
                                                                    as usize];
                                                            }
                                                            m += s
                                                                [13 as ::core::ffi::c_int as usize];
                                                        }
                                                        l += s[12 as ::core::ffi::c_int as usize];
                                                    }
                                                    k += s[11 as ::core::ffi::c_int as usize];
                                                }
                                                j += s[10 as ::core::ffi::c_int as usize];
                                            }
                                            i += s[9 as ::core::ffi::c_int as usize];
                                        }
                                        h += s[8 as ::core::ffi::c_int as usize];
                                    }
                                    g += s[7 as ::core::ffi::c_int as usize];
                                }
                                f += s[6 as ::core::ffi::c_int as usize];
                            }
                            e += s[5 as ::core::ffi::c_int as usize];
                        }
                        d += s[4 as ::core::ffi::c_int as usize];
                    }
                    c += s[3 as ::core::ffi::c_int as usize];
                }
                b += s[2 as ::core::ffi::c_int as usize];
            }
            a += 1;
        }
    }
    if *unsym.offset(1 as ::core::ffi::c_int as isize) == 0 as ::core::ffi::c_int {
        return;
    }
    punsym = pts / (psym * psym);
    mult = punsym / *unsym.offset(1 as ::core::ffi::c_int as isize);
    test = (*unsym.offset(1 as ::core::ffi::c_int as isize)
        * *unsym.offset(2 as ::core::ffi::c_int as isize)
        - 1 as ::core::ffi::c_int)
        * mult
        * psym;
    lk = mult;
    dk = mult;
    k = 2 as ::core::ffi::c_int;
    while k <= nest {
        if *unsym.offset(k as isize) == 0 as ::core::ffi::c_int {
            break;
        }
        lk = lk * *unsym.offset((k - 1 as ::core::ffi::c_int) as isize);
        dk = dk / *unsym.offset(k as isize);
        u[k as usize] = (lk - dk) * psym;
        mods = k;
        k += 1;
    }
    onemod = (mods < 3 as ::core::ffi::c_int) as ::core::ffi::c_int;
    if onemod == 0 {
        j = 3 as ::core::ffi::c_int;
        while j <= mods {
            jj = mods + 3 as ::core::ffi::c_int - j;
            modulo[jj as usize] = u[j as usize];
            j += 1;
        }
    }
    modulo[2 as ::core::ffi::c_int as usize] = u[2 as ::core::ffi::c_int as usize];
    u[10 as ::core::ffi::c_int as usize] = (punsym - 3 as ::core::ffi::c_int) * psym;
    s[13 as ::core::ffi::c_int as usize] = punsym * psym;
    j = psym;
    while j <= u[10 as ::core::ffi::c_int as usize] {
        k = j;
        loop {
            k = k * mult;
            if onemod == 0 {
                i = 3 as ::core::ffi::c_int;
                while i <= mods {
                    k = k - k / modulo[i as usize] * modulo[i as usize];
                    i += 1;
                }
            }
            if k < test {
                k = k - k / modulo[2 as ::core::ffi::c_int as usize]
                    * modulo[2 as ::core::ffi::c_int as usize];
            } else {
                k = k - k / modulo[2 as ::core::ffi::c_int as usize]
                    * modulo[2 as ::core::ffi::c_int as usize]
                    + modulo[2 as ::core::ffi::c_int as usize];
            }
            if !(k < j) {
                break;
            }
        }
        if k != j {
            delta = (k - j) * sep;
            l = 1 as ::core::ffi::c_int;
            while l <= psym {
                m = l;
                while m <= pts {
                    p1 = (m + j - 1 as ::core::ffi::c_int) * sep + 1 as ::core::ffi::c_int;
                    p0 = p1;
                    while p0 <= nt {
                        p3 = p0 + size;
                        jj = p0 - 1 as ::core::ffi::c_int;
                        while jj < p3 {
                            kk = jj + delta;
                            t = *x.offset(jj as isize);
                            *x.offset(jj as isize) = *x.offset(kk as isize);
                            *x.offset(kk as isize) = t;
                            t = *y.offset(jj as isize);
                            *y.offset(jj as isize) = *y.offset(kk as isize);
                            *y.offset(kk as isize) = t;
                            jj += p4;
                        }
                        p0 += p2;
                    }
                    m += s[13 as ::core::ffi::c_int as usize];
                }
                l += 1;
            }
        }
        j += psym;
    }
}
