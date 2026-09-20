//! Translation of `IMOD/libfft/diprp.c`.
#![allow(non_snake_case, unused_mut)]

/// C `diprp(pts, sym, psym, unsym, dim, x, y)`: `x` is `data`, and `y` is
/// `data` biased by `odd` -- 1 for interleaved storage, `ny` when `todfft`
/// has transposed the real and imaginary parts into separate rows.
pub fn diprp(
    pts: i32,
    sym: &[i32],
    psym: i32,
    unsym: &[i32],
    dim: &[i32; 6],
    data: &mut [f32],
    odd: usize,
) {
    let mut onemod = 0;
    let mut modulo = [0; 15];
    let mut dk = 0;
    let mut jj = 0;
    let mut kk = 0;
    let mut lk = 0;
    let mut mods = 0;
    let mut mult = 0;
    let mut nest = 0;
    let mut punsym = 0;
    let mut test = 0;
    let mut nt = 0;
    let mut sep = 0;
    let mut delta = 0;
    let mut p = 0;
    let mut p0 = 0;
    let mut p1 = 0;
    let mut p2 = 0;
    let mut p3 = 0;
    let mut p4 = 0;
    let mut p5 = 0;
    let mut size = 0;
    let mut s = [0; 15];
    let mut u = [0; 15];
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut d = 0;
    let mut e = 0;
    let mut f = 0;
    let mut g = 0;
    let mut h = 0;
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    let mut l = 0;
    let mut m = 0;
    let mut n = 0;
    nest = 14;
    nt = dim[1];
    sep = dim[2];
    p2 = dim[3];
    size = dim[4] - 1;
    p4 = dim[5];
    if sym[1] != 0 {
        j = 1;
        while j <= nest {
            u[j as usize] = 1;
            s[j as usize] = 1;
            j += 1;
        }
        n = pts;
        j = 1;
        while j <= nest {
            if sym[j as usize] == 0 {
                break;
            }
            jj = nest + 1 - j;
            u[jj as usize] = n;
            s[jj as usize] = n / sym[j as usize];
            n /= sym[j as usize];
            j += 1;
        }
        jj = 0;
        a = 1;
        while a <= u[1] {
            b = a;
            while b <= u[2] {
                c = b;
                while c <= u[3] {
                    d = c;
                    while d <= u[4] {
                        e = d;
                        while e <= u[5] {
                            f = e;
                            while f <= u[6] {
                                g = f;
                                while g <= u[7] {
                                    h = g;
                                    while h <= u[8] {
                                        i = h;
                                        while i <= u[9] {
                                            j = i;
                                            while j <= u[10] {
                                                k = j;
                                                while k <= u[11] {
                                                    l = k;
                                                    while l <= u[12] {
                                                        m = l;
                                                        while m <= u[13] {
                                                            n = m;
                                                            while n <= u[14] {
                                                                jj += 1;
                                                                if !(jj >= n) {
                                                                    delta = (n - jj) * sep;
                                                                    p1 = (jj - 1) * sep + 1;
                                                                    p0 = p1;
                                                                    while p0 <= nt {
                                                                        p3 = p0 + size;
                                                                        p = p0 - 1;
                                                                        while p < p3 {
                                                                            p5 = p + delta;
                                                                            data.swap(
                                                                                p as usize,
                                                                                p5 as usize,
                                                                            );
                                                                            data.swap(
                                                                                p as usize + odd,
                                                                                p5 as usize + odd,
                                                                            );
                                                                            p += p4;
                                                                        }
                                                                        p0 += p2;
                                                                    }
                                                                }
                                                                n += s[14];
                                                            }
                                                            m += s[13];
                                                        }
                                                        l += s[12];
                                                    }
                                                    k += s[11];
                                                }
                                                j += s[10];
                                            }
                                            i += s[9];
                                        }
                                        h += s[8];
                                    }
                                    g += s[7];
                                }
                                f += s[6];
                            }
                            e += s[5];
                        }
                        d += s[4];
                    }
                    c += s[3];
                }
                b += s[2];
            }
            a += 1;
        }
    }
    if unsym[1] == 0 {
        return;
    }
    punsym = pts / (psym * psym);
    mult = punsym / unsym[1];
    test = (unsym[1] * unsym[2] - 1) * mult * psym;
    lk = mult;
    dk = mult;
    k = 2;
    while k <= nest {
        if unsym[k as usize] == 0 {
            break;
        }
        lk *= unsym[(k - 1) as usize];
        dk /= unsym[k as usize];
        u[k as usize] = (lk - dk) * psym;
        mods = k;
        k += 1;
    }
    onemod = i32::from(mods < 3);
    if onemod == 0 {
        j = 3;
        while j <= mods {
            jj = mods + 3 - j;
            modulo[jj as usize] = u[j as usize];
            j += 1;
        }
    }
    modulo[2] = u[2];
    u[10] = (punsym - 3) * psym;
    s[13] = punsym * psym;
    j = psym;
    while j <= u[10] {
        k = j;
        loop {
            k = k * mult;
            if onemod == 0 {
                i = 3;
                while i <= mods {
                    k = k - k / modulo[i as usize] * modulo[i as usize];
                    i += 1;
                }
            }
            if k < test {
                k = k - k / modulo[2] * modulo[2];
            } else {
                k = k - k / modulo[2] * modulo[2] + modulo[2];
            }
            if !(k < j) {
                break;
            }
        }
        if k != j {
            delta = (k - j) * sep;
            l = 1;
            while l <= psym {
                m = l;
                while m <= pts {
                    p1 = (m + j - 1) * sep + 1;
                    p0 = p1;
                    while p0 <= nt {
                        p3 = p0 + size;
                        jj = p0 - 1;
                        while jj < p3 {
                            kk = jj + delta;
                            data.swap(jj as usize, kk as usize);
                            data.swap(jj as usize + odd, kk as usize + odd);
                            jj += p4;
                        }
                        p0 += p2;
                    }
                    m += s[13];
                }
                l += 1;
            }
        }
        j += psym;
    }
}
