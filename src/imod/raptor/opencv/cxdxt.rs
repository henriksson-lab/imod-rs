//! Translation of `IMOD/raptor/opencv/cxdxt.cpp` (the parts RAPTOR reaches):
//! the double-precision discrete Fourier transform `cvDFT`,
//! `cvMulSpectrums` and `cvGetOptimalDFTSize`.
//!
//! RAPTOR reaches `cvDFT` only from `icvCrossCorr`, always in place (the
//! source and destination are the same `CV_64FC1` matrix or sub-matrix),
//! forward (`CV_DXT_FORWARD`, with or without `CV_DXT_SCALE`) or inverse
//! (`CV_DXT_INVERSE`) of a real array, with a `nonzero_rows` count.  So
//! [`cv_dft`] takes one mutable `f64` array; the single-precision kernels,
//! the complex-input/-output channel combinations, `CV_DXT_ROWS` and the IPP
//! hooks (their function pointers are always null in this build) are not
//! reached and are recorded in `DEAD_CODE.md`.
//!
//! The kernels work on `double` arrays that the C reinterprets as arrays of
//! `CvComplex64f`; here they are `f64` slices and complex element `k` is
//! `[2k]` (re) and `[2k + 1]` (im).  Where the C passes the same pointer as
//! source and destination the Rust passes `None` for the source.  The
//! scratch buffer the C carves out of one `cvStackAlloc` block (twiddle
//! table, permutation table, row buffer, column buffers, radix scratch) is
//! separate `Vec`s; no reached path reads a part of it before writing it.

use super::cxerror::{CV_STS_NOT_IMPLEMENTED, cv_error};

/// `CvComplex64f`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CvComplex64f {
    pub re: f64,
    pub im: f64,
}

/// `CV_DXT_FORWARD`.
pub const CV_DXT_FORWARD: i32 = 0;
/// `CV_DXT_INVERSE`.
pub const CV_DXT_INVERSE: i32 = 1;
/// `CV_DXT_SCALE`: divide result by size of array.
pub const CV_DXT_SCALE: i32 = 2;
/// `CV_DXT_ROWS`: transform each row individually.
pub const CV_DXT_ROWS: i32 = 4;
/// `CV_DXT_MUL_CONJ`: conjugate the second argument of `cvMulSpectrums`.
pub const CV_DXT_MUL_CONJ: i32 = 8;

const LOG2TAB: [u8; 16] = [0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];

/// `icvlog2( int n )` (`cxdxt.cpp:100`).
fn icvlog2(mut n: i32) -> i32 {
    let mut m = 0;
    let mut f = (n >= (1 << 16)) as i32 * 16;
    n >>= f;
    m += f;
    f = (n >= (1 << 8)) as i32 * 8;
    n >>= f;
    m += f;
    f = (n >= (1 << 4)) as i32 * 4;
    n >>= f;
    m + f + LOG2TAB[n as usize] as i32
}

/// `icvRevTable` (`cxdxt.cpp:114`).
static ICV_REV_TABLE: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];

/// `icvDxtTab` (`cxdxt.cpp:134`).
static ICV_DXT_TAB: [[f64; 2]; 32] = [
    [1.00000000000000000, 0.00000000000000000],
    [-1.00000000000000000, 0.00000000000000000],
    [0.00000000000000000, 1.00000000000000000],
    [0.70710678118654757, 0.70710678118654746],
    [0.92387953251128674, 0.38268343236508978],
    [0.98078528040323043, 0.19509032201612825],
    [0.99518472667219693, 0.09801714032956060],
    [0.99879545620517241, 0.04906767432741802],
    [0.99969881869620425, 0.02454122852291229],
    [0.99992470183914450, 0.01227153828571993],
    [0.99998117528260111, 0.00613588464915448],
    [0.99999529380957619, 0.00306795676296598],
    [0.99999882345170188, 0.00153398018628477],
    [0.99999970586288223, 0.00076699031874270],
    [0.99999992646571789, 0.00038349518757140],
    [0.99999998161642933, 0.00019174759731070],
    [0.99999999540410733, 0.00009587379909598],
    [0.99999999885102686, 0.00004793689960307],
    [0.99999999971275666, 0.00002396844980842],
    [0.99999999992818922, 0.00001198422490507],
    [0.99999999998204725, 0.00000599211245264],
    [0.99999999999551181, 0.00000299605622633],
    [0.99999999999887801, 0.00000149802811317],
    [0.99999999999971945, 0.00000074901405658],
    [0.99999999999992983, 0.00000037450702829],
    [0.99999999999998246, 0.00000018725351415],
    [0.99999999999999567, 0.00000009362675707],
    [0.99999999999999889, 0.00000004681337854],
    [0.99999999999999978, 0.00000002340668927],
    [0.99999999999999989, 0.00000001170334463],
    [1.00000000000000000, 0.00000000585167232],
    [1.00000000000000000, 0.00000000292583616],
];

/// The `icvBitRev(i, shift)` macro (`cxdxt.cpp:170`).
fn icv_bit_rev(i: i32, shift: i32) -> i32 {
    let u = ((ICV_REV_TABLE[(i & 255) as usize] as u32) << 24)
        .wrapping_add((ICV_REV_TABLE[((i >> 8) & 255) as usize] as u32) << 16)
        .wrapping_add((ICV_REV_TABLE[((i >> 16) & 255) as usize] as u32) << 8)
        .wrapping_add(ICV_REV_TABLE[((i >> 24) & 255) as usize] as u32);
    (u >> shift) as i32
}

/// `icvDFTFactorize( int n, int* factors )` (`cxdxt.cpp:177`).
fn icv_dft_factorize(mut n: i32, factors: &mut [i32; 34]) -> i32 {
    let mut nf = 0usize;
    let mut f;

    if n <= 5 {
        factors[0] = n;
        return 1;
    }

    f = (((n - 1) ^ n) + 1) >> 1;
    if f > 1 {
        factors[nf] = f;
        nf += 1;
        n = if f == n { 1 } else { n / f };
    }

    f = 3;
    while n > 1 {
        let d = n / f;
        if d * f == n {
            factors[nf] = f;
            nf += 1;
            n = d;
        } else {
            f += 2;
            if f * f > n {
                break;
            }
        }
    }

    if n > 1 {
        factors[nf] = n;
        nf += 1;
    }

    let f = ((factors[0] & 1) == 0) as usize;
    let mut i = f;
    while i < (nf + f) / 2 {
        factors.swap(i, nf - i - 1 + f);
        i += 1;
    }

    nf as i32
}

/// `icvDFTInit( int n0, int nf, int* factors, int* itab, int elem_size,
/// void* _wave, int inv_itab )` (`cxdxt.cpp:240`), for `elem_size ==
/// sizeof(CvComplex64f)`.  Where the C borrows the twiddle array as integer
/// scratch for the inverse permutation (`itab = (int*)_wave`), a separate
/// scratch array is used; the twiddles are written over it afterwards either
/// way.
fn icv_dft_init(
    n0: i32,
    nf: i32,
    factors: &[i32; 34],
    itab0: &mut [i32],
    wave: &mut [CvComplex64f],
    inv_itab: bool,
) {
    let mut digits = [0i32; 34];
    let mut radix = [0i32; 34];
    let mut n = factors[0];
    let mut m = 0;
    let mut w: CvComplex64f;
    let mut w1 = CvComplex64f::default();
    let mut t: f64;

    if n0 <= 5 {
        itab0[0] = 0;
        itab0[(n0 - 1) as usize] = n0 - 1;

        if n0 != 4 {
            for i in 1..n0 - 1 {
                itab0[i as usize] = i;
            }
        } else {
            itab0[1] = 2;
            itab0[2] = 1;
        }
        if n0 == 5 {
            wave[0] = CvComplex64f { re: 1., im: 0. };
        }
        if n0 != 4 {
            return;
        }
        m = 2;
    } else {
        // radix[] is initialized from index 'nf' down to zero
        let nfu = nf as usize;
        assert!(nf < 34);
        radix[nfu] = 1;
        digits[nfu] = 0;
        for i in 0..nfu {
            digits[i] = 0;
            radix[nfu - i - 1] = radix[nfu - i] * factors[nfu - i - 1];
        }

        // The table is generated into `itab`, which is `itab0` itself
        // unless the inverse permutation is wanted.
        let use_scratch = inv_itab && factors[0] != factors[nfu - 1];
        let mut itab = vec![0i32; n0 as usize];

        if (n & 1) == 0 {
            let a = radix[1];
            let na2 = (n * a) >> 1;
            let na4 = na2 >> 1;
            m = icvlog2(n);

            if n <= 2 {
                itab[0] = 0;
                itab[1] = na2;
            } else if n <= 256 {
                let shift = 10 - m;
                let mut i = 0;
                while i <= n - 4 {
                    let j = (ICV_REV_TABLE[(i >> 2) as usize] as i32 >> shift) * a;
                    itab[i as usize] = j;
                    itab[(i + 1) as usize] = j + na2;
                    itab[(i + 2) as usize] = j + na4;
                    itab[(i + 3) as usize] = j + na2 + na4;
                    i += 4;
                }
            } else {
                let shift = 34 - m;
                let mut i = 0;
                while i < n {
                    let i4 = i >> 2;
                    let j = icv_bit_rev(i4, shift) * a;
                    itab[i as usize] = j;
                    itab[(i + 1) as usize] = j + na2;
                    itab[(i + 2) as usize] = j + na4;
                    itab[(i + 3) as usize] = j + na2 + na4;
                    i += 4;
                }
            }

            digits[1] += 1;

            // The C reads radix[2] here even when nf == 1 (the assert that
            // guarded it was removed upstream, "to allow DFT of powers of
            // 2"); the value is then unused because the loop below does not
            // run (n == n0).
            let mut i = n;
            let mut j = if nf >= 2 { radix[2] } else { 0 };
            while i < n0 {
                for k in 0..n {
                    itab[(i + k) as usize] = itab[k as usize] + j;
                }
                i += n;
                if i >= n0 {
                    break;
                }
                j += radix[2];
                let mut k = 1usize;
                loop {
                    digits[k] += 1;
                    if digits[k] < factors[k] {
                        break;
                    }
                    digits[k] = 0;
                    j += radix[k + 2] - radix[k];
                    k += 1;
                }
            }
        } else {
            let mut i = 0;
            let mut j = 0;
            loop {
                itab[i as usize] = j;
                i += 1;
                if i >= n0 {
                    break;
                }
                j += radix[1];
                let mut k = 0usize;
                loop {
                    digits[k] += 1;
                    if digits[k] < factors[k] {
                        break;
                    }
                    digits[k] = 0;
                    j += radix[k + 2] - radix[k];
                    k += 1;
                }
            }
        }

        if !use_scratch {
            itab0[..n0 as usize].copy_from_slice(&itab);
        } else {
            itab0[0] = 0;
            let mut i = n0 & 1;
            while i < n0 {
                let k0 = itab[i as usize];
                let k1 = itab[(i + 1) as usize];
                itab0[k0 as usize] = i;
                itab0[k1 as usize] = i + 1;
                i += 2;
            }
        }
    }

    if (n0 & (n0 - 1)) == 0 {
        w1.re = ICV_DXT_TAB[m as usize][0];
        w1.im = -ICV_DXT_TAB[m as usize][1];
        w = w1;
    } else {
        t = -CV_PI * 2. / n0 as f64;
        w1.im = t.sin();
        w1.re = (1. - w1.im * w1.im).sqrt();
        w = w1;
    }
    n = (n0 + 1) / 2;

    wave[0].re = 1.;
    wave[0].im = 0.;

    if (n0 & 1) == 0 {
        wave[n as usize].re = -1.;
        wave[n as usize].im = 0.;
    }

    for i in 1..n {
        wave[i as usize] = w;
        wave[(n0 - i) as usize].re = w.re;
        wave[(n0 - i) as usize].im = -w.im;

        t = w.re * w1.re - w.im * w1.im;
        w.im = w.re * w1.im + w.im * w1.re;
        w.re = t;
    }
}

const ICV_SIN_120: f64 = 0.86602540378443864676372317075294;
const ICV_FFT5_2: f64 = 0.559016994374947424102293417182819;
const ICV_FFT5_3: f64 = -0.951056516295153572116439333379382;
const ICV_FFT5_4: f64 = -1.538841768587626701285145288018455;
const ICV_FFT5_5: f64 = 0.363271264002680442947733378740309;

/// `ICV_DFT_NO_PERMUTE`.
const ICV_DFT_NO_PERMUTE: i32 = 2;
/// `ICV_DFT_COMPLEX_INPUT_OR_OUTPUT`.
const ICV_DFT_COMPLEX_INPUT_OR_OUTPUT: i32 = 4;

/// `CV_PI` (`cxtypes.h`).
const CV_PI: f64 = 3.1415926535897932384626433832795;

/// `icvDFT_64fc( const CvComplex64f* src, CvComplex64f* dst, int n, int nf,
/// int* factors, const int* itab, const CvComplex64f* wave, int tab_size,
/// const void* spec, CvComplex64f* buf, int flags, double scale )`
/// (`cxdxt.cpp:451`): mixed-radix complex discrete Fourier transform,
/// double-precision version.  `src` is `None` where the C passes `dst`
/// itself.  `spec` (the IPP plan) is always null.  Returns the C status
/// (`CV_OK` 0, or `CV_INPLACE_NOT_SUPPORTED_ERR`).
#[allow(clippy::too_many_arguments)]
fn icv_dft_64fc(
    src: Option<&[f64]>,
    dst: &mut [f64],
    mut n: i32,
    nf: i32,
    factors: &[i32],
    itab: &[i32],
    wave: &[CvComplex64f],
    tab_size: i32,
    buf: &mut [CvComplex64f],
    flags: i32,
    scale: f64,
) -> i32 {
    let n0 = n;
    let inv = flags & CV_DXT_INVERSE;
    let mut dw0 = tab_size;
    let mut dw;
    let mut nx;
    let tab_step = if tab_size == n {
        1
    } else if tab_size == n * 2 {
        2
    } else {
        tab_size / n
    };
    let mut it = 0usize; // `itab` pointer

    // 0. shuffle data
    if let Some(src) = src {
        assert!((flags & ICV_DFT_NO_PERMUTE) == 0);
        let mut i = 0;
        if inv == 0 {
            while i <= n - 2 {
                let k0 = itab[it] as usize;
                let k1 = itab[it + tab_step as usize] as usize;
                let iu = i as usize;
                dst[2 * iu] = src[2 * k0];
                dst[2 * iu + 1] = src[2 * k0 + 1];
                dst[2 * iu + 2] = src[2 * k1];
                dst[2 * iu + 3] = src[2 * k1 + 1];
                i += 2;
                it += 2 * tab_step as usize;
            }

            if i < n {
                let l = (n - 1) as usize;
                dst[2 * l] = src[2 * l];
                dst[2 * l + 1] = src[2 * l + 1];
            }
        } else {
            while i <= n - 2 {
                let k0 = itab[it] as usize;
                let k1 = itab[it + tab_step as usize] as usize;
                let iu = i as usize;
                dst[2 * iu] = src[2 * k0];
                dst[2 * iu + 1] = -src[2 * k0 + 1];
                dst[2 * iu + 2] = src[2 * k1];
                dst[2 * iu + 3] = -src[2 * k1 + 1];
                i += 2;
                it += 2 * tab_step as usize;
            }

            if i < n {
                let l = (n - 1) as usize;
                let iu = i as usize;
                dst[2 * iu] = src[2 * l];
                dst[2 * iu + 1] = -src[2 * l + 1];
            }
        }
    } else {
        if (flags & ICV_DFT_NO_PERMUTE) == 0 {
            if factors[0] != factors[(nf - 1) as usize] {
                return CV_INPLACE_NOT_SUPPORTED_ERR;
            }
            if nf == 1 {
                if (n & 3) == 0 {
                    let n2 = n / 2;
                    let h = n2 as usize; // `dsth = dst + n2`
                    let mut i = 0;
                    while i < n2 {
                        let j = itab[it];
                        assert!((j as u32) < (n2 as u32));
                        let (iu, ju) = (i as usize, j as usize);

                        swap_c(dst, iu + 1, h + ju);
                        if j > i {
                            swap_c(dst, iu, ju);
                            swap_c(dst, h + iu + 1, h + ju + 1);
                        }
                        i += 2;
                        it += (tab_step * 2) as usize;
                    }
                }
                // else do nothing
            } else {
                for i in 0..n {
                    let j = itab[it];
                    assert!((j as u32) < (n as u32));
                    if j > i {
                        swap_c(dst, i as usize, j as usize);
                    }
                    it += tab_step as usize;
                }
            }
        }

        if inv != 0 {
            let mut i = 0;
            while i <= n - 2 {
                let iu = i as usize;
                let t0 = -dst[2 * iu + 1];
                let t1 = -dst[2 * iu + 3];
                dst[2 * iu + 1] = t0;
                dst[2 * iu + 3] = t1;
                i += 2;
            }

            if i < n {
                let l = (n - 1) as usize;
                dst[2 * l + 1] = -dst[2 * l + 1];
            }
        }
    }

    n = 1;
    // 1. power-2 transforms
    if (factors[0] & 1) == 0 {
        // radix-4 transform
        while n * 4 <= factors[0] {
            nx = n;
            n *= 4;
            dw0 /= 4;

            let mut i = 0;
            while i < n0 {
                let v0 = i as usize;
                let v1 = v0 + (nx * 2) as usize;
                let nxu = nx as usize;

                let mut r2 = dst[2 * v0];
                let mut i2 = dst[2 * v0 + 1];
                let mut r1 = dst[2 * (v0 + nxu)];
                let mut i1 = dst[2 * (v0 + nxu) + 1];

                let r0 = r1 + r2;
                let i0 = i1 + i2;
                r2 -= r1;
                i2 -= i1;

                let mut i3 = dst[2 * (v1 + nxu)];
                let mut r3 = dst[2 * (v1 + nxu) + 1];
                let i4 = dst[2 * v1];
                let r4 = dst[2 * v1 + 1];

                r1 = i4 + i3;
                i1 = r4 + r3;
                r3 = r4 - r3;
                i3 = i3 - i4;

                dst[2 * v0] = r0 + r1;
                dst[2 * v0 + 1] = i0 + i1;
                dst[2 * v1] = r0 - r1;
                dst[2 * v1 + 1] = i0 - i1;
                dst[2 * (v0 + nxu)] = r2 + r3;
                dst[2 * (v0 + nxu) + 1] = i2 + i3;
                dst[2 * (v1 + nxu)] = r2 - r3;
                dst[2 * (v1 + nxu) + 1] = i2 - i3;

                let mut j = 1;
                dw = dw0;
                while j < nx {
                    let v0 = (i + j) as usize;
                    let v1 = v0 + (nx * 2) as usize;
                    let (w1, w2, w3) = (
                        wave[dw as usize],
                        wave[(dw * 2) as usize],
                        wave[(dw * 3) as usize],
                    );

                    let mut r2 = dst[2 * (v0 + nxu)] * w2.re - dst[2 * (v0 + nxu) + 1] * w2.im;
                    let mut i2 = dst[2 * (v0 + nxu)] * w2.im + dst[2 * (v0 + nxu) + 1] * w2.re;
                    let mut r0 = dst[2 * v1] * w1.im + dst[2 * v1 + 1] * w1.re;
                    let mut i0 = dst[2 * v1] * w1.re - dst[2 * v1 + 1] * w1.im;
                    let mut r3 = dst[2 * (v1 + nxu)] * w3.im + dst[2 * (v1 + nxu) + 1] * w3.re;
                    let mut i3 = dst[2 * (v1 + nxu)] * w3.re - dst[2 * (v1 + nxu) + 1] * w3.im;

                    let r1 = i0 + i3;
                    let i1 = r0 + r3;
                    r3 = r0 - r3;
                    i3 = i3 - i0;
                    let r4 = dst[2 * v0];
                    let i4 = dst[2 * v0 + 1];

                    r0 = r4 + r2;
                    i0 = i4 + i2;
                    r2 = r4 - r2;
                    i2 = i4 - i2;

                    dst[2 * v0] = r0 + r1;
                    dst[2 * v0 + 1] = i0 + i1;
                    dst[2 * v1] = r0 - r1;
                    dst[2 * v1 + 1] = i0 - i1;
                    dst[2 * (v0 + nxu)] = r2 + r3;
                    dst[2 * (v0 + nxu) + 1] = i2 + i3;
                    dst[2 * (v1 + nxu)] = r2 - r3;
                    dst[2 * (v1 + nxu) + 1] = i2 - i3;

                    j += 1;
                    dw += dw0;
                }
                i += n;
            }
        }

        while n < factors[0] {
            // do the remaining radix-2 transform
            nx = n;
            n *= 2;
            dw0 /= 2;

            let mut i = 0;
            while i < n0 {
                let v = i as usize;
                let nxu = nx as usize;
                let r0 = dst[2 * v] + dst[2 * (v + nxu)];
                let i0 = dst[2 * v + 1] + dst[2 * (v + nxu) + 1];
                let r1 = dst[2 * v] - dst[2 * (v + nxu)];
                let i1 = dst[2 * v + 1] - dst[2 * (v + nxu) + 1];
                dst[2 * v] = r0;
                dst[2 * v + 1] = i0;
                dst[2 * (v + nxu)] = r1;
                dst[2 * (v + nxu) + 1] = i1;

                let mut j = 1;
                dw = dw0;
                while j < nx {
                    let v = (i + j) as usize;
                    let wd = wave[dw as usize];
                    let r1 = dst[2 * (v + nxu)] * wd.re - dst[2 * (v + nxu) + 1] * wd.im;
                    let i1 = dst[2 * (v + nxu) + 1] * wd.re + dst[2 * (v + nxu)] * wd.im;
                    let r0 = dst[2 * v];
                    let i0 = dst[2 * v + 1];

                    dst[2 * v] = r0 + r1;
                    dst[2 * v + 1] = i0 + i1;
                    dst[2 * (v + nxu)] = r0 - r1;
                    dst[2 * (v + nxu) + 1] = i0 - i1;
                    j += 1;
                    dw += dw0;
                }
                i += n;
            }
        }
    }

    // 2. all the other transforms
    let mut f_idx = if (factors[0] & 1) != 0 { 0 } else { 1 };
    while f_idx < nf {
        let factor = factors[f_idx as usize];
        nx = n;
        n *= factor;
        dw0 /= factor;
        let nxu = nx as usize;

        if factor == 3 {
            // radix-3
            let mut i = 0;
            while i < n0 {
                let v = i as usize;

                let r1 = dst[2 * (v + nxu)] + dst[2 * (v + 2 * nxu)];
                let i1 = dst[2 * (v + nxu) + 1] + dst[2 * (v + 2 * nxu) + 1];
                let mut r0 = dst[2 * v];
                let mut i0 = dst[2 * v + 1];
                let r2 = ICV_SIN_120 * (dst[2 * (v + nxu) + 1] - dst[2 * (v + 2 * nxu) + 1]);
                let i2 = ICV_SIN_120 * (dst[2 * (v + 2 * nxu)] - dst[2 * (v + nxu)]);
                dst[2 * v] = r0 + r1;
                dst[2 * v + 1] = i0 + i1;
                r0 -= 0.5 * r1;
                i0 -= 0.5 * i1;
                dst[2 * (v + nxu)] = r0 + r2;
                dst[2 * (v + nxu) + 1] = i0 + i2;
                dst[2 * (v + 2 * nxu)] = r0 - r2;
                dst[2 * (v + 2 * nxu) + 1] = i0 - i2;

                let mut j = 1;
                dw = dw0;
                while j < nx {
                    let v = (i + j) as usize;
                    let w1 = wave[dw as usize];
                    let w2 = wave[(dw * 2) as usize];
                    let mut r0 = dst[2 * (v + nxu)] * w1.re - dst[2 * (v + nxu) + 1] * w1.im;
                    let mut i0 = dst[2 * (v + nxu)] * w1.im + dst[2 * (v + nxu) + 1] * w1.re;
                    let mut i2 =
                        dst[2 * (v + 2 * nxu)] * w2.re - dst[2 * (v + 2 * nxu) + 1] * w2.im;
                    let mut r2 =
                        dst[2 * (v + 2 * nxu)] * w2.im + dst[2 * (v + 2 * nxu) + 1] * w2.re;
                    let r1 = r0 + i2;
                    let i1 = i0 + r2;

                    r2 = ICV_SIN_120 * (i0 - r2);
                    i2 = ICV_SIN_120 * (i2 - r0);
                    r0 = dst[2 * v];
                    i0 = dst[2 * v + 1];
                    dst[2 * v] = r0 + r1;
                    dst[2 * v + 1] = i0 + i1;
                    r0 -= 0.5 * r1;
                    i0 -= 0.5 * i1;
                    dst[2 * (v + nxu)] = r0 + r2;
                    dst[2 * (v + nxu) + 1] = i0 + i2;
                    dst[2 * (v + 2 * nxu)] = r0 - r2;
                    dst[2 * (v + 2 * nxu) + 1] = i0 - i2;
                    j += 1;
                    dw += dw0;
                }
                i += n;
            }
        } else if factor == 5 {
            // radix-5
            let mut i = 0;
            while i < n0 {
                let mut j = 0;
                dw = 0;
                while j < nx {
                    let v0 = (i + j) as usize;
                    let v1 = v0 + nxu * 2;
                    let v2 = v1 + nxu * 2;
                    let wd1 = wave[dw as usize];
                    let wd2 = wave[(dw * 2) as usize];
                    let wd3 = wave[(dw * 3) as usize];
                    let wd4 = wave[(dw * 4) as usize];

                    let mut r3 = dst[2 * (v0 + nxu)] * wd1.re - dst[2 * (v0 + nxu) + 1] * wd1.im;
                    let mut i3 = dst[2 * (v0 + nxu)] * wd1.im + dst[2 * (v0 + nxu) + 1] * wd1.re;
                    let mut r2 = dst[2 * v2] * wd4.re - dst[2 * v2 + 1] * wd4.im;
                    let mut i2 = dst[2 * v2] * wd4.im + dst[2 * v2 + 1] * wd4.re;

                    let mut r1 = r3 + r2;
                    let mut i1 = i3 + i2;
                    r3 -= r2;
                    i3 -= i2;

                    let mut r4 = dst[2 * (v1 + nxu)] * wd3.re - dst[2 * (v1 + nxu) + 1] * wd3.im;
                    let mut i4 = dst[2 * (v1 + nxu)] * wd3.im + dst[2 * (v1 + nxu) + 1] * wd3.re;
                    let mut r0 = dst[2 * v1] * wd2.re - dst[2 * v1 + 1] * wd2.im;
                    let mut i0 = dst[2 * v1] * wd2.im + dst[2 * v1 + 1] * wd2.re;

                    r2 = r4 + r0;
                    i2 = i4 + i0;
                    r4 -= r0;
                    i4 -= i0;

                    r0 = dst[2 * v0];
                    i0 = dst[2 * v0 + 1];
                    let mut r5 = r1 + r2;
                    let mut i5 = i1 + i2;

                    dst[2 * v0] = r0 + r5;
                    dst[2 * v0 + 1] = i0 + i5;

                    r0 -= 0.25 * r5;
                    i0 -= 0.25 * i5;
                    r1 = ICV_FFT5_2 * (r1 - r2);
                    i1 = ICV_FFT5_2 * (i1 - i2);
                    r2 = -ICV_FFT5_3 * (i3 + i4);
                    i2 = ICV_FFT5_3 * (r3 + r4);

                    i3 *= -ICV_FFT5_5;
                    r3 *= ICV_FFT5_5;
                    i4 *= -ICV_FFT5_4;
                    r4 *= ICV_FFT5_4;

                    r5 = r2 + i3;
                    i5 = i2 + r3;
                    r2 -= i4;
                    i2 -= r4;

                    r3 = r0 + r1;
                    i3 = i0 + i1;
                    r0 -= r1;
                    i0 -= i1;

                    dst[2 * (v0 + nxu)] = r3 + r2;
                    dst[2 * (v0 + nxu) + 1] = i3 + i2;
                    dst[2 * v2] = r3 - r2;
                    dst[2 * v2 + 1] = i3 - i2;

                    dst[2 * v1] = r0 + r5;
                    dst[2 * v1 + 1] = i0 + i5;
                    dst[2 * (v1 + nxu)] = r0 - r5;
                    dst[2 * (v1 + nxu) + 1] = i0 - i5;
                    j += 1;
                    dw += dw0;
                }
                i += n;
            }
        } else {
            // radix-"factor" - an odd number
            let factor2 = (factor - 1) / 2;
            let dw_f = tab_size / factor;
            // `a = buf`, `b = buf + factor2`
            let bo = factor2 as usize;

            let mut i = 0;
            while i < n0 {
                let mut j = 0;
                dw = 0;
                while j < nx {
                    let v = (i + j) as usize;
                    let v_0 = CvComplex64f {
                        re: dst[2 * v],
                        im: dst[2 * v + 1],
                    };
                    let mut vn_0 = v_0;

                    if j == 0 {
                        let mut p = 1;
                        let mut k = nx;
                        while p <= factor2 {
                            let (ku, nku) = (v + k as usize, v + (n - k) as usize);
                            let r0 = dst[2 * ku] + dst[2 * nku];
                            let i0 = dst[2 * ku + 1] - dst[2 * nku + 1];
                            let r1 = dst[2 * ku] - dst[2 * nku];
                            let i1 = dst[2 * ku + 1] + dst[2 * nku + 1];

                            vn_0.re += r0;
                            vn_0.im += i1;
                            buf[(p - 1) as usize] = CvComplex64f { re: r0, im: i0 };
                            buf[bo + (p - 1) as usize] = CvComplex64f { re: r1, im: i1 };
                            p += 1;
                            k += nx;
                        }
                    } else {
                        // `wave_ = wave + dw*factor`
                        let wave_ = (dw * factor) as usize;
                        let mut d = dw;
                        let mut p = 1;
                        let mut k = nx;
                        while p <= factor2 {
                            let (ku, nku) = (v + k as usize, v + (n - k) as usize);
                            let wd = wave[d as usize];
                            let wm = wave[wave_ - d as usize];
                            let r2 = dst[2 * ku] * wd.re - dst[2 * ku + 1] * wd.im;
                            let i2 = dst[2 * ku] * wd.im + dst[2 * ku + 1] * wd.re;

                            let mut r1 = dst[2 * nku] * wm.re - dst[2 * nku + 1] * wm.im;
                            let mut i1 = dst[2 * nku] * wm.im + dst[2 * nku + 1] * wm.re;

                            let r0 = r2 + r1;
                            let i0 = i2 - i1;
                            r1 = r2 - r1;
                            i1 = i2 + i1;

                            vn_0.re += r0;
                            vn_0.im += i1;
                            buf[(p - 1) as usize] = CvComplex64f { re: r0, im: i0 };
                            buf[bo + (p - 1) as usize] = CvComplex64f { re: r1, im: i1 };
                            p += 1;
                            k += nx;
                            d += dw;
                        }
                    }

                    dst[2 * v] = vn_0.re;
                    dst[2 * v + 1] = vn_0.im;

                    let mut p = 1;
                    let mut k = nx;
                    while p <= factor2 {
                        let mut s0 = v_0;
                        let mut s1 = v_0;
                        let dd = dw_f * p;
                        let mut d = dd;

                        for q in 0..factor2 as usize {
                            let wd = wave[d as usize];
                            let r0 = wd.re * buf[q].re;
                            let i0 = wd.im * buf[q].im;
                            let r1 = wd.re * buf[bo + q].im;
                            let i1 = wd.im * buf[bo + q].re;

                            s1.re += r0 + i0;
                            s0.re += r0 - i0;
                            s1.im += r1 - i1;
                            s0.im += r1 + i1;

                            d += dd;
                            d -= -((d >= tab_size) as i32) & tab_size;
                        }

                        let (ku, nku) = (v + k as usize, v + (n - k) as usize);
                        dst[2 * ku] = s0.re;
                        dst[2 * ku + 1] = s0.im;
                        dst[2 * nku] = s1.re;
                        dst[2 * nku + 1] = s1.im;
                        p += 1;
                        k += nx;
                    }
                    j += 1;
                    dw += dw0;
                }
                i += n;
            }
        }
        f_idx += 1;
    }

    if (scale - 1.).abs() > f64::EPSILON {
        let re_scale = scale;
        let mut im_scale = scale;
        if inv != 0 {
            im_scale = -im_scale;
        }

        for i in 0..n0 as usize {
            let t0 = dst[2 * i] * re_scale;
            let t1 = dst[2 * i + 1] * im_scale;
            dst[2 * i] = t0;
            dst[2 * i + 1] = t1;
        }
    } else if inv != 0 {
        let mut i = 0;
        while i <= n0 - 2 {
            let iu = i as usize;
            let t0 = -dst[2 * iu + 1];
            let t1 = -dst[2 * iu + 3];
            dst[2 * iu + 1] = t0;
            dst[2 * iu + 3] = t1;
            i += 2;
        }

        if i < n0 {
            let l = (n0 - 1) as usize;
            dst[2 * l + 1] = -dst[2 * l + 1];
        }
    }

    CV_OK
}

/// `CV_SWAP` of two complex elements of a `double` array viewed as
/// `CvComplex64f`.
fn swap_c(d: &mut [f64], a: usize, b: usize) {
    d.swap(2 * a, 2 * b);
    d.swap(2 * a + 1, 2 * b + 1);
}

/// `CV_OK`.
const CV_OK: i32 = 0;
/// `CV_INPLACE_NOT_SUPPORTED_ERR`.
const CV_INPLACE_NOT_SUPPORTED_ERR: i32 = -112;

/// `icvRealDFT_64f` (the `ICV_REAL_DFT( 64f, double )` instantiation,
/// `cxdxt.cpp:1305`): FFT of a real vector, output
/// `re(0), re(1), im(1), ..., re(n/2-1), im((n+1)/2-1) [, re((n+1)/2)]`.
/// Complex output (`ICV_DFT_COMPLEX_INPUT_OR_OUTPUT`) is not reached (the
/// source and destination of RAPTOR's `cvDFT` have the same channel count).
#[allow(clippy::too_many_arguments)]
fn icv_real_dft_64f(
    src: Option<&[f64]>,
    dst: &mut [f64],
    n: i32,
    nf: i32,
    factors: &mut [i32; 34],
    itab: &[i32],
    wave: &[CvComplex64f],
    tab_size: i32,
    buf: &mut [CvComplex64f],
    flags: i32,
    scale: f64,
) -> i32 {
    let complex_output = (flags & ICV_DFT_COMPLEX_INPUT_OR_OUTPUT) != 0;
    assert!(!complex_output);
    let n2 = n >> 1;
    assert!(tab_size == n);
    let rd = |dst: &[f64], k: usize| -> f64 {
        match src {
            Some(s) => s[k],
            None => dst[k],
        }
    };

    if n == 1 {
        dst[0] = rd(dst, 0) * scale;
    } else if n == 2 {
        let t = (rd(dst, 0) + rd(dst, 1)) * scale;
        dst[1] = (rd(dst, 0) - rd(dst, 1)) * scale;
        dst[0] = t;
    } else if (n & 1) != 0 {
        // `_dst = (CvComplex64f*)dst`
        dst[0] = rd(dst, 0) * scale;
        dst[1] = 0.;
        let mut j = 1;
        while j < n {
            let t0 = rd(dst, itab[j as usize] as usize) * scale;
            let t1 = rd(dst, itab[(j + 1) as usize] as usize) * scale;
            let ju = j as usize;
            dst[2 * ju] = t0;
            dst[2 * ju + 1] = 0.;
            dst[2 * ju + 2] = t1;
            dst[2 * ju + 3] = 0.;
            j += 2;
        }
        icv_dft_64fc(
            None,
            dst,
            n,
            nf,
            &factors[..],
            itab,
            wave,
            tab_size,
            buf,
            ICV_DFT_NO_PERMUTE,
            1.,
        );
        if !complex_output {
            dst[1] = dst[0];
        }
        return CV_OK;
    } else {
        let mut t0;
        let mut t;
        let mut h1_re;
        let mut h1_im;
        let mut h2_re;
        let mut h2_im;
        let scale2 = scale * 0.5;
        factors[0] >>= 1;

        let off = (factors[0] == 1) as usize;
        let sub_factors: [i32; 34] = *factors;
        icv_dft_64fc(
            src,
            dst,
            n2,
            nf - off as i32,
            &sub_factors[off..],
            itab,
            wave,
            tab_size,
            buf,
            0,
            1.,
        );
        factors[0] <<= 1;

        t = dst[0] - dst[1];
        dst[0] = (dst[0] + dst[1]) * scale;
        dst[1] = t * scale;

        t0 = dst[n2 as usize];
        t = dst[(n - 1) as usize];
        dst[(n - 1) as usize] = dst[1];

        let nu = n as usize;
        let mut j = 2usize;
        let mut w = 1usize; // `wave++`
        while j < n2 as usize {
            // calc odd
            h2_re = scale2 * (dst[j + 1] + t);
            h2_im = scale2 * (dst[nu - j] - dst[j]);

            // calc even
            h1_re = scale2 * (dst[j] + dst[nu - j]);
            h1_im = scale2 * (dst[j + 1] - t);

            // rotate
            t = h2_re * wave[w].re - h2_im * wave[w].im;
            h2_im = h2_re * wave[w].im + h2_im * wave[w].re;
            h2_re = t;
            t = dst[nu - j - 1];

            dst[j - 1] = h1_re + h2_re;
            dst[nu - j - 1] = h1_re - h2_re;
            dst[j] = h1_im + h2_im;
            dst[nu - j] = h2_im - h1_im;
            j += 2;
            w += 1;
        }

        if j <= n2 as usize {
            dst[(n2 - 1) as usize] = t0 * scale;
            dst[n2 as usize] = -t * scale;
        }
    }

    CV_OK
}

/// `icvCCSIDFT_64f` (the `ICV_CCS_IDFT( 64f, double )` instantiation,
/// `cxdxt.cpp:1424`): inverse FFT of a complex conjugate-symmetric vector
/// stored as `re[0], re[1], im[1], ..., re[n/2]`.  Complex input is not
/// reached.
#[allow(clippy::too_many_arguments)]
fn icv_ccs_idft_64f(
    src: Option<&[f64]>,
    dst: &mut [f64],
    n: i32,
    nf: i32,
    factors: &mut [i32; 34],
    itab: &[i32],
    wave: &[CvComplex64f],
    tab_size: i32,
    buf: &mut [CvComplex64f],
    flags: i32,
    scale: f64,
) -> i32 {
    let complex_input = (flags & ICV_DFT_COMPLEX_INPUT_OR_OUTPUT) != 0;
    assert!(!complex_input);
    let n2 = (n + 1) >> 1;
    let mut t0;
    let mut t1;
    let mut t2;
    let mut t3;
    let mut t;

    assert!(tab_size == n);
    let rd = |dst: &[f64], k: usize| -> f64 {
        match src {
            Some(s) => s[k],
            None => dst[k],
        }
    };

    if n == 1 {
        dst[0] = rd(dst, 0) * scale;
    } else if n == 2 {
        t = (rd(dst, 0) + rd(dst, 1)) * scale;
        dst[1] = (rd(dst, 0) - rd(dst, 1)) * scale;
        dst[0] = t;
    } else if (n & 1) != 0 {
        // `_src = (CvComplex64f*)(src-1)`: complex j of _src is src[2j-1],
        // src[2j]; `_dst = (CvComplex64f*)dst`.  The source is a separate
        // buffer on every reached path (odd lengths are never in place).
        let s = src.expect("odd-length CCS inverse is out of place");
        dst[0] = s[0];
        dst[1] = 0.;
        for j in 1..n2 {
            let k0 = itab[j as usize] as usize;
            let k1 = itab[(n - j) as usize] as usize;
            let ju = j as usize;
            t0 = s[2 * ju - 1];
            t1 = s[2 * ju];
            dst[2 * k0] = t0;
            dst[2 * k0 + 1] = -t1;
            dst[2 * k1] = t0;
            dst[2 * k1 + 1] = t1;
        }

        icv_dft_64fc(
            None,
            dst,
            n,
            nf,
            &factors[..],
            itab,
            wave,
            tab_size,
            buf,
            ICV_DFT_NO_PERMUTE,
            1.,
        );
        dst[0] = dst[0] * scale;
        let mut j = 1usize;
        while j < n as usize {
            t0 = dst[j * 2] * scale;
            t1 = dst[j * 2 + 2] * scale;
            dst[j] = t0;
            dst[j + 1] = t1;
            j += 2;
        }
    } else {
        let inplace = src.is_none();
        let nu = n as usize;
        let n2u = n2 as usize;
        let mut w = 0usize;

        t = rd(dst, 1);
        t0 = rd(dst, 0) + rd(dst, nu - 1);
        t1 = rd(dst, nu - 1) - rd(dst, 0);
        dst[0] = t0;
        dst[1] = t1;

        let mut j = 2usize;
        w += 1;
        while j < n2u {
            let h1_re = t + rd(dst, nu - j - 1);
            let h1_im = rd(dst, j) - rd(dst, nu - j);

            let mut h2_re = t - rd(dst, nu - j - 1);
            let mut h2_im = rd(dst, j) + rd(dst, nu - j);

            t = h2_re * wave[w].re + h2_im * wave[w].im;
            h2_im = h2_im * wave[w].re - h2_re * wave[w].im;
            h2_re = t;

            t = rd(dst, j + 1);
            t0 = h1_re - h2_im;
            t1 = -h1_im - h2_re;
            t2 = h1_re + h2_im;
            t3 = h1_im - h2_re;

            if inplace {
                dst[j] = t0;
                dst[j + 1] = t1;
                dst[nu - j] = t2;
                dst[nu - j + 1] = t3;
            } else {
                let j2 = j >> 1;
                let mut k = itab[j2] as usize;
                dst[k] = t0;
                dst[k + 1] = t1;
                k = itab[n2u - j2] as usize;
                dst[k] = t2;
                dst[k + 1] = t3;
            }
            j += 2;
            w += 1;
        }

        if j <= n2u {
            t0 = t * 2.;
            t1 = rd(dst, n2u) * 2.;

            if inplace {
                dst[n2u] = t0;
                dst[n2u + 1] = t1;
            } else {
                let k = itab[n2u] as usize;
                dst[k * 2] = t0;
                dst[k * 2 + 1] = t1;
            }
        }

        factors[0] >>= 1;
        let off = (factors[0] == 1) as usize;
        let sub_factors: [i32; 34] = *factors;
        icv_dft_64fc(
            None,
            dst,
            n2,
            nf - off as i32,
            &sub_factors[off..],
            itab,
            wave,
            tab_size,
            buf,
            if inplace { 0 } else { ICV_DFT_NO_PERMUTE },
            1.,
        );
        factors[0] <<= 1;

        let mut j = 0usize;
        while j < nu {
            t0 = dst[j] * scale;
            t1 = dst[j + 1] * (-scale);
            dst[j] = t0;
            dst[j + 1] = t1;
            j += 2;
        }
    }

    CV_OK
}

/// `icvCopyColumn( const uchar* _src, int src_step, uchar* _dst, int
/// dst_step, int len, int elem_size )` (`cxdxt.cpp:1590`), in `double`
/// units: `src_step`/`dst_step` and `elem_size` (1 for a real, 2 for a
/// complex element) count `double`s.  A bit copy, as the C's `int` copy is.
fn icv_copy_column(
    src: &[f64],
    src_step: usize,
    dst: &mut [f64],
    dst_step: usize,
    len: usize,
    elem_size: usize,
) {
    let (mut s, mut d) = (0usize, 0usize);
    for _ in 0..len {
        for e in 0..elem_size {
            dst[d + e] = src[s + e];
        }
        s += src_step;
        d += dst_step;
    }
}

/// `icvCopyFrom2Columns( const uchar* _src, int src_step, uchar* _dst0,
/// uchar* _dst1, int len, int elem_size )` (`cxdxt.cpp:1627`), in
/// `double` units (`elem_size` 2: complex).
fn icv_copy_from_2_columns(
    src: &[f64],
    src_step: usize,
    dst0: &mut [f64],
    dst1: &mut [f64],
    len: usize,
    elem_size: usize,
) {
    let mut s = 0usize;
    let mut i = 0usize;
    while i < len * elem_size {
        for e in 0..elem_size {
            dst0[i + e] = src[s + e];
            dst1[i + e] = src[s + elem_size + e];
        }
        i += elem_size;
        s += src_step;
    }
}

/// `icvCopyTo2Columns( const uchar* _src0, const uchar* _src1, uchar* _dst,
/// int dst_step, int len, int elem_size )` (`cxdxt.cpp:1673`), in `double`
/// units.
fn icv_copy_to_2_columns(
    src0: &[f64],
    src1: &[f64],
    dst: &mut [f64],
    dst_step: usize,
    len: usize,
    elem_size: usize,
) {
    let mut d = 0usize;
    let mut i = 0usize;
    while i < len * elem_size {
        for e in 0..elem_size {
            dst[d + e] = src0[i + e];
            dst[d + elem_size + e] = src1[i + e];
        }
        i += elem_size;
        d += dst_step;
    }
}

/// `icvExpandCCS( uchar* _ptr, int len, int elem_size )`
/// (`cxdxt.cpp:1719`) for `double`: `buf` is the C's `_ptr - elem_size`,
/// the start of a `len`-element complex vector whose `[1..=len]` hold a CCS
/// column.
fn icv_expand_ccs(buf: &mut [f64], len: usize) {
    buf[0] = buf[1];
    buf[1] = 0.;
    if (len & 1) == 0 {
        buf[len + 1] = 0.;
    }

    for i in 1..(len + 1) / 2 {
        let re = buf[2 * i];
        let im = -buf[2 * i + 1];
        buf[2 * (len - i)] = re;
        buf[2 * (len - i) + 1] = im;
    }
}

/// Which `cvDFT` kernel (`dft_tbl[3..6]`, the double-precision ones).
#[derive(Clone, Copy, PartialEq)]
enum DftFunc {
    Complex,
    Real,
    CcsInverse,
}

/// Calls the selected `dft_tbl` entry.
#[allow(clippy::too_many_arguments)]
fn call_dft(
    func: DftFunc,
    src: Option<&[f64]>,
    dst: &mut [f64],
    len: i32,
    nf: i32,
    factors: &mut [i32; 34],
    itab: &[i32],
    wave: &[CvComplex64f],
    buf: &mut [CvComplex64f],
    flags: i32,
    scale: f64,
) {
    match func {
        DftFunc::Complex => {
            icv_dft_64fc(
                src,
                dst,
                len,
                nf,
                &factors[..],
                itab,
                wave,
                len,
                buf,
                flags,
                scale,
            );
        }
        DftFunc::Real => {
            icv_real_dft_64f(
                src, dst, len, nf, factors, itab, wave, len, buf, flags, scale,
            );
        }
        DftFunc::CcsInverse => {
            icv_ccs_idft_64f(
                src, dst, len, nf, factors, itab, wave, len, buf, flags, scale,
            );
        }
    }
}

/// `cvDFT( const CvArr* srcarr, CvArr* dstarr, int flags, int
/// nonzero_rows )` (`cxdxt.cpp:1761`), for the case RAPTOR reaches: the
/// source and destination are the same single-channel `CV_64F` array
/// (`data`, `rows` x `cols`, `step` in bytes as in the C header), so the
/// transform is real (forward to CCS packed format, or CCS inverse).
/// `cont` is the header's `CV_IS_MAT_CONT` flag.
pub fn cv_dft(
    data: &mut [f64],
    rows: i32,
    cols: i32,
    step: i32,
    cont: bool,
    flags: i32,
    mut nonzero_rows: i32,
) {
    let mut prev_len = 0;
    let mut stage = 0;
    let mut nf = 0;
    let inv = (flags & CV_DXT_INVERSE) != 0;
    let real_transform = true;
    let complex_elem_size = 2usize; // in doubles
    let elem_size = 1usize;
    let mut factors = [0i32; 34];
    let mut inplace_transform;
    let dstep = (step / 8) as usize;
    let src_is_dst = true;

    // check types and sizes: the source and destination are the same
    // `CV_64FC1` array, so every check of `cvDFT` passes and
    // `real_transform` is set.

    if cols == 1 && nonzero_rows > 0 {
        cv_error(
            CV_STS_NOT_IMPLEMENTED,
            "cvDFT",
            "This mode (using nonzero_rows with a single-column matrix) breaks the function logic, so it is prohibited.\nFor fast convolution/correlation use 2-column matrix or single-row matrix instead",
            "cxdxt.cpp",
            1861,
        );
    }

    // determine, which transform to do first - row-wise
    // (stage 0) or column-wise (stage 1) transform
    if (flags & CV_DXT_ROWS) == 0
        && rows > 1
        && ((cols == 1 && !cont) || (cols > 1 && inv && real_transform))
    {
        stage = 1;
    }

    loop {
        let mut scale = 1.;
        let len;
        let count;
        let mut use_buf = false;
        let mut odd_real = false;

        if stage == 0 {
            // row-wise transform
            let mut l = cols;
            let mut c = rows;
            if l == 1 && (flags & CV_DXT_ROWS) == 0 {
                l = rows;
                c = 1;
            }
            len = l;
            count = c;
            odd_real = real_transform && (len & 1) != 0;
        } else {
            len = rows;
            count = cols;
        }

        if len != prev_len {
            nf = icv_dft_factorize(len, &mut factors);
        }
        prev_len = 0; // the C never records `prev_len`: tables are rebuilt every stage

        inplace_transform = factors[0] == factors[(nf - 1) as usize];
        let i = (nf > 1 && (factors[0] & 1) == 0) as usize;
        let radix_buf_len = if (factors[i] & 1) != 0 && factors[i] > 5 {
            (factors[i] + 1) as usize
        } else {
            0
        };

        if (stage == 0 && ((src_is_dst && !inplace_transform) || odd_real))
            || (stage == 1 && !inplace_transform)
        {
            use_buf = true;
        }

        let lenu = len as usize;
        let mut wave = vec![CvComplex64f::default(); lenu];
        let mut itab = vec![0i32; lenu];
        let mut rbuf = vec![CvComplex64f::default(); radix_buf_len];

        icv_dft_init(
            len,
            nf,
            &factors,
            &mut itab,
            &mut wave,
            stage == 0 && inv && real_transform,
        );

        if stage == 0 {
            let mut tmp_buf: Vec<f64> = Vec::new();
            let mut dptr_offset = 0usize;
            let dst_full_len = lenu * elem_size;
            let _flags = inv as i32;
            if use_buf {
                tmp_buf = vec![0.; lenu * complex_elem_size];
                if odd_real && !inv && len > 1 {
                    dptr_offset = elem_size;
                }
            }

            let dft_func = if !inv {
                DftFunc::Real
            } else {
                DftFunc::CcsInverse
            };

            if count > 1 && (flags & CV_DXT_ROWS) == 0 && (!inv || !real_transform) {
                stage = 1;
            } else if (flags & CV_DXT_SCALE) != 0 {
                scale = 1. / (len * if (flags & CV_DXT_ROWS) != 0 { 1 } else { count }) as f64;
            }

            if nonzero_rows <= 0 || nonzero_rows > count {
                nonzero_rows = count;
            }

            let mut i = 0;
            while i < nonzero_rows {
                let row = i as usize * dstep;
                if use_buf {
                    call_dft(
                        dft_func,
                        Some(&data[row..row + lenu]),
                        &mut tmp_buf,
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        _flags,
                        scale,
                    );
                    data[row..row + dst_full_len]
                        .copy_from_slice(&tmp_buf[dptr_offset..dptr_offset + dst_full_len]);
                } else {
                    call_dft(
                        dft_func,
                        None,
                        &mut data[row..row + lenu],
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        _flags,
                        scale,
                    );
                }
                i += 1;
            }

            while i < count {
                let row = i as usize * dstep;
                data[row..row + dst_full_len].fill(0.);
                i += 1;
            }

            if stage != 1 {
                break;
            }
        } else {
            let mut a = 0;
            let mut b = count;
            let lc = lenu * complex_elem_size;
            let mut buf0 = vec![0.; lc];
            let mut buf1 = vec![0.; lc];
            // `dbuf1 = ptr` when `use_buf` (and `dbuf0 = buf1`)
            let mut buf2 = vec![0.; if use_buf { lc } else { 0 }];
            let mut sptr0 = 0usize;
            let mut dptr0 = 0usize;

            let dft_func = DftFunc::Complex;

            if real_transform && inv && cols > 1 {
                stage = 0;
            } else if (flags & CV_DXT_SCALE) != 0 {
                scale = 1. / (len * count) as f64;
            }

            // `dft_func( X, dbufK, ... )`: out of place into buf1/buf2 when
            // `use_buf`, else in place.
            let flags_inv = inv as i32;

            if real_transform {
                a = 1;
                let even = (count & 1) == 0;
                b = (count + 1) / 2;
                if !inv {
                    buf0.fill(0.);
                    icv_copy_column(
                        &data[sptr0..],
                        dstep,
                        &mut buf0,
                        complex_elem_size,
                        lenu,
                        elem_size,
                    );
                    sptr0 += elem_size; // `CV_MAT_CN(dst->type)*elem_size`
                    if even {
                        buf1.fill(0.);
                        icv_copy_column(
                            &data[sptr0 + (count - 2) as usize * elem_size..],
                            dstep,
                            &mut buf1,
                            complex_elem_size,
                            lenu,
                            elem_size,
                        );
                    }
                } else {
                    icv_copy_column(
                        &data[sptr0..],
                        dstep,
                        &mut buf0[1..],
                        elem_size,
                        lenu,
                        elem_size,
                    );
                    icv_expand_ccs(&mut buf0, lenu);
                    if even {
                        icv_copy_column(
                            &data[sptr0 + (count - 1) as usize * elem_size..],
                            dstep,
                            &mut buf1[1..],
                            elem_size,
                            lenu,
                            elem_size,
                        );
                        icv_expand_ccs(&mut buf1, lenu);
                    }
                    sptr0 += elem_size;
                }

                // `dbuf0`/`dbuf1`: with `use_buf`, dbuf0 is buf1 and dbuf1
                // is buf2; otherwise each transform is in place.
                if use_buf {
                    if even {
                        call_dft(
                            dft_func,
                            Some(&buf1),
                            &mut buf2,
                            len,
                            nf,
                            &mut factors,
                            &itab,
                            &wave,
                            &mut rbuf,
                            flags_inv,
                            scale,
                        );
                    }
                    call_dft(
                        dft_func,
                        Some(&buf0),
                        &mut buf1,
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        flags_inv,
                        scale,
                    );
                } else {
                    if even {
                        call_dft(
                            dft_func,
                            None,
                            &mut buf1,
                            len,
                            nf,
                            &mut factors,
                            &itab,
                            &wave,
                            &mut rbuf,
                            flags_inv,
                            scale,
                        );
                    }
                    call_dft(
                        dft_func,
                        None,
                        &mut buf0,
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        flags_inv,
                        scale,
                    );
                }
                let (dbuf0, dbuf1): (&mut Vec<f64>, &mut Vec<f64>) = if use_buf {
                    (&mut buf1, &mut buf2)
                } else {
                    (&mut buf0, &mut buf1)
                };

                // `CV_MAT_CN(dst->type) == 1`
                if !inv {
                    // copy the half of output vector to the first/last column.
                    // before doing that, defgragment the vector
                    dbuf0[1] = dbuf0[0];
                    icv_copy_column(
                        &dbuf0[1..],
                        elem_size,
                        &mut data[dptr0..],
                        dstep,
                        lenu,
                        elem_size,
                    );
                    if even {
                        dbuf1[1] = dbuf1[0];
                        icv_copy_column(
                            &dbuf1[1..],
                            elem_size,
                            &mut data[dptr0 + (count - 1) as usize * elem_size..],
                            dstep,
                            lenu,
                            elem_size,
                        );
                    }
                    dptr0 += elem_size;
                } else {
                    // copy the real part of the complex vector to the first/last column
                    icv_copy_column(
                        dbuf0,
                        complex_elem_size,
                        &mut data[dptr0..],
                        dstep,
                        lenu,
                        elem_size,
                    );
                    if even {
                        icv_copy_column(
                            dbuf1,
                            complex_elem_size,
                            &mut data[dptr0 + (count - 1) as usize * elem_size..],
                            dstep,
                            lenu,
                            elem_size,
                        );
                    }
                    dptr0 += elem_size;
                }
            }

            let mut i = a;
            while i < b {
                if i + 1 < b {
                    icv_copy_from_2_columns(
                        &data[sptr0..],
                        dstep,
                        &mut buf0,
                        &mut buf1,
                        lenu,
                        complex_elem_size,
                    );
                    if use_buf {
                        call_dft(
                            dft_func,
                            Some(&buf1),
                            &mut buf2,
                            len,
                            nf,
                            &mut factors,
                            &itab,
                            &wave,
                            &mut rbuf,
                            flags_inv,
                            scale,
                        );
                    } else {
                        call_dft(
                            dft_func,
                            None,
                            &mut buf1,
                            len,
                            nf,
                            &mut factors,
                            &itab,
                            &wave,
                            &mut rbuf,
                            flags_inv,
                            scale,
                        );
                    }
                } else {
                    icv_copy_column(
                        &data[sptr0..],
                        dstep,
                        &mut buf0,
                        complex_elem_size,
                        lenu,
                        complex_elem_size,
                    );
                }

                if use_buf {
                    call_dft(
                        dft_func,
                        Some(&buf0),
                        &mut buf1,
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        flags_inv,
                        scale,
                    );
                } else {
                    call_dft(
                        dft_func,
                        None,
                        &mut buf0,
                        len,
                        nf,
                        &mut factors,
                        &itab,
                        &wave,
                        &mut rbuf,
                        flags_inv,
                        scale,
                    );
                }

                let (dbuf0, dbuf1): (&Vec<f64>, &Vec<f64>) = if use_buf {
                    (&buf1, &buf2)
                } else {
                    (&buf0, &buf1)
                };
                if i + 1 < b {
                    icv_copy_to_2_columns(
                        dbuf0,
                        dbuf1,
                        &mut data[dptr0..],
                        dstep,
                        lenu,
                        complex_elem_size,
                    );
                } else {
                    icv_copy_column(
                        dbuf0,
                        complex_elem_size,
                        &mut data[dptr0..],
                        dstep,
                        lenu,
                        complex_elem_size,
                    );
                }
                sptr0 += 2 * complex_elem_size;
                dptr0 += 2 * complex_elem_size;
                i += 2;
            }

            if stage != 0 {
                break;
            }
        }
    }
}

/// `cvMulSpectrums( const CvArr* srcAarr, const CvArr* srcBarr, CvArr*
/// dstarr, int flags )` (`cxdxt.cpp:2225`), for the case RAPTOR reaches:
/// `CV_64FC1` arrays of equal size where the destination is the first
/// source (`a_and_dst`, step `step_a` bytes) and the second is `b` (step
/// `step_b` bytes).  `cont_all` is `CV_IS_MAT_CONT( srcA->type &
/// srcB->type & dst->type )`.
#[allow(clippy::too_many_arguments)]
pub fn cv_mul_spectrums(
    a_and_dst: &mut [f64],
    step_a: i32,
    b: &[f64],
    step_b: i32,
    rows: i32,
    cols: i32,
    cont_all: bool,
    flags: i32,
) {
    let cn = 1;
    let mut rows = rows;
    let mut cols = cols;
    let is_1d = (flags & CV_DXT_ROWS) != 0 || (rows == 1 || (cols == 1 && cont_all));

    if is_1d && (flags & CV_DXT_ROWS) == 0 {
        cols = cols + rows - 1;
        rows = 1;
    }
    let ncols = cols * cn;
    let j0 = (cn == 1) as i32;
    let j1 = ncols - ((cols % 2 == 0 && cn == 1) as i32);

    let step_a = (step_a / 8) as usize;
    let step_b = (step_b / 8) as usize;
    let step_c = step_a;
    // `dataC` is `dataA`: every element pair is read before it is written.
    let mut da = 0usize;
    let mut db = 0usize;

    if !is_1d && cn == 1 {
        let kmax = if cols % 2 != 0 { 1 } else { 2 };
        for k in 0..kmax {
            if k == 1 {
                da += (cols - 1) as usize;
                db += (cols - 1) as usize;
            }
            let ru = rows as usize;
            a_and_dst[da] = a_and_dst[da] * b[db];
            if rows % 2 == 0 {
                a_and_dst[da + (ru - 1) * step_c] =
                    a_and_dst[da + (ru - 1) * step_a] * b[db + (ru - 1) * step_b];
            }
            let mut j = 1usize;
            if (flags & CV_DXT_MUL_CONJ) == 0 {
                while j as i32 <= rows - 2 {
                    let re = a_and_dst[da + j * step_a] * b[db + j * step_b]
                        - a_and_dst[da + (j + 1) * step_a] * b[db + (j + 1) * step_b];
                    let im = a_and_dst[da + j * step_a] * b[db + (j + 1) * step_b]
                        + a_and_dst[da + (j + 1) * step_a] * b[db + j * step_b];
                    a_and_dst[da + j * step_c] = re;
                    a_and_dst[da + (j + 1) * step_c] = im;
                    j += 2;
                }
            } else {
                while j as i32 <= rows - 2 {
                    let re = a_and_dst[da + j * step_a] * b[db + j * step_b]
                        + a_and_dst[da + (j + 1) * step_a] * b[db + (j + 1) * step_b];
                    let im = a_and_dst[da + (j + 1) * step_a] * b[db + j * step_b]
                        - a_and_dst[da + j * step_a] * b[db + (j + 1) * step_b];
                    a_and_dst[da + j * step_c] = re;
                    a_and_dst[da + (j + 1) * step_c] = im;
                    j += 2;
                }
            }
            if k == 1 {
                da -= (cols - 1) as usize;
                db -= (cols - 1) as usize;
            }
        }
    }

    for _ in 0..rows {
        if is_1d && cn == 1 {
            a_and_dst[da] = a_and_dst[da] * b[db];
            if cols % 2 == 0 {
                let j1u = j1 as usize;
                a_and_dst[da + j1u] = a_and_dst[da + j1u] * b[db + j1u];
            }
        }

        let mut j = j0 as usize;
        if (flags & CV_DXT_MUL_CONJ) == 0 {
            while (j as i32) < j1 {
                let re = a_and_dst[da + j] * b[db + j] - a_and_dst[da + j + 1] * b[db + j + 1];
                let im = a_and_dst[da + j + 1] * b[db + j] + a_and_dst[da + j] * b[db + j + 1];
                a_and_dst[da + j] = re;
                a_and_dst[da + j + 1] = im;
                j += 2;
            }
        } else {
            while (j as i32) < j1 {
                let re = a_and_dst[da + j] * b[db + j] + a_and_dst[da + j + 1] * b[db + j + 1];
                let im = a_and_dst[da + j + 1] * b[db + j] - a_and_dst[da + j] * b[db + j + 1];
                a_and_dst[da + j] = re;
                a_and_dst[da + j + 1] = im;
                j += 2;
            }
        }
        da += step_a;
        db += step_b;
    }
}

/// `icvOptimalDFTSize` (`cxdxt.cpp:2820`).
const ICV_OPTIMAL_DFT_SIZE: [i32; 1651] = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60,
    64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200,
    216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480,
    486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960,
    972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
    1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430,
    2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750,
    3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 5625,
    5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776,
    8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000, 10125, 10240, 10368, 10800,
    10935, 11250, 11520, 11664, 12000, 12150, 12288, 12500, 12800, 12960, 13122, 13500, 13824,
    14400, 14580, 15000, 15360, 15552, 15625, 16000, 16200, 16384, 16875, 17280, 17496, 18000,
    18225, 18432, 18750, 19200, 19440, 19683, 20000, 20250, 20480, 20736, 21600, 21870, 22500,
    23040, 23328, 24000, 24300, 24576, 25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800,
    29160, 30000, 30375, 30720, 31104, 31250, 32000, 32400, 32768, 32805, 33750, 34560, 34992,
    36000, 36450, 36864, 37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740,
    45000, 46080, 46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000,
    54675, 55296, 56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800,
    65536, 65610, 67500, 69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760, 78125, 78732,
    80000, 81000, 81920, 82944, 84375, 86400, 87480, 90000, 91125, 92160, 93312, 93750, 96000,
    97200, 98304, 98415, 100000, 101250, 102400, 103680, 104976, 108000, 109350, 110592, 112500,
    115200, 116640, 118098, 120000, 121500, 122880, 124416, 125000, 128000, 129600, 131072, 131220,
    135000, 138240, 139968, 140625, 144000, 145800, 147456, 150000, 151875, 153600, 155520, 156250,
    157464, 160000, 162000, 163840, 164025, 165888, 168750, 172800, 174960, 177147, 180000, 182250,
    184320, 186624, 187500, 192000, 194400, 196608, 196830, 200000, 202500, 204800, 207360, 209952,
    216000, 218700, 221184, 225000, 230400, 233280, 234375, 236196, 240000, 243000, 245760, 248832,
    250000, 253125, 256000, 259200, 262144, 262440, 270000, 273375, 276480, 279936, 281250, 288000,
    291600, 294912, 295245, 300000, 303750, 307200, 311040, 312500, 314928, 320000, 324000, 327680,
    328050, 331776, 337500, 345600, 349920, 354294, 360000, 364500, 368640, 373248, 375000, 384000,
    388800, 390625, 393216, 393660, 400000, 405000, 409600, 414720, 419904, 421875, 432000, 437400,
    442368, 450000, 455625, 460800, 466560, 468750, 472392, 480000, 486000, 491520, 492075, 497664,
    500000, 506250, 512000, 518400, 524288, 524880, 531441, 540000, 546750, 552960, 559872, 562500,
    576000, 583200, 589824, 590490, 600000, 607500, 614400, 622080, 625000, 629856, 640000, 648000,
    655360, 656100, 663552, 675000, 691200, 699840, 703125, 708588, 720000, 729000, 737280, 746496,
    750000, 759375, 768000, 777600, 781250, 786432, 787320, 800000, 810000, 819200, 820125, 829440,
    839808, 843750, 864000, 874800, 884736, 885735, 900000, 911250, 921600, 933120, 937500, 944784,
    960000, 972000, 983040, 984150, 995328, 1000000, 1012500, 1024000, 1036800, 1048576, 1049760,
    1062882, 1080000, 1093500, 1105920, 1119744, 1125000, 1152000, 1166400, 1171875, 1179648,
    1180980, 1200000, 1215000, 1228800, 1244160, 1250000, 1259712, 1265625, 1280000, 1296000,
    1310720, 1312200, 1327104, 1350000, 1366875, 1382400, 1399680, 1406250, 1417176, 1440000,
    1458000, 1474560, 1476225, 1492992, 1500000, 1518750, 1536000, 1555200, 1562500, 1572864,
    1574640, 1594323, 1600000, 1620000, 1638400, 1640250, 1658880, 1679616, 1687500, 1728000,
    1749600, 1769472, 1771470, 1800000, 1822500, 1843200, 1866240, 1875000, 1889568, 1920000,
    1944000, 1953125, 1966080, 1968300, 1990656, 2000000, 2025000, 2048000, 2073600, 2097152,
    2099520, 2109375, 2125764, 2160000, 2187000, 2211840, 2239488, 2250000, 2278125, 2304000,
    2332800, 2343750, 2359296, 2361960, 2400000, 2430000, 2457600, 2460375, 2488320, 2500000,
    2519424, 2531250, 2560000, 2592000, 2621440, 2624400, 2654208, 2657205, 2700000, 2733750,
    2764800, 2799360, 2812500, 2834352, 2880000, 2916000, 2949120, 2952450, 2985984, 3000000,
    3037500, 3072000, 3110400, 3125000, 3145728, 3149280, 3188646, 3200000, 3240000, 3276800,
    3280500, 3317760, 3359232, 3375000, 3456000, 3499200, 3515625, 3538944, 3542940, 3600000,
    3645000, 3686400, 3732480, 3750000, 3779136, 3796875, 3840000, 3888000, 3906250, 3932160,
    3936600, 3981312, 4000000, 4050000, 4096000, 4100625, 4147200, 4194304, 4199040, 4218750,
    4251528, 4320000, 4374000, 4423680, 4428675, 4478976, 4500000, 4556250, 4608000, 4665600,
    4687500, 4718592, 4723920, 4782969, 4800000, 4860000, 4915200, 4920750, 4976640, 5000000,
    5038848, 5062500, 5120000, 5184000, 5242880, 5248800, 5308416, 5314410, 5400000, 5467500,
    5529600, 5598720, 5625000, 5668704, 5760000, 5832000, 5859375, 5898240, 5904900, 5971968,
    6000000, 6075000, 6144000, 6220800, 6250000, 6291456, 6298560, 6328125, 6377292, 6400000,
    6480000, 6553600, 6561000, 6635520, 6718464, 6750000, 6834375, 6912000, 6998400, 7031250,
    7077888, 7085880, 7200000, 7290000, 7372800, 7381125, 7464960, 7500000, 7558272, 7593750,
    7680000, 7776000, 7812500, 7864320, 7873200, 7962624, 7971615, 8000000, 8100000, 8192000,
    8201250, 8294400, 8388608, 8398080, 8437500, 8503056, 8640000, 8748000, 8847360, 8857350,
    8957952, 9000000, 9112500, 9216000, 9331200, 9375000, 9437184, 9447840, 9565938, 9600000,
    9720000, 9765625, 9830400, 9841500, 9953280, 10000000, 10077696, 10125000, 10240000, 10368000,
    10485760, 10497600, 10546875, 10616832, 10628820, 10800000, 10935000, 11059200, 11197440,
    11250000, 11337408, 11390625, 11520000, 11664000, 11718750, 11796480, 11809800, 11943936,
    12000000, 12150000, 12288000, 12301875, 12441600, 12500000, 12582912, 12597120, 12656250,
    12754584, 12800000, 12960000, 13107200, 13122000, 13271040, 13286025, 13436928, 13500000,
    13668750, 13824000, 13996800, 14062500, 14155776, 14171760, 14400000, 14580000, 14745600,
    14762250, 14929920, 15000000, 15116544, 15187500, 15360000, 15552000, 15625000, 15728640,
    15746400, 15925248, 15943230, 16000000, 16200000, 16384000, 16402500, 16588800, 16777216,
    16796160, 16875000, 17006112, 17280000, 17496000, 17578125, 17694720, 17714700, 17915904,
    18000000, 18225000, 18432000, 18662400, 18750000, 18874368, 18895680, 18984375, 19131876,
    19200000, 19440000, 19531250, 19660800, 19683000, 19906560, 20000000, 20155392, 20250000,
    20480000, 20503125, 20736000, 20971520, 20995200, 21093750, 21233664, 21257640, 21600000,
    21870000, 22118400, 22143375, 22394880, 22500000, 22674816, 22781250, 23040000, 23328000,
    23437500, 23592960, 23619600, 23887872, 23914845, 24000000, 24300000, 24576000, 24603750,
    24883200, 25000000, 25165824, 25194240, 25312500, 25509168, 25600000, 25920000, 26214400,
    26244000, 26542080, 26572050, 26873856, 27000000, 27337500, 27648000, 27993600, 28125000,
    28311552, 28343520, 28800000, 29160000, 29296875, 29491200, 29524500, 29859840, 30000000,
    30233088, 30375000, 30720000, 31104000, 31250000, 31457280, 31492800, 31640625, 31850496,
    31886460, 32000000, 32400000, 32768000, 32805000, 33177600, 33554432, 33592320, 33750000,
    34012224, 34171875, 34560000, 34992000, 35156250, 35389440, 35429400, 35831808, 36000000,
    36450000, 36864000, 36905625, 37324800, 37500000, 37748736, 37791360, 37968750, 38263752,
    38400000, 38880000, 39062500, 39321600, 39366000, 39813120, 39858075, 40000000, 40310784,
    40500000, 40960000, 41006250, 41472000, 41943040, 41990400, 42187500, 42467328, 42515280,
    43200000, 43740000, 44236800, 44286750, 44789760, 45000000, 45349632, 45562500, 46080000,
    46656000, 46875000, 47185920, 47239200, 47775744, 47829690, 48000000, 48600000, 48828125,
    49152000, 49207500, 49766400, 50000000, 50331648, 50388480, 50625000, 51018336, 51200000,
    51840000, 52428800, 52488000, 52734375, 53084160, 53144100, 53747712, 54000000, 54675000,
    55296000, 55987200, 56250000, 56623104, 56687040, 56953125, 57600000, 58320000, 58593750,
    58982400, 59049000, 59719680, 60000000, 60466176, 60750000, 61440000, 61509375, 62208000,
    62500000, 62914560, 62985600, 63281250, 63700992, 63772920, 64000000, 64800000, 65536000,
    65610000, 66355200, 66430125, 67108864, 67184640, 67500000, 68024448, 68343750, 69120000,
    69984000, 70312500, 70778880, 70858800, 71663616, 72000000, 72900000, 73728000, 73811250,
    74649600, 75000000, 75497472, 75582720, 75937500, 76527504, 76800000, 77760000, 78125000,
    78643200, 78732000, 79626240, 79716150, 80000000, 80621568, 81000000, 81920000, 82012500,
    82944000, 83886080, 83980800, 84375000, 84934656, 85030560, 86400000, 87480000, 87890625,
    88473600, 88573500, 89579520, 90000000, 90699264, 91125000, 92160000, 93312000, 93750000,
    94371840, 94478400, 94921875, 95551488, 95659380, 96000000, 97200000, 97656250, 98304000,
    98415000, 99532800, 100000000, 100663296, 100776960, 101250000, 102036672, 102400000,
    102515625, 103680000, 104857600, 104976000, 105468750, 106168320, 106288200, 107495424,
    108000000, 109350000, 110592000, 110716875, 111974400, 112500000, 113246208, 113374080,
    113906250, 115200000, 116640000, 117187500, 117964800, 118098000, 119439360, 119574225,
    120000000, 120932352, 121500000, 122880000, 123018750, 124416000, 125000000, 125829120,
    125971200, 126562500, 127401984, 127545840, 128000000, 129600000, 131072000, 131220000,
    132710400, 132860250, 134217728, 134369280, 135000000, 136048896, 136687500, 138240000,
    139968000, 140625000, 141557760, 141717600, 143327232, 144000000, 145800000, 146484375,
    147456000, 147622500, 149299200, 150000000, 150994944, 151165440, 151875000, 153055008,
    153600000, 155520000, 156250000, 157286400, 157464000, 158203125, 159252480, 159432300,
    160000000, 161243136, 162000000, 163840000, 164025000, 165888000, 167772160, 167961600,
    168750000, 169869312, 170061120, 170859375, 172800000, 174960000, 175781250, 176947200,
    177147000, 179159040, 180000000, 181398528, 182250000, 184320000, 184528125, 186624000,
    187500000, 188743680, 188956800, 189843750, 191102976, 191318760, 192000000, 194400000,
    195312500, 196608000, 196830000, 199065600, 199290375, 200000000, 201326592, 201553920,
    202500000, 204073344, 204800000, 205031250, 207360000, 209715200, 209952000, 210937500,
    212336640, 212576400, 214990848, 216000000, 218700000, 221184000, 221433750, 223948800,
    225000000, 226492416, 226748160, 227812500, 230400000, 233280000, 234375000, 235929600,
    236196000, 238878720, 239148450, 240000000, 241864704, 243000000, 244140625, 245760000,
    246037500, 248832000, 250000000, 251658240, 251942400, 253125000, 254803968, 255091680,
    256000000, 259200000, 262144000, 262440000, 263671875, 265420800, 265720500, 268435456,
    268738560, 270000000, 272097792, 273375000, 276480000, 279936000, 281250000, 283115520,
    283435200, 284765625, 286654464, 288000000, 291600000, 292968750, 294912000, 295245000,
    298598400, 300000000, 301989888, 302330880, 303750000, 306110016, 307200000, 307546875,
    311040000, 312500000, 314572800, 314928000, 316406250, 318504960, 318864600, 320000000,
    322486272, 324000000, 327680000, 328050000, 331776000, 332150625, 335544320, 335923200,
    337500000, 339738624, 340122240, 341718750, 345600000, 349920000, 351562500, 353894400,
    354294000, 358318080, 360000000, 362797056, 364500000, 368640000, 369056250, 373248000,
    375000000, 377487360, 377913600, 379687500, 382205952, 382637520, 384000000, 388800000,
    390625000, 393216000, 393660000, 398131200, 398580750, 400000000, 402653184, 403107840,
    405000000, 408146688, 409600000, 410062500, 414720000, 419430400, 419904000, 421875000,
    424673280, 425152800, 429981696, 432000000, 437400000, 439453125, 442368000, 442867500,
    447897600, 450000000, 452984832, 453496320, 455625000, 460800000, 466560000, 468750000,
    471859200, 472392000, 474609375, 477757440, 478296900, 480000000, 483729408, 486000000,
    488281250, 491520000, 492075000, 497664000, 500000000, 503316480, 503884800, 506250000,
    509607936, 510183360, 512000000, 512578125, 518400000, 524288000, 524880000, 527343750,
    530841600, 531441000, 536870912, 537477120, 540000000, 544195584, 546750000, 552960000,
    553584375, 559872000, 562500000, 566231040, 566870400, 569531250, 573308928, 576000000,
    583200000, 585937500, 589824000, 590490000, 597196800, 597871125, 600000000, 603979776,
    604661760, 607500000, 612220032, 614400000, 615093750, 622080000, 625000000, 629145600,
    629856000, 632812500, 637009920, 637729200, 640000000, 644972544, 648000000, 655360000,
    656100000, 663552000, 664301250, 671088640, 671846400, 675000000, 679477248, 680244480,
    683437500, 691200000, 699840000, 703125000, 707788800, 708588000, 716636160, 720000000,
    725594112, 729000000, 732421875, 737280000, 738112500, 746496000, 750000000, 754974720,
    755827200, 759375000, 764411904, 765275040, 768000000, 777600000, 781250000, 786432000,
    787320000, 791015625, 796262400, 797161500, 800000000, 805306368, 806215680, 810000000,
    816293376, 819200000, 820125000, 829440000, 838860800, 839808000, 843750000, 849346560,
    850305600, 854296875, 859963392, 864000000, 874800000, 878906250, 884736000, 885735000,
    895795200, 900000000, 905969664, 906992640, 911250000, 921600000, 922640625, 933120000,
    937500000, 943718400, 944784000, 949218750, 955514880, 956593800, 960000000, 967458816,
    972000000, 976562500, 983040000, 984150000, 995328000, 996451875, 1000000000, 1006632960,
    1007769600, 1012500000, 1019215872, 1020366720, 1024000000, 1025156250, 1036800000, 1048576000,
    1049760000, 1054687500, 1061683200, 1062882000, 1073741824, 1074954240, 1080000000, 1088391168,
    1093500000, 1105920000, 1107168750, 1119744000, 1125000000, 1132462080, 1133740800, 1139062500,
    1146617856, 1152000000, 1166400000, 1171875000, 1179648000, 1180980000, 1194393600, 1195742250,
    1200000000, 1207959552, 1209323520, 1215000000, 1220703125, 1224440064, 1228800000, 1230187500,
    1244160000, 1250000000, 1258291200, 1259712000, 1265625000, 1274019840, 1275458400, 1280000000,
    1289945088, 1296000000, 1310720000, 1312200000, 1318359375, 1327104000, 1328602500, 1342177280,
    1343692800, 1350000000, 1358954496, 1360488960, 1366875000, 1382400000, 1399680000, 1406250000,
    1415577600, 1417176000, 1423828125, 1433272320, 1440000000, 1451188224, 1458000000, 1464843750,
    1474560000, 1476225000, 1492992000, 1500000000, 1509949440, 1511654400, 1518750000, 1528823808,
    1530550080, 1536000000, 1537734375, 1555200000, 1562500000, 1572864000, 1574640000, 1582031250,
    1592524800, 1594323000, 1600000000, 1610612736, 1612431360, 1620000000, 1632586752, 1638400000,
    1640250000, 1658880000, 1660753125, 1677721600, 1679616000, 1687500000, 1698693120, 1700611200,
    1708593750, 1719926784, 1728000000, 1749600000, 1757812500, 1769472000, 1771470000, 1791590400,
    1800000000, 1811939328, 1813985280, 1822500000, 1843200000, 1845281250, 1866240000, 1875000000,
    1887436800, 1889568000, 1898437500, 1911029760, 1913187600, 1920000000, 1934917632, 1944000000,
    1953125000, 1966080000, 1968300000, 1990656000, 1992903750, 2000000000, 2013265920, 2015539200,
    2025000000, 2038431744, 2040733440, 2048000000, 2050312500, 2073600000, 2097152000, 2099520000,
    2109375000, 2123366400, 2125764000,
];

/// `cvGetOptimalDFTSize( int size0 )` (`cxdxt.cpp:3003`).
pub fn cv_get_optimal_dft_size(size0: i32) -> i32 {
    let mut a = 0usize;
    let mut b = ICV_OPTIMAL_DFT_SIZE.len() - 1;
    if (size0 as u32) >= (ICV_OPTIMAL_DFT_SIZE[b] as u32) {
        return -1;
    }

    while a < b {
        let c = (a + b) >> 1;
        if size0 <= ICV_OPTIMAL_DFT_SIZE[c] {
            b = c;
        } else {
            a = c + 1;
        }
    }

    ICV_OPTIMAL_DFT_SIZE[b]
}
