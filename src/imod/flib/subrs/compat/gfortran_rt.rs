//! Rust-only stand-ins for the gfortran runtime's degree-trigonometry
//! intrinsics, shared by every translated Fortran unit.
//!
//! This is boundary code, not a translation of an IMOD source unit.
//! `IMOD/flib/subrs/compat/degtrig.f` defines `SIND`/`COSD`/... as
//! `SIN(degrees * 0.01745329252)` for compilers that lack them, but with
//! gfortran >= 10 the names resolve to the intrinsics first: every reference
//! object that uses them (`nm flib/image/newstack.o`, `flib/blend/bsubs.o`,
//! `blendmont.o`, ...) carries undefined `_gfortran_sind_r4` /
//! `_gfortran_cosd_r4` (and `combinefft.o` `_gfortran_tand_r4`), the
//! `GFORTRAN_10` symbols of the `libgfortran.so.5` the reference loads
//! (gfortran 11.4, `-O3`).  Those library routines are what the native
//! programs compute, so they are transcribed here from its disassembly.
//! `asind`/`acosd`/`atand`/`atan2d` are *not* library calls: gfortran 11
//! inlines them as the radian function times a folded `f32` constant (see
//! `bsubs.rs`'s `ATAND_FACTOR`), so a caller reproduces them in place.
//! `tand` has no translated caller yet and is not transcribed.
//!
//! Verified bit-identical to the linked library over 11053 `real*4` inputs,
//! including zeros, infinities, NaN, multiples of 30 and values near the
//! reduction thresholds (`/big/henriksson/realbench/wave4-bsubs/trig`,
//! re-run from this module in `/big/henriksson/realbench/gfortran-trig`).

/// `pi/180` split into a high part with a short mantissa and a low part,
/// as `_gfortran_sind_r4`/`_gfortran_cosd_r4` hold it (`0x3c8f0000`,
/// `0xb6395dad` in `libgfortran.so.5`).
const PIO180H: f32 = f32::from_bits(0x3c8f0000);
const PIO180L: f32 = f32::from_bits(0xb6395dad);

/// libgfortran `_gfortran_sind_r4`, the gfortran `SIND` intrinsic for
/// `real*4` (the runtime every Fortran unit links, as `c_format` is for
/// `printf`).  Transcribed from the disassembly of the
/// `GFORTRAN_10` symbol in the `libgfortran.so.5` the reference build loads:
/// non-finite gives `x - x`; `|x| < 1/32` gives `fmaf(x, H, x * L)`;
/// otherwise the sign is split off, `|x|` is reduced with `fmodf(., 360)`,
/// exact multiples of 30 return exact values, and the rest is folded into
/// `[0, 45]` and evaluated as `sinf`/`cosf` of `fmaf(t, H, t * L)`.
pub fn gfortran_sind_r4(x: f32) -> f32 {
    let ax = x.abs();
    // `ucomiss |x|, FLT_MAX; jb`: unordered or larger than FLT_MAX.
    if !(ax <= f32::MAX) {
        return x - x;
    }
    if 0.0 > ax - 0.03125 {
        return x.mul_add(PIO180H, x * PIO180L);
    }
    let s = 1.0f32.copysign(x);
    let y = ax % 360.0;
    let n = y as i32;
    if n as f32 - y == 0.0 && n % 30 == 0 {
        if n % 180 == 0 {
            return if n == 180 { -s * 0.0 } else { s * 0.0 };
        }
        if n % 90 == 0 {
            return if n == 90 { s } else { -s };
        }
        if n % 60 == 0 {
            return if n > 179 {
                s * f32::from_bits(0xbf5db3d7)
            } else {
                s * f32::from_bits(0x3f5db3d7)
            };
        }
        return if n > 179 { s * -0.5 } else { s * 0.5 };
    }
    let (t, use_cos, sgn) = if 0.0 >= y - 180.0 {
        if 0.0 >= y - 90.0 {
            if y - 45.0 <= 0.0 {
                (y, false, s)
            } else {
                (90.0 - y, true, s)
            }
        } else if 0.0 >= y - 135.0 {
            (y - 90.0, true, s)
        } else {
            (180.0 - y, false, s)
        }
    } else if 0.0 >= y - 270.0 {
        if 0.0 >= y - 225.0 {
            (y - 180.0, false, -s)
        } else {
            (270.0 - y, true, -s)
        }
    } else if 0.0 >= y - 315.0 {
        (y - 270.0, true, -s)
    } else {
        (360.0 - y, false, -s)
    };
    let r = t.mul_add(PIO180H, t * PIO180L);
    sgn * if use_cos { r.cos() } else { r.sin() }
}

/// libgfortran `_gfortran_cosd_r4`, the gfortran `COSD` intrinsic for
/// `real*4`, transcribed from the same library as [`gfortran_sind_r4`]:
/// non-finite gives `x - x`; `|x| <= 1/128` gives `1` (for 0) or
/// `1 - 7.888609e-31`; otherwise `fmodf(|x|, 360)`, exact values at multiples
/// of 30, and `cosf`/`sinf` of `fmaf(t, H, t * L)` on the folded angle.
pub fn gfortran_cosd_r4(x: f32) -> f32 {
    let ax = x.abs();
    if !(ax <= f32::MAX) {
        return x - x;
    }
    if 0.0 >= ax - 0.0078125 {
        return if x == 0.0 {
            1.0
        } else {
            1.0 - f32::from_bits(0x0d800000)
        };
    }
    let y = ax % 360.0;
    let n = y as i32;
    if n as f32 - y == 0.0 && n % 30 == 0 {
        if n % 180 == 0 {
            return if n == 180 { -1.0 } else { 1.0 };
        }
        if n % 90 == 0 {
            return 0.0;
        }
        if n % 60 == 0 {
            return if n == 60 || n == 300 { 0.5 } else { -0.5 };
        }
        return if n == 30 || n == 330 {
            f32::from_bits(0x3f5db3d7)
        } else {
            f32::from_bits(0xbf5db3d7)
        };
    }
    let (t, use_sin, negate) = if 0.0 >= y - 180.0 {
        if 0.0 >= y - 90.0 {
            if y - 45.0 <= 0.0 {
                (y, false, false)
            } else {
                (90.0 - y, true, false)
            }
        } else if 0.0 >= y - 135.0 {
            (y - 90.0, true, true)
        } else {
            (180.0 - y, false, true)
        }
    } else if 0.0 >= y - 270.0 {
        if 0.0 >= y - 225.0 {
            (y - 180.0, false, true)
        } else {
            (270.0 - y, true, true)
        }
    } else if 0.0 >= y - 315.0 {
        (y - 270.0, true, false)
    } else {
        (360.0 - y, false, false)
    };
    let r = t.mul_add(PIO180H, t * PIO180L);
    let v = if use_sin { r.sin() } else { r.cos() };
    if negate { -v } else { v }
}
