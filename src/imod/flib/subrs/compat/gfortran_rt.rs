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
//! `tand` is transcribed the same way ([`gfortran_tand_r4`], for
//! `combinefft`'s `getTilts`).
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

/// x86 `MAXSS dest, src` for a gfortran `MAX` of two `real*4` values.
///
/// gfortran 11 lowers `MAX`/`MIN` of reals to `MAX_EXPR`/`MIN_EXPR`
/// ("whatever is fastest", `trans-intrinsic.cc`), which the i386 back end
/// expands to the non-commutative `maxss`/`minss`: the result is `dest` only
/// when `dest > src` is true, so an unordered (NaN) comparison yields `src`.
/// *Which* Fortran argument ends up as `src` is a code-generation choice
/// (operand canonicalisation, register allocation), not a property of the
/// source, so every caller names the order it read from the reference
/// object's disassembly at that site.
pub fn maxss(dest: f32, src: f32) -> f32 {
    if dest > src { dest } else { src }
}

/// x86 `MINSS dest, src`: `dest` only when `dest < src`, otherwise (including
/// NaN in either operand) `src`.  See [`maxss`].
pub fn minss(dest: f32, src: f32) -> f32 {
    if dest < src { dest } else { src }
}

/// x86 `CVTTSS2SI`, gfortran's conversion of a `real*4` to `integer*4`
/// (`int(x)`, or an assignment to an integer): truncation toward zero, and
/// `i32::MIN` (the "integer indefinite") for NaN or an out-of-range value,
/// where Rust's `as i32` saturates and maps NaN to 0.
pub fn cvttss2si(x: f32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE is part of the x86-64 baseline.
        unsafe { core::arch::x86_64::_mm_cvttss_si32(core::arch::x86_64::_mm_set_ss(x)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x as i32
    }
}

/// x86 `MAXSD dest, src`, the `real*8` form of [`maxss`].
pub fn maxsd(dest: f64, src: f64) -> f64 {
    if dest > src { dest } else { src }
}

/// libgfortran `write_float` for the `Fw.d` edit descriptor of a `real*4`
/// (or a `real*8`, widened exactly).
///
/// Digits: for the default (unspecified) rounding mode libgfortran lets
/// `snprintf` round the exact binary value to `d` decimals, which is
/// round-to-nearest with ties to even on the exact expansion -- the same as
/// Rust's `{:.d$}`.  Checked against gfortran 11.4 `f7.1` over 200000 values,
/// 100000 of them exact decimal ties (`0.25` prints `0.2`, `0.75` `0.8`).
/// What differs from Rust's `{:w.d$}` is the field: the optional leading `0`
/// of a magnitude below 1 is dropped when it does not fit, a value that still
/// does not fit is `w` asterisks instead of a wider field, `d = 0` keeps the
/// decimal point, a negative value
/// that rounds to zero keeps its `-`, and NaN/infinity follow `write_infnan`
/// (`NaN`, `Infinity` from width 8 else `Inf`, `-Infinity` from 9 else
/// `-Inf`, asterisks when even that does not fit).
pub fn format_f(value: f64, w: usize, d: usize) -> String {
    if value.is_nan() {
        return if w >= 3 {
            format!("{:>w$}", "NaN")
        } else {
            "*".repeat(w)
        };
    }
    if value.is_infinite() {
        let text = match (value < 0.0, w) {
            (false, 8..) => "Infinity",
            (false, 3..) => "Inf",
            (true, 9..) => "-Infinity",
            (true, 4..) => "-Inf",
            _ => return "*".repeat(w),
        };
        return format!("{text:>w$}");
    }
    let mut text = format!("{:.*}", d, value);
    // `Fw.0` still writes the decimal point (`59.`, `0.`).
    if d == 0 {
        text.push('.');
    }
    if text.len() > w {
        if let Some(rest) = text.strip_prefix("0.") {
            text = format!(".{rest}");
        } else if let Some(rest) = text.strip_prefix("-0.") {
            text = format!("-.{rest}");
        }
    }
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// libgfortran `_gfortran_tand_r4`, the gfortran `TAND` intrinsic for
/// `real*4` (`combinefft.f90`'s `getTilts` declares `real*4 tand`, and the
/// reference `combinefft` binary imports this `GFORTRAN_10` symbol).
/// Transcribed from the disassembly of the `libgfortran.so.5` the reference
/// build loads, with the same `pi/180` pair as [`gfortran_sind_r4`]:
/// non-finite gives `x - x`; `|x| < 1/32` gives `fmaf(x, H, x * L)`;
/// otherwise the sign `s` is split off and `|x|` reduced with
/// `fmodf(., 360)`.  An exact multiple of 45 returns an exact value (`s * 0`
/// at 0 and 180, `s * inf` at 90, `s * -inf` at 270, `s` at 45 and 225,
/// `-s` at 135 and 315); anything else is folded into `[0, 90]` (`y`,
/// `180 - y` negated, `y - 180`, `360 - y` negated) and evaluated as
/// `s * tanf(fmaf(t, H, t * L))`.
pub fn gfortran_tand_r4(x: f32) -> f32 {
    let ax = x.abs();
    // `ucomiss |x|, FLT_MAX; jb`: unordered or larger than FLT_MAX.
    if !(ax <= f32::MAX) {
        return x - x;
    }
    if 0.0 > ax - 0.03125 {
        return x.mul_add(PIO180H, x * PIO180L);
    }
    let mut s = 1.0f32.copysign(x);
    let y = ax % 360.0;
    let n = y as i32;
    if n as f32 - y == 0.0 && n % 45 == 0 {
        if n % 180 == 0 {
            return s * 0.0;
        }
        if n % 90 == 0 {
            return if n == 90 {
                s * f32::INFINITY
            } else {
                s * f32::NEG_INFINITY
            };
        }
        return if n == 45 || n == 225 { s } else { -s };
    }
    let t;
    if 0.0 >= y - 180.0 {
        if y - 90.0 <= 0.0 {
            t = y;
        } else {
            t = 180.0 - y;
            s = -s;
        }
    } else if 0.0 >= y - 270.0 {
        t = y - 180.0;
    } else {
        t = 360.0 - y;
        s = -s;
    }
    let r = t.mul_add(PIO180H, t * PIO180L);
    s * r.tan()
}
