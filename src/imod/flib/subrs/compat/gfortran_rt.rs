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
        if x >= -2147483648.0 && x < 2147483648.0 {
            x as i32
        } else {
            i32::MIN
        }
    }
}

/// x86 `CVTTSD2SI`, the `real*8`/`double` form of [`cvttss2si`]: truncation
/// toward zero, and `i32::MIN` for NaN or a value outside `int` range.  Off
/// x86_64 the scalar arm computes the same answer, so every target gives the
/// Linux x86_64 result (`PORTABILITY.md`).
pub fn cvttsd2si(x: f64) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86-64 baseline.
        unsafe { core::arch::x86_64::_mm_cvttsd_si32(core::arch::x86_64::_mm_set_sd(x)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if x > -2147483649.0 && x < 2147483648.0 {
            x as i32
        } else {
            i32::MIN
        }
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

/// gfortran `NINT` of a `real*4` to a default integer (`__builtin_iroundf`):
/// round half away from zero.  An out-of-range value or NaN gives the x86
/// "integer indefinite", as `lroundf` truncated to 32 bits does.
pub fn nint_r4(x: f32) -> i32 {
    let r = x.round();
    if r.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        i32::MIN
    } else {
        r as i32
    }
}

/// gfortran `NINT` of a `real*8` to a default integer (`IDNINT`).
pub fn nint_r8(x: f64) -> i32 {
    let r = x.round();
    if r.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        i32::MIN
    } else {
        r as i32
    }
}

/// libgcc `__powisf2`, which gfortran calls for `real*4 ** integer` with a
/// non-constant exponent: repeated squaring in single precision, and the
/// reciprocal for a negative exponent.
pub fn powi_r4(x: f32, m: i32) -> f32 {
    let mut n = m.unsigned_abs();
    let mut x = x;
    let mut y = if n % 2 == 1 { x } else { 1.0 };
    loop {
        n >>= 1;
        if n == 0 {
            break;
        }
        x *= x;
        if n % 2 == 1 {
            y *= x;
        }
    }
    if m < 0 { 1.0 / y } else { y }
}

/// libgfortran `write_i` for the `Iw` edit descriptor: right-justified in
/// `w` columns, `w` asterisks when the value does not fit.
pub fn format_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// A list-directed (`print *`, `write(*,*)`) `integer*4` item: a blank
/// separator (the record's leading blank when it is the first item) and
/// `I11`.
pub fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}

/// A list-directed `real*4` item (its blank separator included): `F` form
/// with nine significant digits and four trailing blanks for magnitudes in
/// [0.1, 1e9), `E` form with a two-digit exponent otherwise; `NaN` and
/// `Infinity` right-justified in the 17 columns.
pub fn ld_real(value: f32) -> String {
    if value.is_nan() {
        return format!("{:>17}", "NaN");
    }
    if value.is_infinite() {
        return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
    }
    if value == 0. {
        return format!("{:>13}    ", format!("{value:.8}"));
    }
    let scientific = format!("{:.8e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=9).contains(&k) {
        let mut text = format!("{:.*}", (9 - k) as usize, value);
        if k == 9 {
            text.push('.');
        }
        format!("{text:>13}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>17}",
            format!(
                "{}{}E{}{:02}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}

/// The libgfortran runtime error a failed `READ` with no `END=`/`ERR=`
/// branch ends in: the message on standard error and exit status 2.
pub fn read_runtime_error(error: crate::imod::flib::subrs::hvem::frefor::ListReadError) -> ! {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    match error {
        crate::imod::flib::subrs::hvem::frefor::ListReadError::End => {
            eprintln!("Fortran runtime error: End of file")
        }
        crate::imod::flib::subrs::hvem::frefor::ListReadError::Error => {
            eprintln!("Fortran runtime error: Bad value during list input")
        }
    }
    crate::imod::libcfshr::b3dutil::exit(2)
}

/// `READ(5, *) items` with no `END=`/`ERR=`: a list-directed read from
/// standard input, the runtime error on failure.  Pending output (a `$`
/// prompt) is flushed first, as the runtime does before reading a
/// preconnected terminal unit.
pub(crate) fn read_list_stdin(items: &mut [crate::imod::flib::subrs::hvem::frefor::ListItem]) {
    if let Err(error) = read_list_stdin_err(items) {
        read_runtime_error(error);
    }
}

/// `READ(5, *, ERR=...) items`: as [`read_list_stdin`], returning the
/// failure for the statement's branch.
pub(crate) fn read_list_stdin_err(
    items: &mut [crate::imod::flib::subrs::hvem::frefor::ListItem],
) -> Result<(), crate::imod::flib::subrs::hvem::frefor::ListReadError> {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    crate::imod::flib::subrs::hvem::frefor::list_read(&mut lock, items)
}

/// `READ(5, '(a)') var` into a `character*len`: one record, cut to `len`
/// bytes and blank-padded to `len`, as the variable holds it.  End of file
/// is the runtime error.
pub fn read_line_stdin(len: usize) -> Vec<u8> {
    use std::io::{BufRead as _, Write as _};
    let _ = std::io::stdout().flush();
    let mut line = Vec::new();
    match std::io::stdin().lock().read_until(b'\n', &mut line) {
        Ok(0) | Err(_) => {
            read_runtime_error(crate::imod::flib::subrs::hvem::frefor::ListReadError::End)
        }
        Ok(_) => {}
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    line.truncate(len);
    line.resize(len, b' ');
    line
}

/// Fortran `len_trim` of a blank-padded character variable.
pub fn len_trim(text: &[u8]) -> usize {
    text.iter().rposition(|&c| c != b' ').map_or(0, |p| p + 1)
}

/// Fortran `adjustl`: leading blanks moved to the end, same length.
pub fn adjustl(text: &[u8]) -> Vec<u8> {
    let start = text.iter().position(|&c| c != b' ').unwrap_or(text.len());
    let mut out = text[start..].to_vec();
    out.resize(text.len(), b' ');
    out
}

/// A character assignment `var = value` into a `character*len`: cut or
/// blank-padded to `len`.
pub fn fortran_assign(value: &[u8], len: usize) -> Vec<u8> {
    let mut out = value[..value.len().min(len)].to_vec();
    out.resize(len, b' ');
    out
}
