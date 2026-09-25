//! Translation of `IMOD/libcfshr/linearxforms.c`.
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use std::cell::Cell;
use std::io::Write;

/// Matches `xfUnit`.
pub fn xf_unit(matrix: &mut [f32], value: f32, rows: usize) {
    matrix[..3 * rows].fill(0.);
    matrix[0] = value;
    matrix[1 + rows] = value;
    if rows > 2 {
        matrix[2 * (1 + rows)] = 1.;
    }
}
/// Matches `xfunit`.
pub fn xfunit(matrix: &mut [f32], value: f32) {
    xf_unit(matrix, value, 2)
}
/// Matches `xfCopy`.
pub fn xf_copy(from: &[f32], from_rows: usize, to: &mut [f32], to_rows: usize) {
    if to_rows > 2 {
        xf_unit(to, 1., to_rows);
    }
    for column in 0..3 {
        for row in 0..2 {
            to[row + column * to_rows] = from[row + column * from_rows];
        }
    }
}
/// Matches `xfcopy`.
pub fn xfcopy(from: &[f32], to: &mut [f32]) {
    xf_copy(from, 2, to, 2)
}
/// Matches `xfMult`.
pub fn xf_mult(first: &[f32], second: &[f32], product: &mut [f32], rows: usize) {
    let mut temp = [0.; 9];
    let x = 2 * rows;
    let y = x + 1;
    temp[0] = second[0] * first[0] + second[rows] * first[1];
    temp[1] = second[1] * first[0] + second[rows + 1] * first[1];
    temp[rows] = second[0] * first[rows] + second[rows] * first[rows + 1];
    temp[rows + 1] = second[1] * first[rows] + second[rows + 1] * first[rows + 1];
    temp[x] = second[0] * first[x] + second[rows] * first[y] + second[x];
    temp[y] = second[1] * first[x] + second[rows + 1] * first[y] + second[y];
    if rows > 2 {
        temp[2] = 0.;
        temp[5] = 0.;
        temp[8] = 1.;
    }
    product[..3 * rows].copy_from_slice(&temp[..3 * rows]);
}
/// Matches `xfmult`.
pub fn xfmult(first: &[f32], second: &[f32], product: &mut [f32]) {
    xf_mult(first, second, product, 2)
}
/// Matches `xfInvert`.
pub fn xf_invert(matrix: &[f32], inverse: &mut [f32], rows: usize) {
    let mut temp = [0.; 9];
    let determinant = matrix[0] * matrix[rows + 1] - matrix[rows] * matrix[1];
    let x = 2 * rows;
    let y = x + 1;
    temp[0] = matrix[rows + 1] / determinant;
    temp[rows] = -matrix[rows] / determinant;
    temp[1] = -matrix[1] / determinant;
    temp[rows + 1] = matrix[0] / determinant;
    temp[x] = -(temp[0] * matrix[x] + temp[rows] * matrix[y]);
    temp[y] = -(temp[1] * matrix[x] + temp[rows + 1] * matrix[y]);
    if rows > 2 {
        temp[2] = 0.;
        temp[5] = 0.;
        temp[8] = 1.;
    }
    inverse[..3 * rows].copy_from_slice(&temp[..3 * rows]);
}
/// Matches `xfinvert`.
pub fn xfinvert(matrix: &[f32], inverse: &mut [f32]) {
    xf_invert(matrix, inverse, 2)
}
/// Matches `xfApply`.
pub fn xf_apply(
    matrix: &[f32],
    x_center: f32,
    y_center: f32,
    x: f32,
    y: f32,
    rows: usize,
) -> (f32, f32) {
    let xa = x - x_center;
    let ya = y - y_center;
    (
        matrix[0] * xa + matrix[rows] * ya + matrix[2 * rows] + x_center,
        matrix[1] * xa + matrix[rows + 1] * ya + matrix[2 * rows + 1] + y_center,
    )
}
/// Matches `xfapply`.
pub fn xfapply(matrix: &[f32], x_center: f32, y_center: f32, x: f32, y: f32) -> (f32, f32) {
    xf_apply(matrix, x_center, y_center, x, y, 2)
}
/// Matches `anglesToMatrix`.
pub fn angles_to_matrix(angles: &[f32; 3], matrix: &mut [f32], rows: usize) {
    let (a, b, g) = (
        angles[0] as f64 * 0.0174532921,
        angles[1] as f64 * 0.0174532921,
        angles[2] as f64 * 0.0174532921,
    );
    let (ca, cb, cg, sa, sb, sg) = (a.cos(), b.cos(), g.cos(), a.sin(), b.sin(), g.sin());
    matrix[0] = (cb * cg) as f32;
    matrix[rows] = (-cb * sg) as f32;
    matrix[2 * rows] = sb as f32;
    matrix[1] = (sa * sb * cg + ca * sg) as f32;
    matrix[1 + rows] = (-sa * sb * sg + ca * cg) as f32;
    matrix[1 + 2 * rows] = (-sa * cb) as f32;
    matrix[2] = (-ca * sb * cg + sa * sg) as f32;
    matrix[2 + rows] = (ca * sb * sg + sa * cg) as f32;
    matrix[2 + 2 * rows] = (ca * cb) as f32;
}
/// Matches `icalc_matrix`.
pub fn icalc_matrix(angles: &[f32; 3], matrix: &mut [f32]) {
    angles_to_matrix(angles, matrix, 3)
}
thread_local! {
    /// `linearxforms.c:226`: `double sDet;` is a file-scope global, not a local,
    /// and `icalc_angles` prints it after `matrixToAngles` has failed.  Keeping
    /// it a local would lose the value that failure path reports.  A
    /// thread-local `Cell` holds it without `static mut`: it is written and read
    /// inside a single `icalc_angles` call, so the narrower scope changes
    /// nothing about when it is set or what is printed.
    pub static S_DET: Cell<f64> = const { Cell::new(0.0) };
}
/// Matches `matrixToAngles`.
pub fn matrix_to_angles(matrix: &[f32], rows: usize) -> Result<(f64, f64, f64), ()> {
    let (r11, r12, r13, r21, r22, r23, r31, r32, r33) = (
        matrix[0] as f64,
        matrix[rows] as f64,
        matrix[2 * rows] as f64,
        matrix[1] as f64,
        matrix[1 + rows] as f64,
        matrix[1 + 2 * rows] as f64,
        matrix[2] as f64,
        matrix[2 + rows] as f64,
        matrix[2 + 2 * rows] as f64,
    );
    S_DET.set(
        r11 * r22 * r33 - r11 * r23 * r32 + r12 * r23 * r31 - r12 * r21 * r33 + r13 * r21 * r32
            - r13 * r22 * r31,
    );
    S_DET.set(S_DET.get() - 1.0);
    if S_DET.get() > 0.01 || S_DET.get() < -0.01 {
        return Err(());
    }
    let (a, b, g) = if (r13 - 1.).abs() < 1e-7 || (r13 + 1.).abs() < 1e-7 {
        (0., r13.asin(), r21.atan2(r22))
    } else if r13.abs() <= 1e-7 {
        ((-r23).atan2(r33), 0., (-r12).atan2(r11))
    } else {
        let a = (-r23).atan2(r33);
        let g = (-r12).atan2(r11);
        let cb = if g.cos().abs() > 0.01 {
            r11 / g.cos()
        } else {
            -r12 / g.sin()
        };
        (a, r13.atan2(cb), g)
    };
    Ok((a / 0.017453292, b / 0.017453292, g / 0.017453292))
}
/// Matches `icalc_angles`.  `linearxforms.c:302` returns `void`: on a matrix
/// that is not a pure rotation it leaves `angles` untouched and prints the five
/// lines the old Fortran routine printed, to stdout.  That printing is part of
/// the contract — a caller has no other way to learn the call failed.
pub fn icalc_angles(angles: &mut [f32; 3], matrix: &[f32]) {
    if let Ok((x, y, z)) = matrix_to_angles(matrix, 3) {
        angles[0] = x as f32;
        angles[1] = y as f32;
        angles[2] = z as f32;
        return;
    }

    // And here is what the old fortran routine did:
    let mut out = std::io::stdout();
    let _ = out.write_all(
        c_format(
            "icalc_angles - matrix %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[0] as f64),
                CArg::Dbl(matrix[3] as f64),
                CArg::Dbl(matrix[6] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(
        c_format(
            "                      %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[1] as f64),
                CArg::Dbl(matrix[4] as f64),
                CArg::Dbl(matrix[7] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(
        c_format(
            "                      %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[2] as f64),
                CArg::Dbl(matrix[5] as f64),
                CArg::Dbl(matrix[8] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(c_format("determinant - %f\n", &[CArg::Dbl(S_DET.get())]).as_bytes());
    let _ = out.write_all(b"ERROR: icalc_angles - Not a pure rotation matrix\n");
    let _ = out.flush();
}

/// Matches `invertMatrix`.
pub fn invert_matrix(matrix: &[f32; 9], inverse: &mut [f32; 9]) {
    let d = matrix[0] * matrix[4] * matrix[8]
        + matrix[1] * matrix[5] * matrix[6]
        + matrix[3] * matrix[7] * matrix[2]
        - matrix[6] * matrix[4] * matrix[2]
        - matrix[1] * matrix[3] * matrix[8]
        - matrix[0] * matrix[5] * matrix[7];
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / d;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / d;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / d;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / d;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / d;
    inverse[5] = (matrix[2] * matrix[3] - matrix[5] * matrix[0]) / d;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / d;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / d;
    inverse[8] = (matrix[4] * matrix[0] - matrix[1] * matrix[3]) / d;
}
/// Matches `inv_matrix`.
pub fn inv_matrix(matrix: &[f32; 9], inverse: &mut [f32; 9]) {
    invert_matrix(matrix, inverse)
}

/// C's `errno` as `readOneXform` leaves it for `exitFromXFReadError`
/// (`linearxforms.c:357`, `:395`).  The source zeroes the global `errno`
/// before its `fscanf` and tests it afterwards, and `exitFromXFReadError`
/// reads it again to choose and fill its message; this module-level value is
/// that global for the two functions that share it.
static S_XF_ERRNO: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Original `readOneXform` (`linearxforms.c:354`).
///
/// The C reads with `fscanf(fp, "%f %f %f %f %f %f \n", &xf[0], &xf[2],
/// &xf[1], &xf[3], &xf[4], &xf[5])`, and the scan is translated here rather
/// than replaced by a parser (`NATIVE.md` §2): glibc's `vfscanf` float
/// conversion, as it behaves in the C locale with no field width.  `fp` is a
/// `BufRead` because that is what a `FILE *` read stream is — its buffer is
/// the stdio buffer, and `fill_buf` gives the one character of lookahead that
/// `vfscanf` gets from `ungetc`.  What `vfscanf` does, and so what this does:
///
/// - white space (`isspace`) before each conversion is skipped; end of file
///   there is an *input* failure, which returns `EOF` if nothing was
///   converted yet;
/// - the characters of a number are collected greedily — a sign, then `nan`,
///   `inf`/`infinity`, a `0x` hexadecimal significand or decimal digits with
///   one `.`, then `e`/`p` with an optional sign — and are consumed even when
///   they turn out not to form a number (`100e` consumes the `e` and converts
///   100; `infx` consumes `x` and fails);
/// - the collected text goes to `strtof`, and the conversion fails (a
///   *matching* failure, returning the count so far) if `strtof` converts
///   nothing, or if it is only a sign or only a sign and `0x`;
/// - `strtof` rounds straight to `float`, as Rust's `f32` parser does, and
///   sets `ERANGE` on overflow and on an inexact result in the subnormal
///   range (glibc detects tininess after rounding on x86).  A hexadecimal
///   significand goes through the double `strtod` and is then narrowed, which
///   can double-round beyond 24 significant bits; no transform file carries
///   one;
/// - after the six conversions the format's trailing `" \n"` skips any white
///   space, and end of file there is not an error.
///
/// A read error is an end of file to the scan, with `errno` set from it.
pub fn read_one_xform<R: std::io::BufRead + ?Sized>(fp: &mut R, xf: &mut [f32]) -> i32 {
    use std::sync::atomic::Ordering;
    let num_ret: i32;
    let mut errno: i32 = 0;
    {
        // `vfscanf`: `inchar` is `peek` followed by `consume(1)`; `ungetc`
        // is not consuming the peeked byte.
        let mut peek = |fp: &mut R, errno: &mut i32| -> Option<u8> {
            match fp.fill_buf() {
                Ok(buf) => buf.first().copied(),
                Err(e) => {
                    *errno = e.raw_os_error().unwrap_or(5);
                    None
                }
            }
        };
        let is_space = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
        // `&xf[0], &xf[2], &xf[1], &xf[3], &xf[4], &xf[5]`
        let dest: [usize; 6] = [0, 2, 1, 3, 4, 5];
        let mut done: i32 = 0;
        let mut completed = true;
        'format: for &slot in dest.iter() {
            // Eat whitespace; end of file is an input failure.
            let mut c = loop {
                match peek(fp, &mut errno) {
                    Some(c) if is_space(c) => fp.consume(1),
                    Some(c) => break c,
                    None => {
                        if done == 0 {
                            done = -1;
                        }
                        completed = false;
                        break 'format;
                    }
                }
            };
            let mut charbuf: Vec<u8> = Vec::new();
            let mut got_sign = 0usize;
            let mut hexa = false;
            let mut conv_error = false;
            let mut special = false;
            // Check for a sign.
            if c == b'-' || c == b'+' {
                got_sign = 1;
                charbuf.push(c);
                fp.consume(1);
                match peek(fp, &mut errno) {
                    Some(next) => c = next,
                    None => conv_error = true,
                }
            }
            // Take care for the special arguments "nan" and "inf": each
            // mismatching character is read, so consumed, before failing.
            if !conv_error && (c | 32) == b'n' {
                charbuf.push(c);
                fp.consume(1);
                for want in [b'a', b'n'] {
                    match peek(fp, &mut errno) {
                        Some(next) if (next | 32) == want => {
                            charbuf.push(next);
                            fp.consume(1);
                        }
                        Some(_) => {
                            fp.consume(1);
                            conv_error = true;
                            break;
                        }
                        None => {
                            conv_error = true;
                            break;
                        }
                    }
                }
                special = true;
            } else if !conv_error && (c | 32) == b'i' {
                charbuf.push(c);
                fp.consume(1);
                for want in [b'n', b'f'] {
                    match peek(fp, &mut errno) {
                        Some(next) if (next | 32) == want => {
                            charbuf.push(next);
                            fp.consume(1);
                        }
                        Some(_) => {
                            fp.consume(1);
                            conv_error = true;
                            break;
                        }
                        None => {
                            conv_error = true;
                            break;
                        }
                    }
                }
                // It is as least "inf".
                if !conv_error {
                    if let Some(next) = peek(fp, &mut errno) {
                        if (next | 32) == b'i' {
                            charbuf.push(next);
                            fp.consume(1);
                            for want in [b'n', b'i', b't', b'y'] {
                                match peek(fp, &mut errno) {
                                    Some(next) if (next | 32) == want => {
                                        charbuf.push(next);
                                        fp.consume(1);
                                    }
                                    Some(_) => {
                                        fp.consume(1);
                                        conv_error = true;
                                        break;
                                    }
                                    None => {
                                        conv_error = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                special = true;
            }
            if !conv_error && !special {
                let exp_char;
                let mut got_digit = false;
                let mut got_dot = false;
                let mut got_e = false;
                let mut cur = Some(c);
                if c == b'0' {
                    charbuf.push(c);
                    fp.consume(1);
                    cur = peek(fp, &mut errno);
                    if cur.map(|x| x | 32) == Some(b'x') {
                        // It is a number in hexadecimal format.
                        charbuf.push(cur.unwrap());
                        fp.consume(1);
                        hexa = true;
                        exp_char = b'p';
                        cur = peek(fp, &mut errno);
                    } else {
                        exp_char = b'e';
                        got_digit = true;
                    }
                } else {
                    exp_char = b'e';
                }
                while let Some(ch) = cur {
                    if ch.is_ascii_digit() {
                        charbuf.push(ch);
                        got_digit = true;
                    } else if !got_e && hexa && ch.is_ascii_hexdigit() {
                        charbuf.push(ch);
                        got_digit = true;
                    } else if got_e
                        && charbuf.last() == Some(&exp_char)
                        && (ch == b'-' || ch == b'+')
                    {
                        charbuf.push(ch);
                    } else if got_digit && !got_e && (ch | 32) == exp_char {
                        charbuf.push(exp_char);
                        got_e = true;
                        got_dot = true;
                    } else if !got_dot && ch == b'.' {
                        charbuf.push(ch);
                        got_dot = true;
                    } else {
                        // The last read character is not part of the number
                        // anymore.
                        break;
                    }
                    fp.consume(1);
                    cur = peek(fp, &mut errno);
                }
                // Have we read any character?  If we try to read a number in
                // hexadecimal notation and we have read only the `0x' prefix
                // this is an error.
                if charbuf.len() == got_sign || (hexa && charbuf.len() == 2 + got_sign) {
                    conv_error = true;
                }
            }
            if !conv_error {
                // scan_float: `__strtof_internal` on the collected text.
                let s = &charbuf[..];
                let mut i = 0usize;
                let negative = s.first() == Some(&b'-');
                if s.first() == Some(&b'-') || s.first() == Some(&b'+') {
                    i = 1;
                }
                let rest = &s[i..];
                let lower: Vec<u8> = rest.iter().map(|b| b | 32).collect();
                let mut value: Option<f32> = None;
                if lower.starts_with(b"inf") || lower.starts_with(b"nan") {
                    let v = if lower.starts_with(b"inf") {
                        f32::INFINITY
                    } else {
                        f32::NAN
                    };
                    value = Some(if negative { -v } else { v });
                } else if hexa {
                    let mut end = 0usize;
                    let v = crate::imod::libcfshr::parse_params::strtod(s, &mut end);
                    if end > 0 {
                        let v32 = v as f32;
                        if v32.is_infinite() || (v != 0. && v32.abs() < f32::MIN_POSITIVE) {
                            errno = 34; // ERANGE
                        }
                        value = Some(v32);
                    }
                } else {
                    // The longest prefix `strtof` accepts: digits, one `.`,
                    // at least one digit in all, and an exponent only when
                    // digits follow it.
                    let mut j = 0usize;
                    let mut mantissa_digits = 0usize;
                    let mut nonzero = false;
                    while j < rest.len() && rest[j].is_ascii_digit() {
                        nonzero |= rest[j] != b'0';
                        j += 1;
                        mantissa_digits += 1;
                    }
                    if j < rest.len() && rest[j] == b'.' {
                        j += 1;
                        while j < rest.len() && rest[j].is_ascii_digit() {
                            nonzero |= rest[j] != b'0';
                            j += 1;
                            mantissa_digits += 1;
                        }
                    }
                    if mantissa_digits > 0 {
                        let mantissa_end = j;
                        if j < rest.len() && (rest[j] | 32) == b'e' {
                            let mut k = j + 1;
                            if k < rest.len() && (rest[k] == b'+' || rest[k] == b'-') {
                                k += 1;
                            }
                            let digits_start = k;
                            while k < rest.len() && rest[k].is_ascii_digit() {
                                k += 1;
                            }
                            j = if k > digits_start { k } else { mantissa_end };
                        }
                        let text = std::str::from_utf8(&s[..i + j]).unwrap_or("0");
                        let v32: f32 = text.parse().unwrap_or(0.);
                        if v32.is_infinite() {
                            errno = 34; // ERANGE: overflow
                        } else if nonzero && v32.abs() < f32::MIN_POSITIVE {
                            // Underflow: ERANGE when the result is inexact.
                            // A subnormal float is exact in double, so a
                            // double parse that lands on it is taken as the
                            // exact value.
                            let v64: f64 = text.parse().unwrap_or(0.);
                            if v32 == 0. || v64 != v32 as f64 {
                                errno = 34;
                            }
                        }
                        value = Some(v32);
                    }
                }
                match value {
                    Some(v) => {
                        xf[slot] = v;
                        done += 1;
                    }
                    None => conv_error = true,
                }
            }
            if conv_error {
                completed = false;
                break 'format;
            }
        }
        if completed {
            // The format's trailing " \n": consume the last white spaces.
            while let Some(c) = peek(fp, &mut errno) {
                if !is_space(c) {
                    break;
                }
                fp.consume(1);
            }
        }
        num_ret = done;
    }
    S_XF_ERRNO.store(errno, Ordering::Relaxed);
    if errno != 0 {
        return 2;
    }
    if num_ret > 0 && num_ret != -1 && num_ret < 6 {
        return 3;
    }
    if num_ret == -1 {
        return 1;
    }
    0
}

/// Original `readAllXforms` (`linearxforms.c:374`).  `num_read` is written
/// only for a transform read or an end of file, as in the C.
pub fn read_all_xforms<R: std::io::BufRead + ?Sized>(
    fp: &mut R,
    xforms: &mut [f32],
    max_read: i32,
    num_read: &mut i32,
) -> i32 {
    for ind in 0..max_read {
        let ret_val = read_one_xform(fp, &mut xforms[6 * ind as usize..]);
        if ret_val < 2 {
            *num_read = ind + 1;
        }
        if ret_val != 0 {
            return if ret_val == 1 { 0 } else { ret_val };
        }
    }
    0
}

/// Original `exitFromXFReadError` (`linearxforms.c:391`).  `errno` is the
/// value [`read_one_xform`] left, and `strerror` is the operating system's
/// message, which `std::io::Error` renders with a ` (os error N)` suffix that
/// the C library does not add.
pub fn exit_from_xf_read_error(ierr: i32, descrip: &str) {
    if ierr == 0 {
        return;
    }
    let errno = S_XF_ERRNO.load(std::sync::atomic::Ordering::Relaxed);
    if ierr == 2 && errno != 0 {
        let text = std::io::Error::from_raw_os_error(errno).to_string();
        let text = text
            .split(" (os error ")
            .next()
            .unwrap_or(&text)
            .to_string();
        crate::imod::libcfshr::parse_params::exit_error(
            c_format(
                "Reading %s transform file: %s",
                &[CArg::Str(descrip), CArg::Str(&text)],
            )
            .as_bytes(),
        );
    }
    crate::imod::libcfshr::parse_params::exit_error(
        c_format(
            "Reading %s transform file%s",
            &[
                CArg::Str(descrip),
                CArg::Str(if ierr == 3 {
                    ": fewer than 6 values on a line"
                } else {
                    ""
                }),
            ],
        )
        .as_bytes(),
    );
}

/// Original `writeXform` (`linearxforms.c:406`).  Returns the operating
/// system's error number when the write fails, and 0 otherwise.
pub fn write_xform(fp: &mut dyn Write, xf: &[f32]) -> i32 {
    let ret = fp.write_all(
        c_format(
            " %11.7f %11.7f %11.7f %11.7f %11.3f %11.3f\n",
            &[
                CArg::Dbl(xf[0] as f64),
                CArg::Dbl(xf[2] as f64),
                CArg::Dbl(xf[1] as f64),
                CArg::Dbl(xf[3] as f64),
                CArg::Dbl(xf[4] as f64),
                CArg::Dbl(xf[5] as f64),
            ],
        )
        .as_bytes(),
    );
    match ret {
        Ok(()) => 0,
        // An error with no system error number behind it is reported as
        // `EIO`, where the C library would have left one in `errno`.
        Err(e) => e.raw_os_error().unwrap_or(5),
    }
}
