//! Rust-only stand-in for the parts of the C++ standard library's iostreams
//! that the translated C++ units (`IMOD/raptor`) read and write text with.
//! A runtime boundary like [`crate::imod::c_sort`], not a translated unit.
//!
//! Output: `ostream << double` (and `<< float`, which the stream promotes to
//! `double`) is libstdc++'s `__convert_from_v(.., "%.*g", 6, v)` -- C's `%g`
//! with the default precision of six significant digits -- so
//! [`ostream_double`] is [`c_format`] of `%g`.  Rust's `{}` prints the
//! shortest round-tripping decimal and is not the same thing.
//!
//! Input: `istream >> x` for a number is libstdc++ 11's `num_get`, which does
//! *not* hand the text to `strtod` directly.  `_M_extract_float` first
//! accumulates the characters that fit a decimal-number pattern (optional
//! sign, digits, one `.`, one `e`/`E` after a digit with an optional sign) and
//! stops, unconsumed, at the first character that does not; then
//! `__convert_to_v` runs `strtod` over the accumulated text and fails -- value
//! 0, `failbit` -- unless it consumed all of it.  So `"0.5,"` reads `0.5` and
//! leaves the comma, `"-inf"` and `"nan"` read as a *failed* 0, `"1e"` fails
//! outright, and an overflow reads as `DBL_MAX` with `failbit`.  The sentry in
//! front of it skips white space and, when it meets the end of the stream
//! instead, sets `eofbit | failbit` and leaves the value untouched -- which is
//! why a `while (in.good()) { in >> v; list.push_back(v); }` loop pushes the
//! last value twice.  [`IStream`] reproduces those rules; `str::parse` does
//! not.

use crate::imod::libcfshr::b3dutil::{CArg, c_format};

/// `ostream << double` / `ostream << float` with the stream's default
/// formatting (`precision` 6, no `floatfield`): C's `%g`.
pub fn ostream_double(x: f64) -> String {
    c_format("%g", &[CArg::Dbl(x)])
}

/// An input stream over a byte buffer with the three state bits the
/// translated code consults (`eofbit`, `failbit`; `badbit` never arises from a
/// memory buffer).  `ifstream` reads the whole file up front; `istringstream`
/// wraps a string.  Extraction after a failure does nothing, as a C++ sentry
/// on a stream whose state is not good does nothing.
pub struct IStream {
    buf: Vec<u8>,
    pos: usize,
    eof: bool,
    fail: bool,
}

impl IStream {
    /// `ifstream in(path)`: `None` where the C++ stream's `is_open()` would be
    /// false.
    pub fn open(path: &str) -> Option<IStream> {
        std::fs::read(path).ok().map(IStream::from_bytes)
    }

    /// `istringstream ss(text)`.
    pub fn from_bytes(buf: Vec<u8>) -> IStream {
        IStream {
            buf,
            pos: 0,
            eof: false,
            fail: false,
        }
    }

    /// `good()`.
    pub fn good(&self) -> bool {
        !self.eof && !self.fail
    }

    /// `eof()`.
    pub fn eof(&self) -> bool {
        self.eof
    }

    /// `fail()`.
    pub fn fail(&self) -> bool {
        self.fail
    }

    /// `operator bool` / `!in`: true while `fail()` is false.
    pub fn ok(&self) -> bool {
        !self.fail
    }

    /// The `istream::sentry` of a formatted extractor: false (and the stream
    /// marked) when the stream is not good or only white space remains.
    fn sentry(&mut self) -> bool {
        if !self.good() {
            self.fail = true;
            return false;
        }
        while self.pos < self.buf.len() && is_space(self.buf[self.pos]) {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            self.eof = true;
            self.fail = true;
            return false;
        }
        true
    }

    /// libstdc++ `num_get::_M_extract_float`: the accumulated text, with
    /// `eofbit` set when the end of the buffer ended the scan.
    fn extract_float_text(&mut self) -> String {
        let mut xtrc = String::new();
        let end = self.buf.len();
        let mut testeof = self.pos >= end;
        let mut c = if testeof { 0 } else { self.buf[self.pos] };
        // First check for sign.
        if !testeof && (c == b'+' || c == b'-') {
            xtrc.push(c as char);
            self.pos += 1;
            if self.pos < end {
                c = self.buf[self.pos];
            } else {
                testeof = true;
            }
        }
        // Next, look for leading zeros.
        let mut found_mantissa = false;
        while !testeof {
            if c == b'.' {
                break;
            } else if c == b'0' {
                if !found_mantissa {
                    xtrc.push('0');
                    found_mantissa = true;
                }
                self.pos += 1;
                if self.pos < end {
                    c = self.buf[self.pos];
                } else {
                    testeof = true;
                }
            } else {
                break;
            }
        }
        // Only need acceptable digits for floating point numbers.
        let mut found_dec = false;
        let mut found_sci = false;
        while !testeof {
            if c.is_ascii_digit() {
                xtrc.push(c as char);
                found_mantissa = true;
            } else if c == b'.' && !found_dec && !found_sci {
                xtrc.push('.');
                found_dec = true;
            } else if (c == b'e' || c == b'E') && !found_sci && found_mantissa {
                // Scientific notation.
                xtrc.push('e');
                found_sci = true;
                // Remove optional plus or minus sign, if they exist.
                self.pos += 1;
                if self.pos < end {
                    c = self.buf[self.pos];
                    if c == b'+' || c == b'-' {
                        xtrc.push(c as char);
                    } else {
                        continue;
                    }
                } else {
                    testeof = true;
                    break;
                }
            } else {
                break;
            }
            self.pos += 1;
            if self.pos < end {
                c = self.buf[self.pos];
            } else {
                testeof = true;
            }
        }
        if testeof {
            self.eof = true;
        }
        xtrc
    }

    /// `in >> v` for a `double` (`__convert_to_v` with `strtod`).
    pub fn read_f64(&mut self, v: &mut f64) {
        if !self.sentry() {
            return;
        }
        let text = self.extract_float_text();
        match strtod_whole(&text) {
            Some(x) if x == f64::INFINITY => {
                *v = f64::MAX;
                self.fail = true;
            }
            Some(x) if x == f64::NEG_INFINITY => {
                *v = -f64::MAX;
                self.fail = true;
            }
            Some(x) => *v = x,
            None => {
                *v = 0.0;
                self.fail = true;
            }
        }
    }

    /// `in >> v` for a `float` (`__convert_to_v` with `strtof`).
    pub fn read_f32(&mut self, v: &mut f32) {
        if !self.sentry() {
            return;
        }
        let text = self.extract_float_text();
        match strtof_whole(&text) {
            Some(x) if x == f32::INFINITY => {
                *v = f32::MAX;
                self.fail = true;
            }
            Some(x) if x == f32::NEG_INFINITY => {
                *v = -f32::MAX;
                self.fail = true;
            }
            Some(x) => *v = x,
            None => {
                *v = 0.0;
                self.fail = true;
            }
        }
    }

    /// `in >> v` for an `int` (`num_get::_M_extract_int`, base 10): optional
    /// sign and decimal digits; no digits is a failed 0, and a value outside
    /// `int` is `INT_MAX`/`INT_MIN` with `failbit`.
    pub fn read_i32(&mut self, v: &mut i32) {
        if !self.sentry() {
            return;
        }
        let end = self.buf.len();
        let mut negative = false;
        let c = self.buf[self.pos];
        if c == b'+' || c == b'-' {
            negative = c == b'-';
            self.pos += 1;
        }
        let mut found = false;
        let mut overflow = false;
        let mut result: u64 = 0;
        while self.pos < end && self.buf[self.pos].is_ascii_digit() {
            found = true;
            if !overflow {
                result = result * 10 + (self.buf[self.pos] - b'0') as u64;
                if result > i32::MAX as u64 + 1 {
                    overflow = true;
                }
            }
            self.pos += 1;
        }
        if self.pos >= end {
            self.eof = true;
        }
        if !found {
            *v = 0;
            self.fail = true;
        } else if overflow || (!negative && result > i32::MAX as u64) {
            *v = if negative { i32::MIN } else { i32::MAX };
            self.fail = true;
        } else if negative {
            *v = (result as i64).wrapping_neg() as i32;
        } else {
            *v = result as i32;
        }
    }

    /// `in >> s` for a `std::string`: skip white space, then take characters
    /// up to the next white space.
    pub fn read_string(&mut self, s: &mut String) {
        if !self.sentry() {
            return;
        }
        let start = self.pos;
        while self.pos < self.buf.len() && !is_space(self.buf[self.pos]) {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            self.eof = true;
        }
        *s = String::from_utf8_lossy(&self.buf[start..self.pos]).into_owned();
    }

    /// `getline(in, s)`: the characters up to the next `'\n'`, which is
    /// consumed and not stored.  Reaching the end sets `eofbit`, and also
    /// `failbit` when nothing at all was extracted.
    pub fn getline(&mut self, s: &mut String) -> bool {
        if !self.good() {
            self.fail = true;
            return false;
        }
        let start = self.pos;
        let end = self.buf.len();
        while self.pos < end && self.buf[self.pos] != b'\n' {
            self.pos += 1;
        }
        *s = String::from_utf8_lossy(&self.buf[start..self.pos]).into_owned();
        if self.pos < end {
            self.pos += 1;
        } else {
            self.eof = true;
            if start == end {
                self.fail = true;
            }
        }
        !self.fail
    }
}

/// The C locale's `isspace`.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

/// `strtod` over the accumulated text, `None` unless it consumes all of it.
/// The text is already restricted to `[+-]digits[.digits][e[+-]digits]`, for
/// which Rust's correctly rounded parse and glibc's `strtod` agree; what is
/// left is the partial-parse rule.
fn strtod_whole(text: &str) -> Option<f64> {
    if !valid_decimal(text) {
        return None;
    }
    text.parse::<f64>().ok()
}

/// As [`strtod_whole`], for `strtof`.
fn strtof_whole(text: &str) -> Option<f32> {
    if !valid_decimal(text) {
        return None;
    }
    text.parse::<f32>().ok()
}

/// Whether `strtod` would consume the whole accumulated text: a mantissa
/// with at least one digit, and an exponent (if present) with at least one
/// digit.
fn valid_decimal(text: &str) -> bool {
    let b = text.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let mut digits = 0;
    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
        if b[i] != b'.' {
            digits += 1;
        }
        i += 1;
    }
    if digits == 0 {
        return false;
    }
    if i == b.len() {
        return true;
    }
    if b[i] != b'e' {
        return false;
    }
    i += 1;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let exp_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    i == b.len() && i > exp_start
}

thread_local! {
    /// Where `std::cout`, `std::cerr` and `std::clog` write: `None` for the
    /// process's standard output / error, `Some` after the program
    /// redirected them with `rdbuf` (RAPTOR points all three at its log
    /// file).  Thread-local so a program run in process gets a fresh one.
    static REDIRECT: std::cell::RefCell<Option<std::fs::File>> =
        const { std::cell::RefCell::new(None) };
}

/// `cout.rdbuf(file)`, `cerr.rdbuf(file)` and `clog.rdbuf(file)` together:
/// sends all three streams to `file` (`None` restores the standard streams)
/// and returns the previous target.
pub fn redirect_standard_streams(file: Option<std::fs::File>) -> Option<std::fs::File> {
    REDIRECT.with(|r| std::mem::replace(&mut *r.borrow_mut(), file))
}

/// `std::cout << text`.  Every `endl` in the translated code is part of
/// `text`; the C++ stream is flushed at each `endl`, and so the text is
/// written through at once.
pub fn cout(text: &str) {
    use std::io::Write as _;
    REDIRECT.with(|r| match &mut *r.borrow_mut() {
        Some(file) => {
            let _ = file.write_all(text.as_bytes());
        }
        None => {
            let mut out = std::io::stdout();
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }
    })
}

/// `std::cerr << text` (unbuffered in C++).
pub fn cerr(text: &str) {
    use std::io::Write as _;
    REDIRECT.with(|r| match &mut *r.borrow_mut() {
        Some(file) => {
            let _ = file.write_all(text.as_bytes());
        }
        None => {
            let _ = std::io::stderr().write_all(text.as_bytes());
        }
    })
}
