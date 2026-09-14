//! Translation of `IMOD/libwarp/nncommon.c`.
//!
//! The source's three globals (`nn_verbose`, `nn_test_vertice`, `nn_rule`) are
//! declared by `nn.h` and live in [`crate::imod::libwarp::nn`], where they are
//! thread-local cells; this module reads them with `.get()`.
#![allow(dead_code)]

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::delaunay::Circle;
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};

/// C `BUFSIZE` (`nncommon.c:44`).
const BUFSIZE: usize = 1024;
/// C `EPS` (`nncommon.c:45`).
const EPS: f64 = 1.0e-15;
/// C `NALLOCATED_START` (`nncommon.c:46`).
const NALLOCATED_START: i32 = 1024;

/// Original `nn_quit` (`nncommon.c:54`).
///
/// The source is `nn_quit(char* format, ...)` and `vfprintf`s its arguments.
/// A varargs signature cannot survive the conversion, so the five call sites
/// format with [`c_format`] and hand the finished message over; the fixed
/// prefix, the `fflush(stdout)` before it and the `exit(1)` after it stay
/// here, where the source puts them.
pub fn nn_quit(message: &str) -> ! {
    /* just in case, to have the exit message last */
    let _ = ImodFile::Stdout.flush();

    let mut err = ImodFile::Stderr;
    let _ = err.write_all(b"  error: libnn: ");
    let _ = err.write_all(message.as_bytes());

    std::process::exit(1);
}

/// Original `circle_contains` (`nncommon.c:69`).
pub fn circle_contains(c: &Circle, p: &Point) -> i32 {
    ((c.x - p.x).hypot(c.y - p.y) <= c.r) as i32
}

/// Original `points_thingrid` (`nncommon.c:85`).
///
/// The source takes `int* pn, point** ppoints`, frees the incoming array and
/// hands back a fresh one. A `&mut Vec<Point>` is both roles at once; `*pn`
/// stays as a separate out-parameter because the source's callers read it.
pub fn points_thingrid(pn: &mut i32, ppoints: &mut Vec<Point>, nx: i32, ny: i32) {
    let n = *pn;
    let points: Vec<Point> = std::mem::take(ppoints);
    let mut xmin = f64::MAX;
    let mut xmax = -f64::MAX;
    let mut ymin = f64::MAX;
    let mut ymax = -f64::MAX;
    let nxy = nx * ny;
    let mut sumx: Vec<f64>;
    let mut sumy: Vec<f64>;
    let mut sumz: Vec<f64>;
    let mut count: Vec<i32>;
    let stepx: f64;
    let stepy: f64;
    let mut nnew = 0;
    let mut pointsnew: Vec<Point>;
    let (mut i, mut j, mut index): (i32, i32, i32);

    /* DNM: removed two redeclarations of index to eliminate compiler warnings */

    if NN_VERBOSE.get() != 0 {
        let _ = ImodFile::Stderr
            .write_all(c_format("thinned: %d points -> ", &[CArg::Int(*pn as i64)]).as_bytes());
    }

    if nx < 1 || ny < 1 {
        // `free(points)` — the local copy goes out of scope here.
        *ppoints = Vec::new();
        *pn = 0;
        if NN_VERBOSE.get() != 0 {
            let _ = ImodFile::Stderr.write_all(b"0 points");
        }
        return;
    }

    // `calloc(nxy, …)` for the four accumulators, after the early return so
    // that a bad `nx`/`ny` never sizes them.
    sumx = vec![0.0; nxy as usize];
    sumy = vec![0.0; nxy as usize];
    sumz = vec![0.0; nxy as usize];
    count = vec![0i32; nxy as usize];

    for ii in 0..n {
        let p = &points[ii as usize];

        if p.x < xmin {
            xmin = p.x;
        }
        if p.x > xmax {
            xmax = p.x;
        }
        if p.y < ymin {
            ymin = p.y;
        }
        if p.y > ymax {
            ymax = p.y;
        }
    }

    stepx = if nx > 1 {
        (xmax - xmin) / nx as f64
    } else {
        0.0
    };
    stepy = if ny > 1 {
        (ymax - ymin) / ny as f64
    } else {
        0.0
    };

    for ii in 0..n {
        let p = &points[ii as usize];

        if nx == 1 {
            i = 0;
        } else {
            let fi = (p.x - xmin) / stepx;

            // `rint` is round-to-nearest, ties to even, in the default mode.
            if (fi.round_ties_even() - fi).abs() < EPS {
                i = fi.round_ties_even() as i32;
            } else {
                i = fi.floor() as i32;
            }
        }
        if ny == 1 {
            j = 0;
        } else {
            let fj = (p.y - ymin) / stepy;

            if (fj.round_ties_even() - fj).abs() < EPS {
                j = fj.round_ties_even() as i32;
            } else {
                j = fj.floor() as i32;
            }
        }

        if i == nx {
            i -= 1;
        }
        if j == ny {
            j -= 1;
        }
        index = i + j * nx;
        sumx[index as usize] += p.x;
        sumy[index as usize] += p.y;
        sumz[index as usize] += p.z;
        count[index as usize] += 1;
    }

    for j in 0..ny {
        for i in 0..nx {
            index = i + j * nx;

            if count[index as usize] > 0 {
                nnew += 1;
            }
        }
    }

    pointsnew = Vec::with_capacity(nnew as usize);

    index = 0;
    for _j in 0..ny {
        for _i in 0..nx {
            let nn = count[index as usize];

            if nn > 0 {
                // `p = &pointsnew[ii]`, filled in place; `ii++` is the push.
                pointsnew.push(Point {
                    x: sumx[index as usize] / nn as f64,
                    y: sumy[index as usize] / nn as f64,
                    z: sumz[index as usize] / nn as f64,
                });
            }
            index += 1;
        }
    }

    if NN_VERBOSE.get() != 0 {
        let _ = ImodFile::Stderr
            .write_all(c_format("%d points\n", &[CArg::Int(nnew as i64)]).as_bytes());
    }

    drop(sumx);
    drop(sumy);
    drop(sumz);
    drop(count);

    drop(points);
    *ppoints = pointsnew;
    *pn = nnew;
}

/// Original `points_thinlin` (`nncommon.c:217`).
pub fn points_thinlin(nin: &mut i32, pin: &mut Vec<Point>, rmax: f64) {
    let mut nout = 0;
    let mut nallocated = NALLOCATED_START;
    let mut pout: Vec<Point> = vec![Point::default(); nallocated as usize];
    let mut n: f64 = 0.;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut sum_r = 0.0;
    // `point *pprev` is only ever read for its x and y, so the index into the
    // input array carries everything the pointer did.
    let mut pprev: Option<usize> = None;

    for i in 0..*nin {
        let p = pin[i as usize];
        let dist;

        if p.x.is_nan() || p.y.is_nan() || p.z.is_nan() {
            if pprev.is_some() {
                /*
                 * write point
                 */
                if nout == nallocated {
                    nallocated *= 2;
                    pout.resize(nallocated as usize, Point::default());
                }
                pout[nout as usize].x = sum_x / n;
                pout[nout as usize].y = sum_y / n;
                pout[nout as usize].z = sum_z / n;
                nout += 1;
                /*
                 * reset cluster
                 */
                pprev = None;
            }
            continue;
        }

        /*
         * init cluster
         */
        if pprev.is_none() {
            sum_x = p.x;
            sum_y = p.y;
            sum_z = p.z;
            sum_r = 0.0;
            n = 1.;
            pprev = Some(i as usize);
            continue;
        }

        let prev = pin[pprev.unwrap()];
        dist = (p.x - prev.x).hypot(p.y - prev.y);
        if sum_r + dist > rmax {
            /*
             * write point
             */
            if nout == nallocated {
                nallocated *= 2;
                pout.resize(nallocated as usize, Point::default());
            }
            pout[nout as usize].x = sum_x / n;
            pout[nout as usize].y = sum_y / n;
            pout[nout as usize].z = sum_z / n;
            nout += 1;
            /*
             * reset cluster
             */
            pprev = None;
        } else {
            /*
             * add to cluster
             */
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
            sum_r += dist;
            n += 1.;
            pprev = Some(i as usize);
        }
    }

    // `free(*pin); *pin = realloc(pout, nout * sizeof(point));`
    pout.truncate(nout as usize);
    *pin = pout;
    *nin = nout;
}

/// Original `points_getrange` (`nncommon.c:313`).
///
/// The source takes four `double*` that may be NULL, and *reassigns the local
/// pointer to NULL* when the caller supplied a non-NaN value, to mean "leave
/// this one alone" for the rest of the routine. The parameters are plain
/// `&mut f64` here, so that second role is carried by rebinding each one to an
/// `Option` — the same two states the C pointer has, minus the caller-side
/// NULL, which no caller in this tree passes.
pub fn points_getrange(
    n: i32,
    points: &[Point],
    zoom: f64,
    xmin: &mut f64,
    xmax: &mut f64,
    ymin: &mut f64,
    ymax: &mut f64,
) {
    let mut xmin: Option<&mut f64> = Some(xmin);
    let mut xmax: Option<&mut f64> = Some(xmax);
    let mut ymin: Option<&mut f64> = Some(ymin);
    let mut ymax: Option<&mut f64> = Some(ymax);

    if let Some(v) = xmin.as_deref_mut() {
        if v.is_nan() {
            *v = f64::MAX;
        } else {
            xmin = None;
        }
    }
    if let Some(v) = xmax.as_deref_mut() {
        if v.is_nan() {
            *v = -f64::MAX;
        } else {
            xmax = None;
        }
    }
    if let Some(v) = ymin.as_deref_mut() {
        if v.is_nan() {
            *v = f64::MAX;
        } else {
            ymin = None;
        }
    }
    if let Some(v) = ymax.as_deref_mut() {
        if v.is_nan() {
            *v = -f64::MAX;
        } else {
            ymax = None;
        }
    }

    for i in 0..n {
        let p = &points[i as usize];

        if let Some(v) = xmin.as_deref_mut()
            && p.x < *v
        {
            *v = p.x;
        }
        if let Some(v) = xmax.as_deref_mut()
            && p.x > *v
        {
            *v = p.x;
        }
        if let Some(v) = ymin.as_deref_mut()
            && p.y < *v
        {
            *v = p.y;
        }
        if let Some(v) = ymax.as_deref_mut()
            && p.y > *v
        {
            *v = p.y;
        }
    }

    if zoom.is_nan() || zoom <= 0.0 || zoom == 1.0 {
        return;
    }

    if xmin.is_some() && xmax.is_some() {
        let lo = *xmin.as_deref().unwrap();
        let hi = *xmax.as_deref().unwrap();
        let xdiff2 = (hi - lo) / 2.0;
        let xav = (hi + lo) / 2.0;

        *xmin.as_deref_mut().unwrap() = xav - xdiff2 * zoom;
        *xmax.as_deref_mut().unwrap() = xav + xdiff2 * zoom;
    }
    if ymin.is_some() && ymax.is_some() {
        let lo = *ymin.as_deref().unwrap();
        let hi = *ymax.as_deref().unwrap();
        let ydiff2 = (hi - lo) / 2.0;
        let yav = (hi + lo) / 2.0;

        *ymin.as_deref_mut().unwrap() = yav - ydiff2 * zoom;
        *ymax.as_deref_mut().unwrap() = yav + ydiff2 * zoom;
    }
}

/// Original `points_generate` (`nncommon.c:387`).
pub fn points_generate(
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: i32,
    ny: i32,
    nout: &mut i32,
    pout: &mut Vec<Point>,
) {
    let stepx: f64;
    let stepy: f64;
    let x0: f64;
    let mut xx: f64;
    let mut yy: f64;
    let mut ii: usize;

    if nx < 1 || ny < 1 {
        *pout = Vec::new();
        *nout = 0;
        return;
    }

    *nout = nx * ny;
    // `malloc(*nout * sizeof(point))`: the source leaves `p->z` untouched, so
    // the generated points carry whatever `malloc` left there. A `Vec` is
    // zeroed instead — the documented consequence of NATIVE.md §4.
    *pout = vec![Point::default(); *nout as usize];

    stepx = if nx > 1 {
        (xmax - xmin) / (nx - 1) as f64
    } else {
        0.0
    };
    stepy = if ny > 1 {
        (ymax - ymin) / (ny - 1) as f64
    } else {
        0.0
    };
    x0 = if nx > 1 { xmin } else { (xmin + xmax) / 2.0 };
    yy = if ny > 1 { ymin } else { (ymin + ymax) / 2.0 };

    ii = 0;
    for _j in 0..ny {
        xx = x0;
        for _i in 0..nx {
            let p = &mut pout[ii];

            p.x = xx;
            p.y = yy;
            xx += stepx;
            ii += 1;
        }
        yy += stepy;
    }
}

/// Original `str2double` (`nncommon.c:422`).
///
/// The C is `strtod` plus one test, `end == token`, so the whole behaviour is
/// `strtod`'s: skip leading whitespace, take the **longest valid prefix**, and
/// report failure only when nothing at all was consumed. `f64::from_str` is
/// all-or-nothing and rejects the hexadecimal and `INFINITY`/`NAN(chars)`
/// forms, so the prefix scan is written out here.
///
/// One deviation, in a form no caller can produce from a columnar data file: a
/// hexadecimal significand is accumulated into a `u128` and scaled by a power
/// of two, which is exact for up to 27 significant hex digits and any
/// non-subnormal result, but can double-round beyond that where `strtod`
/// rounds once.
pub fn str2double(token: &str, value: &mut f64) -> i32 {
    // The C's `token == NULL` arm: the callers in `points_read` have already
    // tested for it, and `&str` cannot be null, so only the parse arm remains.
    let b = token.as_bytes();
    let mut i = 0usize;

    // strtod skips leading isspace()
    while i < b.len() && (b[i] == b' ' || (0x09..=0x0d).contains(&b[i])) {
        i += 1;
    }

    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }

    let mut end = 0usize; // 0 means "no conversion", i.e. end == token
    let mut val = 0.0f64;

    let rest = &b[i..];
    let lower = |c: u8| c.to_ascii_lowercase();

    if rest.len() >= 3 && lower(rest[0]) == b'i' && lower(rest[1]) == b'n' && lower(rest[2]) == b'f'
    {
        val = f64::INFINITY;
        end = i + 3;
        if rest.len() >= 8
            && rest[3..8]
                .iter()
                .map(|c| lower(*c))
                .eq(b"inity".iter().copied())
        {
            end = i + 8;
        }
    } else if rest.len() >= 3
        && lower(rest[0]) == b'n'
        && lower(rest[1]) == b'a'
        && lower(rest[2]) == b'n'
    {
        val = f64::NAN;
        end = i + 3;
        // strtod also takes an optional `(n-char-sequence)` after NAN.
        if rest.len() > 3 && rest[3] == b'(' {
            let mut k = 4usize;
            while k < rest.len() && (rest[k].is_ascii_alphanumeric() || rest[k] == b'_') {
                k += 1;
            }
            if k < rest.len() && rest[k] == b')' {
                end = i + k + 1;
            }
        }
    } else if rest.len() >= 3
        && rest[0] == b'0'
        && lower(rest[1]) == b'x'
        && (rest[2].is_ascii_hexdigit()
            || (rest[2] == b'.' && rest.len() >= 4 && rest[3].is_ascii_hexdigit()))
    {
        // Hexadecimal: 0x hexdigits [. hexdigits] [pP [sign] digits].
        // C99 7.20.1.3: if `0x` is *not* followed by a hex digit the subject
        // sequence is the initial `0` alone, so the guard above sends `"0x"`
        // and `"0xg"` down the decimal arm, which is what `strtod` does.
        let mut k = 2usize;
        let mut mant: u128 = 0;
        let mut ndig = 0i32;
        let mut extra = 0i32; // binary exponent contributed by dropped digits
        let mut seen = false;
        while k < rest.len() && rest[k].is_ascii_hexdigit() {
            seen = true;
            if ndig < 27 {
                mant = mant * 16 + (rest[k] as char).to_digit(16).unwrap() as u128;
                ndig += 1;
            } else {
                extra += 4;
            }
            k += 1;
        }
        if k < rest.len() && rest[k] == b'.' {
            k += 1;
            while k < rest.len() && rest[k].is_ascii_hexdigit() {
                seen = true;
                if ndig < 27 {
                    mant = mant * 16 + (rest[k] as char).to_digit(16).unwrap() as u128;
                    ndig += 1;
                    extra -= 4;
                }
                k += 1;
            }
        }
        if seen {
            end = i + k;
            let mut pexp = 0i32;
            if k < rest.len() && lower(rest[k]) == b'p' {
                let mut m = k + 1;
                let mut esign = 1i32;
                if m < rest.len() && (rest[m] == b'+' || rest[m] == b'-') {
                    if rest[m] == b'-' {
                        esign = -1;
                    }
                    m += 1;
                }
                if m < rest.len() && rest[m].is_ascii_digit() {
                    let mut e = 0i32;
                    while m < rest.len() && rest[m].is_ascii_digit() {
                        e = e.saturating_mul(10).saturating_add((rest[m] - b'0') as i32);
                        m += 1;
                    }
                    pexp = esign.saturating_mul(e);
                    end = i + m;
                }
            }
            val = mant as f64 * (2.0f64).powi(extra.saturating_add(pexp));
        }
    } else {
        // Decimal: digits [. digits] [eE [sign] digits], at least one digit
        let mut k = 0usize;
        let mut seen = false;
        while k < rest.len() && rest[k].is_ascii_digit() {
            seen = true;
            k += 1;
        }
        if k < rest.len() && rest[k] == b'.' {
            k += 1;
            while k < rest.len() && rest[k].is_ascii_digit() {
                seen = true;
                k += 1;
            }
        }
        if seen {
            let mut stop = k;
            if k < rest.len() && lower(rest[k]) == b'e' {
                let mut m = k + 1;
                if m < rest.len() && (rest[m] == b'+' || rest[m] == b'-') {
                    m += 1;
                }
                if m < rest.len() && rest[m].is_ascii_digit() {
                    while m < rest.len() && rest[m].is_ascii_digit() {
                        m += 1;
                    }
                    stop = m;
                }
            }
            end = i + stop;
            let text = std::str::from_utf8(&rest[..stop]).unwrap_or("0");
            val = text.parse::<f64>().unwrap_or(0.0);
        }
    }

    if end == 0 {
        *value = f64::NAN;
        return 0;
    }

    *value = if neg { -val } else { val };

    1
}

/// Original `points_read` (`nncommon.c:448`).
///
/// The source's `FILE*` becomes an [`ImodFile`]; both spellings of standard
/// input select it. `fgets(buf, BUFSIZE, f)` splits a longer line into
/// 1023-byte pieces and `strtok` sees the bytes only up to the first NUL, so
/// the stream is walked here the same way rather than by `lines()`. The whole
/// stream is read up front, which the source's loop does too — it runs to EOF
/// either way.
pub fn points_read(fname: &str, dim: i32, n: &mut i32, points: &mut Vec<Point>) {
    let mut f: ImodFile;
    let mut nallocated = NALLOCATED_START;
    let seps: &[u8] = b" ,;\t";
    let from_stdin: bool;

    if dim < 2 || dim > 3 {
        *n = 0;
        *points = Vec::new();
        return;
    }

    if fname == "stdin" || fname == "-" {
        f = ImodFile::Stdin;
        from_stdin = true;
    } else {
        match ImodFile::open(fname, "r") {
            Some(h) => {
                f = h;
                from_stdin = false;
            }
            None => {
                // `strerror(errno)`. `std::io::Error` renders the same text
                // with a " (os error N)" suffix std adds and the C library
                // does not, so the suffix is trimmed back off.
                let e = std::io::Error::last_os_error().to_string();
                let e = e.split(" (os error ").next().unwrap_or(&e).to_string();
                nn_quit(&c_format("%s: %s\n", &[CArg::Str(fname), CArg::Str(&e)]))
            }
        }
    }

    *points = vec![Point::default(); nallocated as usize];
    *n = 0;

    let mut all = Vec::<u8>::new();
    let _ = std::io::Read::read_to_end(&mut f, &mut all);
    let mut pos = 0usize;

    while pos < all.len() {
        // `fgets`: at most BUFSIZE-1 bytes, stopping after a newline.
        let mut stop = pos;
        let limit = (pos + BUFSIZE - 1).min(all.len());
        while stop < limit {
            let c = all[stop];
            stop += 1;
            if c == b'\n' {
                break;
            }
        }
        let mut buf: &[u8] = &all[pos..stop];
        pos = stop;

        // The C string functions below stop at the first NUL in the buffer.
        if let Some(z) = buf.iter().position(|c| *c == 0) {
            buf = &buf[..z];
        }

        if *n == nallocated {
            nallocated *= 2;
            points.resize(nallocated as usize, Point::default());
        }

        if !buf.is_empty() && buf[0] == b'#' {
            continue;
        }

        let mut x = 0.0;
        let mut y = 0.0;
        let z;
        let mut cur = 0usize;

        // `token = strtok(buf, seps)` — skip separators, then take the run of
        // non-separators. `strtod` reads ASCII only, so a token is handed over
        // up to its first non-ASCII byte, which is where `strtod` would stop.
        while cur < buf.len() && seps.contains(&buf[cur]) {
            cur += 1;
        }
        if cur >= buf.len() {
            continue;
        }
        let mut start = cur;
        while cur < buf.len() && !seps.contains(&buf[cur]) {
            cur += 1;
        }
        let mut tok = &buf[start..cur];
        tok = &tok[..tok.iter().position(|c| !c.is_ascii()).unwrap_or(tok.len())];
        if str2double(std::str::from_utf8(tok).unwrap_or(""), &mut x) == 0 {
            continue;
        }

        // `token = strtok(NULL, seps)`
        while cur < buf.len() && seps.contains(&buf[cur]) {
            cur += 1;
        }
        if cur >= buf.len() {
            continue;
        }
        start = cur;
        while cur < buf.len() && !seps.contains(&buf[cur]) {
            cur += 1;
        }
        tok = &buf[start..cur];
        tok = &tok[..tok.iter().position(|c| !c.is_ascii()).unwrap_or(tok.len())];
        if str2double(std::str::from_utf8(tok).unwrap_or(""), &mut y) == 0 {
            continue;
        }

        if dim == 2 {
            z = f64::NAN;
        } else {
            // `token = strtok(NULL, seps)`
            while cur < buf.len() && seps.contains(&buf[cur]) {
                cur += 1;
            }
            if cur >= buf.len() {
                continue;
            }
            start = cur;
            while cur < buf.len() && !seps.contains(&buf[cur]) {
                cur += 1;
            }
            tok = &buf[start..cur];
            tok = &tok[..tok.iter().position(|c| !c.is_ascii()).unwrap_or(tok.len())];
            let mut zz = 0.0;
            if str2double(std::str::from_utf8(tok).unwrap_or(""), &mut zz) == 0 {
                continue;
            }
            z = zz;
        }
        points[*n as usize] = Point { x, y, z };
        *n += 1;
    }

    if *n == 0 {
        *points = Vec::new();
    } else {
        points.truncate(*n as usize);
    }

    if !from_stdin {
        // `fclose(f)`: dropping the handle closes it. `std::fs::File` surfaces
        // no close error, so the source's `nn_quit` on a failing `fclose`
        // cannot fire here.
        drop(f);
    }
}

/// Original `points_scaletosquare` (`nncommon.c:525`).
pub fn points_scaletosquare(n: i32, points: &mut [Point]) -> f64 {
    let (mut xmin, mut ymin, mut xmax, mut ymax): (f64, f64, f64, f64);
    let k: f64;

    if n <= 0 {
        return f64::NAN;
    }

    xmin = points[0].x;
    xmax = xmin;
    ymin = points[0].y;
    ymax = ymin;

    for i in 1..n {
        let p = &points[i as usize];

        if p.x < xmin {
            xmin = p.x;
        } else if p.x > xmax {
            xmax = p.x;
        }
        if p.y < ymin {
            ymin = p.y;
        } else if p.y > ymax {
            ymax = p.y;
        }
    }

    if xmin == xmax || ymin == ymax {
        return f64::NAN;
    } else {
        k = (ymax - ymin) / (xmax - xmin);
    }

    for i in 0..n {
        points[i as usize].y /= k;
    }

    k
}

/// Original `points_scale` (`nncommon.c:567`).
pub fn points_scale(n: i32, points: &mut [Point], k: f64) {
    for i in 0..n {
        points[i as usize].y /= k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_grid_range_circle_and_scale_paths_are_preserved() {
        let mut count = 0;
        let mut points: Vec<Point> = Vec::new();
        points_generate(0.0, 2.0, 4.0, 6.0, 2, 2, &mut count, &mut points);
        assert_eq!(count, 4);
        assert_eq!((points[0].x, points[3].y), (0.0, 6.0));
        let mut xmin = f64::NAN;
        let mut xmax = f64::NAN;
        let mut ymin = f64::NAN;
        let mut ymax = f64::NAN;
        points_getrange(
            count, &points, 2.0, &mut xmin, &mut xmax, &mut ymin, &mut ymax,
        );
        assert_eq!((xmin, xmax, ymin, ymax), (-1.0, 3.0, 3.0, 7.0));
        let circle = Circle {
            x: 1.0,
            y: 1.0,
            r: 1.0,
        };
        assert_eq!(circle_contains(&circle, &points[0]), 0);
        let mut square_points = [
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point {
                x: 2.0,
                y: 4.0,
                z: 0.0,
            },
        ];
        assert_eq!(points_scaletosquare(2, &mut square_points), 2.0);
        assert_eq!(square_points[1].y, 2.0);
    }

    #[test]
    fn reads_the_vendored_imod_columnar_fixture_with_source_token_rules() {
        let mut count = -1;
        let mut points: Vec<Point> = Vec::new();
        points_read(
            "IMOD/Etomo/uitestData/BB/BBafid_simple-align.xyz",
            3,
            &mut count,
            &mut points,
        );
        assert_eq!(count, 23);
        assert_eq!(
            (points[0].x, points[0].y, points[0].z),
            (1.0, 244.53, 242.58)
        );
    }

    #[test]
    fn str2double_takes_the_longest_valid_prefix_like_strtod() {
        let mut v = 0.0;
        assert_eq!(str2double("  -12.5e2abc", &mut v), 1);
        assert_eq!(v, -1250.0);
        assert_eq!(str2double("0x10", &mut v), 1);
        assert_eq!(v, 16.0);
        assert_eq!(str2double("1e", &mut v), 1);
        assert_eq!(v, 1.0);
        assert_eq!(str2double("abc", &mut v), 0);
        assert!(v.is_nan());
        assert_eq!(str2double("-inf", &mut v), 1);
        assert_eq!(v, f64::NEG_INFINITY);
    }
}
