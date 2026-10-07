//! Native goldens stored as a manifest instead of as files.
//!
//! A suite used to keep every native output under `fixtures/<suite>/golden/`
//! (and `defined/`) byte for byte.  They now live in one file per suite,
//! `fixtures/<suite>/golden.manifest` (`fixtures/golden.manifest` for the loose
//! files at the top of `fixtures/`), keyed by the path the file used to have
//! relative to that directory — `golden/<case>.stdout`, `defined/x.out`, ….
//! Each entry is either
//!
//! - `text <len> <key>` followed by the `<len>` raw bytes and a newline: kept
//!   whole, for exit statuses, short standard output and anything a test reads
//!   rather than only compares, so a failure can show both sides; or
//! - `sha256 <hex> <len> <key>`: the SHA-256 of the *expected* bytes after the
//!   same masking the suite applies to both sides (`mask_stamps`, …) and, where
//!   the suite reconciles native's uninitialised regions, after substituting
//!   the defined values — i.e. exactly what our masked output must equal.
//!
//! Tests reach a golden through [`load`] with the path it used to have, so the
//! old layout is still the addressing scheme.  That is also how the manifest is
//! (re)made: with `IMOD_RS_GOLDEN_RECORD=1` set, [`load`] reads the real file at
//! that path — written there by `fixtures/make-<suite>-goldens.sh` — every
//! comparison runs exactly as the old file-based test did, and each entry is
//! recorded into the manifest as it is used.  `fixtures/regen-golden.sh <suite>`
//! does the three steps (make native outputs, record, delete them); with
//! `KEEP=1` it leaves the native files in place for debugging a mismatch.
//!
//! **Tolerance mode** (`PORTABILITY.md`).  Byte parity with native IMOD is the
//! acceptance target on Linux x86_64 only.  Built for any other target (or
//! with `IMOD_RS_GOLDEN_TOLERANCE=1`, which exists to exercise the mechanism
//! on Linux x86_64), a comparison that fails byte for byte is retried as a
//! numeric comparison, with the tolerance stated in [`REL_TOL`]:
//!
//! - an **MRC file** must have the same length and the same header (with the
//!   four statistics fields `amin`/`amax`/`amean`/`rms` compared numerically
//!   instead), and every data value must agree within the tolerance (plus one
//!   count for an integer mode, the step a float rounded near .5 can take);
//! - **text** must have the same words, and every number in it must agree
//!   within the tolerance or within one unit of its last printed decimal;
//!   integers without a decimal point must be equal;
//! - anything else (TIFF, model files, ...) is still compared exactly.
//!
//! A golden stored whole is compared value by value.  A golden stored only as
//! a digest has no values to compare against, so the manifest is backed by a
//! second file, `fixtures/golden-approx/<suite>.approx`, holding a numeric
//! fingerprint of each digest entry: the header and word skeleton exactly (as
//! digests), and the values summed over 16 blocks.  The block sums must then agree within
//! the tolerance scaled by the block's absolute sum, plus two of the block's
//! largest last-digit units (or integer steps).  The fingerprints are
//! made from our own Linux x86_64 output whenever it matches the digest
//! exactly and `IMOD_RS_GOLDEN_FINGERPRINT=1` is set (the expected bytes are
//! then known exactly), and from the native files when recording.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Largest text golden stored whole when a test only compares it.
pub const INLINE_MAX: usize = 512;

#[derive(Clone, Debug, PartialEq)]
enum Entry {
    Text(Vec<u8>),
    Digest([u8; 32], usize),
}

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// The manifest a golden path belongs to, and its key there.
fn locate(path: &Path) -> (PathBuf, String) {
    let root = fixtures_root();
    let relative = path
        .strip_prefix(&root)
        .unwrap_or_else(|_| panic!("golden {} is not under {}", path.display(), root.display()));
    let parts: Vec<String> = relative
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.len() > 1 {
        (
            root.join(&parts[0]).join("golden.manifest"),
            parts[1..].join("/"),
        )
    } else {
        (root.join("golden.manifest"), parts.join("/"))
    }
}

/// Name of the suite a manifest belongs to (its directory under `fixtures/`).
fn suite_of(manifest: &Path) -> String {
    let parent = manifest.parent().unwrap();
    if parent == fixtures_root() {
        ".".to_owned()
    } else {
        parent.file_name().unwrap().to_string_lossy().into_owned()
    }
}

/// Whether goldens are read from the native files on disk instead of the
/// manifest: when recording (`IMOD_RS_GOLDEN_RECORD=1`), and for an ad hoc
/// native differential (`IMOD_RS_GOLDEN_NATIVE=1`, which writes nothing).
pub fn recording() -> bool {
    let set = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty() && v != "0");
    set("IMOD_RS_GOLDEN_RECORD") || set("IMOD_RS_GOLDEN_NATIVE")
}

/// Whether recorded entries are written to the manifest.
fn writing() -> bool {
    std::env::var_os("IMOD_RS_GOLDEN_RECORD").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether `#full` rows of a `cases.tsv` run too (`IMOD_RS_FULL_CASES=1`).
pub fn full_cases() -> bool {
    std::env::var_os("IMOD_RS_FULL_CASES").is_some_and(|v| !v.is_empty() && v != "0")
}

/// The case rows of a `cases.tsv`: every line that is neither empty nor a
/// comment.  A row pruned from the ordinary run is kept in the table as
/// `#full<TAB>row` so the exhaustive native differential can still run it
/// (`fixtures/README.md`, "Pruned cases"); with `IMOD_RS_FULL_CASES=1` those
/// rows are included.
pub fn case_rows(table: &str) -> Vec<&str> {
    let full = full_cases();
    table
        .lines()
        .filter_map(|line| match line.strip_prefix("#full\t") {
            Some(row) if full => Some(row),
            _ => Some(line),
        })
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// With `IMOD_RS_GOLDEN_ACCESS_LOG=<file>`, every golden a test looks up is
/// appended there as `<manifest>\t<key>` -- how `fixtures/prune-manifest.py`
/// finds the entries no remaining case uses.
fn log_access(manifest: &Path, key: &str) {
    if let Some(log) = std::env::var_os("IMOD_RS_GOLDEN_ACCESS_LOG") {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .unwrap();
        writeln!(file, "{}\t{key}", manifest.display()).unwrap();
    }
}

static MANIFESTS: Mutex<BTreeMap<PathBuf, BTreeMap<String, Entry>>> = Mutex::new(BTreeMap::new());

/// The relative tolerance of a comparison in tolerance mode (see the module
/// documentation).  Results that only differ through the platform's `libm`,
/// or through x86 rounding quirks the scalar code reproduces, stay orders of
/// magnitude inside it; a translation defect normally does not.
pub const REL_TOL: f64 = 1e-4;

/// Whether a failed byte comparison is retried numerically: on every target
/// except Linux x86_64, or when `IMOD_RS_GOLDEN_TOLERANCE=1`.
pub fn tolerance_mode() -> bool {
    !cfg!(all(target_os = "linux", target_arch = "x86_64"))
        || std::env::var_os("IMOD_RS_GOLDEN_TOLERANCE").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether exact matches of digest entries record their fingerprints
/// (`IMOD_RS_GOLDEN_FINGERPRINT=1`).
fn fingerprinting() -> bool {
    std::env::var_os("IMOD_RS_GOLDEN_FINGERPRINT").is_some_and(|v| !v.is_empty() && v != "0")
}

static APPROX: Mutex<BTreeMap<PathBuf, BTreeMap<String, String>>> = Mutex::new(BTreeMap::new());

/// `fixtures/golden-approx/<suite>.approx` (`_top.approx` for suite `.`):
/// outside the suite directories, which several suites copy into each case's
/// working directory.
fn approx_path(manifest: &Path) -> PathBuf {
    let suite = suite_of(manifest);
    let name = if suite == "." {
        "_top".to_owned()
    } else {
        suite
    };
    fixtures_root()
        .join("golden-approx")
        .join(format!("{name}.approx"))
}

fn with_approx<R>(manifest: &Path, body: impl FnOnce(&mut BTreeMap<String, String>) -> R) -> R {
    let mut all = APPROX.lock().unwrap_or_else(|e| e.into_inner());
    let path = approx_path(manifest);
    let map = all.entry(path.clone()).or_insert_with(|| {
        let mut map = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once('\t') {
                    map.insert(key.to_owned(), value.to_owned());
                }
            }
        }
        map
    });
    body(map)
}

/// Records the fingerprint of `expected` (the masked bytes a digest entry
/// stands for) in the suite's approx file.
fn record_fingerprint(manifest: &Path, key: &str, expected: &[u8]) {
    let Some(print) = fingerprint(expected) else {
        return;
    };
    let path = approx_path(manifest);
    with_approx(manifest, |map| {
        if map.get(key) == Some(&print) {
            return;
        }
        map.insert(key.to_owned(), print);
        let mut out = format!(
            "# imod-rs numeric fingerprints of the digest entries in golden.manifest, for\n\
             # the tolerance mode of tests/common/golden.rs (PORTABILITY.md).  Made by a\n\
             # replay with IMOD_RS_GOLDEN_FINGERPRINT=1 or by recording.\n"
        );
        for (key, value) in map.iter() {
            out.push_str(&format!("{key}\t{value}\n"));
        }
        let temporary = path.with_extension(format!("approx.{}", std::process::id()));
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        std::fs::write(&temporary, out).unwrap();
        std::fs::rename(&temporary, &path).unwrap();
    });
}

/// An MRC file's layout, when `bytes` is one: header length (1024 plus the
/// extended header), mode and value count.
fn mrc_layout(bytes: &[u8]) -> Option<(usize, i32, usize)> {
    if bytes.len() < 1024 || &bytes[208..212] != b"MAP " {
        return None;
    }
    let int = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (nx, ny, nz, mode, next) = (int(0), int(4), int(8), int(12), int(92));
    if nx <= 0 || ny <= 0 || nz <= 0 || next < 0 {
        return None;
    }
    let pixels = nx as usize * ny as usize * nz as usize;
    let (values, size) = match mode {
        0 => (pixels, pixels),
        1 | 6 | 12 => (pixels, 2 * pixels),
        2 => (pixels, 4 * pixels),
        3 => (2 * pixels, 4 * pixels),
        4 => (2 * pixels, 8 * pixels),
        16 => (3 * pixels, 3 * pixels),
        101 => (
            pixels,
            (nx as usize).div_ceil(2) * ny as usize * nz as usize,
        ),
        _ => return None,
    };
    let header = 1024 + next as usize;
    (header + size == bytes.len()).then_some((header, mode, values))
}

/// The data values of an MRC file, as `f64`.
fn mrc_values(bytes: &[u8], header: usize, mode: i32, count: usize) -> Vec<f64> {
    let data = &bytes[header..];
    let half = |bits: u16| -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let fraction = (bits & 0x3ff) as f64;
        sign * match exponent {
            0 => fraction * 2f64.powi(-24),
            31 => {
                if fraction == 0.0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
        }
    };
    match mode {
        0 | 16 => data.iter().map(|&b| b as f64).collect(),
        1 | 3 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        6 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        12 => data
            .chunks_exact(2)
            .map(|c| half(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 | 4 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        _ => {
            // 101: 4-bit values, low nibble first, rows padded to a byte
            let mut values = Vec::with_capacity(count);
            for &b in data {
                values.push((b & 0x0f) as f64);
                values.push((b >> 4) as f64);
            }
            values.truncate(count);
            values
        }
    }
}

/// The header with its four statistics fields (`amin`, `amax`, `amean`,
/// `rms`) zeroed, and those four values.
fn mrc_header_parts(bytes: &[u8], header: usize) -> (Vec<u8>, [f64; 4]) {
    let mut head = bytes[..header].to_vec();
    let mut stats = [0.0; 4];
    for (slot, at) in stats.iter_mut().zip([76usize, 80, 84, 216]) {
        *slot = f32::from_le_bytes(head[at..at + 4].try_into().unwrap()) as f64;
        head[at..at + 4].fill(0);
    }
    (head, stats)
}

/// Words and numbers of a text: the text with each number replaced by `#`
/// and runs of blanks collapsed, and each number with the unit of its last
/// printed decimal (0 for an integer without a decimal point or exponent).
fn text_parts(text: &str) -> (String, Vec<(f64, f64)>) {
    let chars: Vec<char> = text.chars().collect();
    let mut skeleton = String::new();
    let mut numbers = Vec::new();
    let mut at = 0;
    let mut blank = false;
    while at < chars.len() {
        let c = chars[at];
        // A number: [-+]digits[.digits][(e|E|d|D)[-+]digits], or .digits
        let start = at;
        let mut k = at;
        if (c == '-' || c == '+') && k + 1 < chars.len() {
            k += 1;
        }
        let digits_start = k;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        let mut int_digits = k - digits_start;
        let mut decimals: Option<i32> = None;
        if k < chars.len() && chars[k] == '.' {
            let mut m = k + 1;
            while m < chars.len() && chars[m].is_ascii_digit() {
                m += 1;
            }
            if int_digits > 0 || m > k + 1 {
                decimals = Some((m - k - 1) as i32);
                int_digits += 1;
                k = m;
            }
        }
        let preceded_by_word =
            start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_');
        if int_digits == 0 || preceded_by_word {
            if c.is_whitespace() {
                if !blank {
                    skeleton.push(' ');
                }
                blank = true;
            } else {
                skeleton.push(c);
                blank = false;
            }
            at += 1;
            continue;
        }
        let mut exponent = 0i32;
        let mut has_exponent = false;
        if k < chars.len() && matches!(chars[k], 'e' | 'E' | 'd' | 'D') {
            let mut m = k + 1;
            if m < chars.len() && (chars[m] == '-' || chars[m] == '+') {
                m += 1;
            }
            let e_start = m;
            while m < chars.len() && chars[m].is_ascii_digit() {
                m += 1;
            }
            if m > e_start {
                let text: String = chars[k + 1..m].iter().collect();
                exponent = text.parse().unwrap_or(0);
                has_exponent = true;
                k = m;
            }
        }
        let token: String = chars[start..k]
            .iter()
            .map(|&c| if c == 'd' || c == 'D' { 'e' } else { c })
            .collect();
        let Ok(value) = token.parse::<f64>() else {
            skeleton.push(c);
            blank = false;
            at += 1;
            continue;
        };
        let unit = match (decimals, has_exponent) {
            (Some(d), _) => 10f64.powi(exponent - d),
            (None, true) => 10f64.powi(exponent),
            (None, false) => 0.0,
        };
        skeleton.push('#');
        blank = false;
        numbers.push((value, unit));
        at = k;
    }
    (skeleton, numbers)
}

/// `|a - b|` within `REL_TOL` of the larger magnitude, or within `slack`.
fn close(a: f64, b: f64, slack: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= (REL_TOL * a.abs().max(b.abs())).max(slack) * (1.0 + 1e-9)
}

/// Value-by-value comparison of two whole files (see the module comment).
pub fn approx_equal(expected: &[u8], ours: &[u8]) -> Result<(), String> {
    if let (Some((header, mode, count)), Some(mine)) = (mrc_layout(expected), mrc_layout(ours)) {
        if (header, mode, count) != mine || expected.len() != ours.len() {
            return Err("MRC layout differs".to_owned());
        }
        let (head_a, stats_a) = mrc_header_parts(expected, header);
        let (head_b, stats_b) = mrc_header_parts(ours, header);
        if head_a != head_b {
            return Err("MRC header differs outside the statistics fields".to_owned());
        }
        let scale = stats_a.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (k, (a, b)) in stats_a.iter().zip(stats_b.iter()).enumerate() {
            if !close(*a, *b, REL_TOL * scale) {
                return Err(format!("MRC header statistic {k}: {a} vs {b}"));
            }
        }
        let slack = if matches!(mode, 2 | 4 | 12) { 0.0 } else { 1.0 };
        let va = mrc_values(expected, header, mode, count);
        let vb = mrc_values(ours, header, mode, count);
        for (k, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
            if !close(*a, *b, slack) {
                return Err(format!("MRC value {k}: {a} vs {b}"));
            }
        }
        return Ok(());
    }
    if let (Ok(a), Ok(b)) = (std::str::from_utf8(expected), std::str::from_utf8(ours)) {
        let (skeleton_a, numbers_a) = text_parts(a);
        let (skeleton_b, numbers_b) = text_parts(b);
        if skeleton_a != skeleton_b || numbers_a.len() != numbers_b.len() {
            return Err("text differs outside its numbers".to_owned());
        }
        for (k, ((a, ua), (b, ub))) in numbers_a.iter().zip(numbers_b.iter()).enumerate() {
            if !close(*a, *b, ua.max(*ub)) {
                return Err(format!("number {k}: {a} vs {b}"));
            }
        }
        return Ok(());
    }
    Err("not MRC or text, compared exactly".to_owned())
}

const BLOCKS: usize = 16;

/// Per-block (sum, sum of magnitudes, largest last-digit unit, count, NaN
/// count) of `values`, split into `BLOCKS` equal runs.
fn block_sums(values: &[(f64, f64)]) -> Vec<[f64; 5]> {
    let mut blocks = vec![[0.0; 5]; BLOCKS];
    let n = values.len().max(1);
    for (k, (value, unit)) in values.iter().enumerate() {
        let block = &mut blocks[k * BLOCKS / n];
        if value.is_nan() {
            block[4] += 1.0;
        } else {
            block[0] += value;
            block[1] += value.abs();
        }
        block[2] = block[2].max(*unit);
        block[3] += 1.0;
    }
    blocks
}

/// The fingerprint of a digest entry's expected bytes, or `None` for a file
/// that is neither MRC nor text.
pub fn fingerprint(expected: &[u8]) -> Option<String> {
    let render = |blocks: Vec<[f64; 5]>| -> String {
        blocks
            .iter()
            .map(|b| format!("{:?} {:?} {:?} {} {}", b[0], b[1], b[2], b[3], b[4]))
            .collect::<Vec<_>>()
            .join(" ")
    };
    if let Some((header, mode, count)) = mrc_layout(expected) {
        let (head, stats) = mrc_header_parts(expected, header);
        let slack = if matches!(mode, 2 | 4 | 12) { 0.0 } else { 1.0 };
        let values: Vec<(f64, f64)> = mrc_values(expected, header, mode, count)
            .into_iter()
            .map(|v| (v, slack))
            .collect();
        return Some(format!(
            "mrc {} {} {:?} {:?} {:?} {:?} {}",
            expected.len(),
            hex(&sha256(&head)),
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            render(block_sums(&values))
        ));
    }
    let text = std::str::from_utf8(expected).ok()?;
    let (skeleton, numbers) = text_parts(text);
    Some(format!(
        "text {} {} {}",
        hex(&sha256(skeleton.as_bytes())),
        numbers.len(),
        render(block_sums(&numbers))
    ))
}

/// Compares `ours` (masked) with a stored fingerprint.
pub fn approx_fingerprint(print: &str, ours: &[u8]) -> Result<(), String> {
    let fields: Vec<&str> = print.split(' ').collect();
    let blocks_of = |fields: &[&str]| -> Vec<[f64; 5]> {
        fields
            .chunks(5)
            .map(|c| {
                let mut b = [0.0; 5];
                for (slot, text) in b.iter_mut().zip(c) {
                    *slot = text.parse().unwrap_or(f64::NAN);
                }
                b
            })
            .collect()
    };
    let check = |want: Vec<[f64; 5]>, got: Vec<[f64; 5]>| -> Result<(), String> {
        for (k, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            if w[3] != g[3] || w[4] != g[4] {
                return Err(format!("block {k}: value or NaN count differs"));
            }
            // Two values in a block may sit on opposite sides of a rounding
            // boundary of their last printed digit (or integer step)
            let slack = REL_TOL * w[1].max(g[1]) + 2.0 * w[2].max(g[2]);
            if (w[0] - g[0]).abs() > slack * (1.0 + 1e-9)
                || (w[1] - g[1]).abs() > slack * (1.0 + 1e-9)
            {
                return Err(format!(
                    "block {k}: sum {} vs {} (slack {slack})",
                    w[0], g[0]
                ));
            }
        }
        Ok(())
    };
    match fields.first() {
        Some(&"mrc") => {
            let Some((header, mode, count)) = mrc_layout(ours) else {
                return Err("not an MRC file".to_owned());
            };
            if fields[1].parse::<usize>().ok() != Some(ours.len()) {
                return Err("MRC length differs".to_owned());
            }
            let (head, stats) = mrc_header_parts(ours, header);
            if hex(&sha256(&head)) != fields[2] {
                return Err("MRC header differs outside the statistics fields".to_owned());
            }
            let want: Vec<f64> = fields[3..7]
                .iter()
                .map(|t| t.parse().unwrap_or(f64::NAN))
                .collect();
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            for k in 0..4 {
                if !close(want[k], stats[k], REL_TOL * scale) {
                    return Err(format!(
                        "MRC header statistic {k}: {} vs {}",
                        want[k], stats[k]
                    ));
                }
            }
            let slack = if matches!(mode, 2 | 4 | 12) { 0.0 } else { 1.0 };
            let values: Vec<(f64, f64)> = mrc_values(ours, header, mode, count)
                .into_iter()
                .map(|v| (v, slack))
                .collect();
            check(blocks_of(&fields[7..]), block_sums(&values))
        }
        Some(&"text") => {
            let text = std::str::from_utf8(ours).map_err(|_| "not text".to_owned())?;
            let (skeleton, numbers) = text_parts(text);
            if hex(&sha256(skeleton.as_bytes())) != fields[1]
                || fields[2].parse::<usize>().ok() != Some(numbers.len())
            {
                return Err("text differs outside its numbers".to_owned());
            }
            check(blocks_of(&fields[3..]), block_sums(&numbers))
        }
        _ => Err("unreadable fingerprint".to_owned()),
    }
}

fn parse(bytes: &[u8], manifest: &Path) -> BTreeMap<String, Entry> {
    let mut map = BTreeMap::new();
    let mut at = 0;
    while at < bytes.len() {
        let end = bytes[at..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |p| at + p);
        let line = std::str::from_utf8(&bytes[at..end])
            .unwrap_or_else(|_| panic!("{}: header line not UTF-8", manifest.display()));
        at = end + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.splitn(4, ' ');
        match fields.next() {
            Some("text") => {
                let len: usize = fields.next().unwrap().parse().unwrap();
                let key = fields.collect::<Vec<_>>().join(" ");
                let content = bytes[at..at + len].to_vec();
                assert_eq!(
                    bytes.get(at + len),
                    Some(&b'\n'),
                    "{}: {key}",
                    manifest.display()
                );
                at += len + 1;
                map.insert(key, Entry::Text(content));
            }
            Some("sha256") => {
                let hex = fields.next().unwrap();
                let len: usize = fields.next().unwrap().parse().unwrap();
                let key = fields.next().unwrap().to_owned();
                let mut digest = [0u8; 32];
                for (k, byte) in digest.iter_mut().enumerate() {
                    *byte = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap();
                }
                map.insert(key, Entry::Digest(digest, len));
            }
            other => panic!("{}: bad manifest line {other:?}", manifest.display()),
        }
    }
    map
}

fn with_manifest<R>(manifest: &Path, body: impl FnOnce(&mut BTreeMap<String, Entry>) -> R) -> R {
    let mut all = MANIFESTS.lock().unwrap_or_else(|e| e.into_inner());
    let map = all
        .entry(manifest.to_path_buf())
        .or_insert_with(|| match std::fs::read(manifest) {
            Ok(bytes) => parse(&bytes, manifest),
            Err(_) => BTreeMap::new(),
        });
    body(map)
}

fn write_manifest(manifest: &Path, map: &BTreeMap<String, Entry>) {
    let mut out = Vec::new();
    out.extend_from_slice(
        format!(
            "# imod-rs native golden manifest for fixtures/{}; see fixtures/README.md.\n\
             # Regenerate with fixtures/regen-golden.sh {}; do not edit by hand.\n",
            suite_of(manifest),
            suite_of(manifest)
        )
        .as_bytes(),
    );
    for (key, entry) in map {
        match entry {
            Entry::Text(content) => {
                out.extend_from_slice(format!("text {} {key}\n", content.len()).as_bytes());
                out.extend_from_slice(content);
                out.push(b'\n');
            }
            Entry::Digest(digest, len) => {
                out.extend_from_slice(format!("sha256 {} {len} {key}\n", hex(digest)).as_bytes());
            }
        }
    }
    let temporary = manifest.with_extension(format!("manifest.{}", std::process::id()));
    std::fs::write(&temporary, &out).unwrap();
    std::fs::rename(&temporary, manifest).unwrap();
}

/// Record `entry` for `key`; an entry stored whole is never downgraded to a digest.
fn record(manifest: &Path, key: &str, entry: Entry, force: bool) {
    if !writing() {
        return;
    }
    with_manifest(manifest, |map| {
        if !force && matches!(map.get(key), Some(Entry::Text(_))) {
            if let Entry::Digest(..) = entry {
                return;
            }
        }
        map.insert(key.to_owned(), entry);
        write_manifest(manifest, map);
    });
}

fn is_small_text(bytes: &[u8]) -> bool {
    bytes.len() <= INLINE_MAX && !bytes.contains(&0) && std::str::from_utf8(bytes).is_ok()
}

/// A native golden, as found in the manifest (or, when recording, on disk).
#[derive(Clone, Debug)]
pub struct Golden {
    manifest: PathBuf,
    key: String,
    entry: Entry,
}

/// Whether a golden exists at `path`.
pub fn exists(path: &Path) -> bool {
    load(path).is_some()
}

/// The golden that used to live at `path`, if there is one.
pub fn load(path: &Path) -> Option<Golden> {
    let (manifest, key) = locate(path);
    log_access(&manifest, &key);
    if recording() {
        // `defined/` expectations come from our own build, not the native
        // run, so a regeneration that did not remake them keeps them.
        let Ok(bytes) = std::fs::read(path) else {
            if key.starts_with("defined/") {
                let entry = with_manifest(&manifest, |map| map.get(&key).cloned())?;
                return Some(Golden {
                    manifest,
                    key,
                    entry,
                });
            }
            return None;
        };
        let entry = if is_small_text(&bytes) {
            Entry::Text(bytes.clone())
        } else {
            Entry::Digest(sha256(&bytes), bytes.len())
        };
        let already = with_manifest(&manifest, |map| map.contains_key(&key));
        if !already {
            record(&manifest, &key, entry, false);
        }
        return Some(Golden {
            manifest,
            key,
            entry: Entry::Text(bytes),
        });
    }
    let entry = with_manifest(&manifest, |map| map.get(&key).cloned())?;
    Some(Golden {
        manifest,
        key,
        entry,
    })
}

/// [`load`], panicking with the path when there is no such golden.
pub fn expect(path: &Path) -> Golden {
    load(path).unwrap_or_else(|| {
        let (manifest, key) = locate(path);
        panic!(
            "no golden {key} in {} (regenerate: fixtures/regen-golden.sh {})",
            manifest.display(),
            suite_of(&manifest)
        )
    })
}

/// The whole golden at `path` (it must be stored whole): what `std::fs::read` returned.
pub fn read(path: &Path) -> Vec<u8> {
    expect(path).bytes().to_vec()
}

/// [`read`] as UTF-8 text.
pub fn read_to_string(path: &Path) -> String {
    String::from_utf8(read(path)).expect("golden is not UTF-8")
}

/// [`read`], or `None` when there is no such golden.
pub fn read_opt(path: &Path) -> Option<Vec<u8>> {
    load(path).map(|g| g.bytes().to_vec())
}

/// Keys (file names) of every golden directly inside directory `dir`, sorted.
pub fn list(dir: &Path) -> Vec<String> {
    if recording() {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        return names;
    }
    let (manifest, key) = locate(&dir.join("x"));
    let prefix = &key[..key.len() - 1];
    with_manifest(&manifest, |map| {
        let mut names: Vec<String> = map
            .keys()
            .filter_map(|k| k.strip_prefix(prefix))
            .map(|rest| rest.split('/').next().unwrap().to_owned())
            .collect();
        names.dedup();
        names
    })
}

/// The no-op mask, for exact comparisons.
pub fn identity(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

impl Golden {
    /// The whole golden.  Recording marks it to be stored whole; replaying a
    /// golden stored only as a digest panics, naming how to keep it whole.
    pub fn bytes(&self) -> &[u8] {
        match &self.entry {
            Entry::Text(bytes) => {
                if recording() {
                    record(&self.manifest, &self.key, Entry::Text(bytes.clone()), true);
                }
                bytes
            }
            Entry::Digest(..) => panic!(
                "golden {} in {} is stored as a digest, but the test reads its content; \
                 re-record the suite (fixtures/regen-golden.sh {})",
                self.key,
                self.manifest.display(),
                suite_of(&self.manifest)
            ),
        }
    }

    /// [`Golden::bytes`] as lossy UTF-8.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(self.bytes()).into_owned()
    }

    /// Length of the native file.
    pub fn len(&self) -> usize {
        match &self.entry {
            Entry::Text(bytes) => bytes.len(),
            Entry::Digest(_, len) => *len,
        }
    }

    /// Exact comparison.
    pub fn matches(&self, ours: &[u8]) -> bool {
        self.compare(ours, identity, false).is_ok()
    }

    /// Comparison after applying `mask` to both sides.
    pub fn matches_masked(&self, ours: &[u8], mask: impl Fn(&[u8]) -> Vec<u8>) -> bool {
        self.compare(ours, mask, false).is_ok()
    }

    /// `mask(ours) == mask(reconcile_uninitialised(ours, native))`.
    pub fn matches_reconciled(&self, ours: &[u8], mask: impl Fn(&[u8]) -> Vec<u8>) -> bool {
        self.compare(ours, mask, true).is_ok()
    }

    /// The comparison behind the `matches*` methods; the error describes both
    /// sides (content when stored whole, digest and length otherwise) and how
    /// to get the native file back.
    pub fn compare(
        &self,
        ours: &[u8],
        mask: impl Fn(&[u8]) -> Vec<u8>,
        reconcile: bool,
    ) -> Result<(), String> {
        let mine = mask(ours);
        match &self.entry {
            Entry::Text(native) => {
                let expected = if reconcile {
                    mask(&super::reconcile_uninitialised(ours, native))
                } else {
                    mask(native)
                };
                if recording() {
                    let entry = if is_small_text(native) {
                        Entry::Text(native.clone())
                    } else {
                        Entry::Digest(sha256(&expected), expected.len())
                    };
                    if writing() && matches!(entry, Entry::Digest(..)) {
                        record_fingerprint(&self.manifest, &self.key, &expected);
                    }
                    record(&self.manifest, &self.key, entry, false);
                }
                if mine == expected {
                    Ok(())
                } else if tolerance_mode() && approx_equal(&expected, &mine).is_ok() {
                    Ok(())
                } else {
                    Err(self.mismatch(&mine, &shown(&expected)))
                }
            }
            Entry::Digest(digest, len) => {
                if sha256(&mine) == *digest && mine.len() == *len {
                    if fingerprinting() {
                        record_fingerprint(&self.manifest, &self.key, &mine);
                    }
                    Ok(())
                } else if tolerance_mode() {
                    let print = with_approx(&self.manifest, |map| map.get(&self.key).cloned());
                    match print.map(|print| approx_fingerprint(&print, &mine)) {
                        Some(Ok(())) => Ok(()),
                        Some(Err(why)) => Err(format!(
                            "{}\n    (tolerance mode: {why})",
                            self.mismatch(&mine, &format!("sha256 {} ({len} bytes)", hex(digest)))
                        )),
                        None => Err(format!(
                            "{}\n    (tolerance mode: no fingerprint in fixtures/golden-approx)",
                            self.mismatch(&mine, &format!("sha256 {} ({len} bytes)", hex(digest)))
                        )),
                    }
                } else {
                    Err(self.mismatch(&mine, &format!("sha256 {} ({len} bytes)", hex(digest))))
                }
            }
        }
    }

    /// For a suite that accepts a tolerance (LAPACK/BLAS, `CLAUDE.md`):
    /// `tolerant(expected, ours)` runs on the masked bytes when the native
    /// bytes are at hand — stored whole, or while recording — and otherwise
    /// the masked output must match the digest exactly.  So a replay is
    /// strict; a difference is then examined against the real native files
    /// with the tolerance by recording (`KEEP=1 fixtures/regen-golden.sh`).
    pub fn compare_tolerant(
        &self,
        ours: &[u8],
        mask: impl Fn(&[u8]) -> Vec<u8>,
        tolerant: impl Fn(&[u8], &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        match &self.entry {
            Entry::Text(native) => {
                let expected = mask(native);
                if recording() {
                    let entry = if is_small_text(native) {
                        Entry::Text(native.clone())
                    } else {
                        Entry::Digest(sha256(&expected), expected.len())
                    };
                    record(&self.manifest, &self.key, entry, false);
                }
                tolerant(&expected, &mask(ours))
            }
            Entry::Digest(..) => self.compare(ours, mask, false).map_err(|why| {
                format!(
                    "{why}\n    (stored as a digest, so compared exactly; the suite's \
                     tolerance applies when recording against the native files)"
                )
            }),
        }
    }

    fn mismatch(&self, mine: &[u8], expected: &str) -> String {
        let ours = if is_small_text(mine) {
            shown(mine)
        } else {
            format!("sha256 {} ({} bytes)", hex(&sha256(mine)), mine.len())
        };
        format!(
            "golden {} ({}) differs:\n    native: {expected}\n    ours:   {ours}\n    \
             native files: KEEP=1 fixtures/regen-golden.sh {}",
            self.key,
            self.manifest.display(),
            suite_of(&self.manifest)
        )
    }

    /// Whether two goldens hold the same expected bytes (compared by digest
    /// when either is stored as one).
    pub fn same_as(&self, other: &Golden) -> bool {
        let digest = |entry: &Entry| match entry {
            Entry::Text(bytes) => (sha256(bytes), bytes.len()),
            Entry::Digest(digest, len) => (*digest, *len),
        };
        digest(&self.entry) == digest(&other.entry)
    }

    /// One line naming the stored expectation and how to get the native
    /// file back, for a failure message.
    pub fn describe(&self) -> String {
        let (digest, len) = match &self.entry {
            Entry::Text(bytes) => (sha256(bytes), bytes.len()),
            Entry::Digest(digest, len) => (*digest, *len),
        };
        format!(
            "{} sha256 {} ({len} bytes); native files: KEEP=1 fixtures/regen-golden.sh {}",
            self.key,
            hex(&digest),
            suite_of(&self.manifest)
        )
    }

    /// Native side for a failure message: the content when stored whole,
    /// otherwise its digest.
    pub fn display(&self) -> String {
        match &self.entry {
            Entry::Text(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            Entry::Digest(digest, len) => format!(
                "<stored as sha256 {} ({len} bytes); native file: KEEP=1 fixtures/regen-golden.sh {}>",
                hex(digest),
                suite_of(&self.manifest)
            ),
        }
    }
}

/// Bytes for a failure message, as a Rust string literal cut at 4000 characters.
fn shown(bytes: &[u8]) -> String {
    let text = format!("{:?}", String::from_utf8_lossy(bytes));
    if text.len() > 4000 {
        let mut cut = 4000;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}... ({} bytes)", &text[..cut], bytes.len())
    } else {
        text
    }
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// FIPS 180-4 SHA-256.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes(block[4 * t..4 * t + 4].try_into().unwrap());
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = [0u8; 32];
    for (k, word) in h.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}
