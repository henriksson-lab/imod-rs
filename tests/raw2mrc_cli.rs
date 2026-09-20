//! `raw2mrc` command-line parity fixtures (`IMOD/mrc/raw2mrc.c`).
//!
//! The suite drives the translated program the way an IMOD install invokes it
//! and checks the written MRC against what the source's own conversion
//! produces.  With `IMOD_NATIVE_RAW2MRC` pointing at a native `raw2mrc`, the
//! same cases run on both sides and are compared byte for byte, masking only
//! the two regions CLAUDE.md records as unmatchable: the `dd-Mmm-yy HH:MM:SS`
//! stamp inside a written label, and the label slots past `nlabl`, which
//! native leaves as whatever its stack held (`BUGS.md` §2 — a native run of
//! this program leaks its own argv and path text there).

mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-raw2mrc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A deterministic byte pattern: no generator crate, and the same bytes on
/// every platform and run.
fn pattern(len: usize) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn write_raw(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    path
}

/// Blank the regions a native comparison cannot match.
fn mask(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.len() < 1024 {
        return masked;
    }
    let nlabl = i32::from_le_bytes(masked[220..224].try_into().unwrap()).clamp(0, 10) as usize;
    for slot in 0..10 {
        let start = 224 + 80 * slot;
        if slot >= nlabl {
            masked[start..start + 80].fill(b' ');
        } else {
            masked[start + 56..start + 76].fill(b' ');
        }
    }
    masked
}

/// Replace a `Mmm DD YYYY HH:MM:SS` compile stamp with a fixed marker: the
/// two binaries were built at different times and `__DATE__`/`__TIME__` is a
/// documented non-achievable.
fn without_build_stamp(bytes: &[u8]) -> Vec<u8> {
    let months: [&[u8]; 12] = [
        b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov",
        b"Dec",
    ];
    let mut out = bytes.to_vec();
    let mut index = 0;
    while index + 20 <= out.len() {
        let window = &out[index..index + 20];
        let dated = months.contains(&&window[..3])
            && window[3] == b' '
            && window[6] == b' '
            && window[11] == b' '
            && window[14] == b':'
            && window[17] == b':';
        if dated {
            out[index..index + 20].fill(b'#');
            index += 20;
        } else {
            index += 1;
        }
    }
    out
}

/// Every case is `(name, argv)`; the input files are created below.
fn cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "byte",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "byte", "b.raw"],
        ),
        (
            "short",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "short", "s.raw"],
        ),
        (
            "ushort",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "ushort", "s.raw"],
        ),
        (
            "float",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "float", "f.raw"],
        ),
        (
            "rgb",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "rgb", "rgb.raw"],
        ),
        (
            "flip",
            vec![
                "-x", "64", "-y", "48", "-z", "3", "-t", "byte", "-f", "b.raw",
            ],
        ),
        (
            "swap-short",
            vec![
                "-x", "64", "-y", "48", "-z", "3", "-t", "short", "-s", "s.raw",
            ],
        ),
        (
            "offset",
            vec![
                "-x", "64", "-y", "48", "-z", "3", "-t", "byte", "-o", "10", "off.raw",
            ],
        ),
        (
            // `-d` is `DivideBy2`, a switch: unsigned shorts are halved
            // instead of having 32767 subtracted.
            "divide",
            vec![
                "-x", "64", "-y", "48", "-z", "3", "-t", "ushort", "-d", "s.raw",
            ],
        ),
        (
            "invert",
            vec![
                "-x", "64", "-y", "48", "-z", "3", "-t", "byte", "-i", "b.raw",
            ],
        ),
        (
            "one-section",
            vec!["-x", "64", "-y", "48", "-z", "1", "-t", "byte", "b.raw"],
        ),
        (
            // An unknown type is accepted and converted as byte; see the note
            // at the status check below.
            "bad-type",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "bogus", "b.raw"],
        ),
        ("no-size", vec!["-t", "byte", "b.raw"]),
        (
            "missing-input",
            vec![
                "-x",
                "64",
                "-y",
                "48",
                "-z",
                "3",
                "-t",
                "byte",
                "missing.raw",
            ],
        ),
        (
            "short-input",
            vec!["-x", "64", "-y", "48", "-z", "3", "-t", "byte", "trunc.raw"],
        ),
    ]
}

fn make_inputs(dir: &Path) {
    let bytes = pattern(64 * 48 * 3);
    write_raw(dir, "b.raw", &bytes);
    write_raw(dir, "s.raw", &pattern(64 * 48 * 3 * 2));
    write_raw(dir, "f.raw", &pattern(64 * 48 * 3 * 4));
    write_raw(dir, "rgb.raw", &pattern(64 * 48 * 3 * 3));
    let mut offset = b"HEADERJUNK".to_vec();
    offset.extend_from_slice(&bytes);
    write_raw(dir, "off.raw", &offset);
    write_raw(dir, "trunc.raw", &bytes[..1000]);
}

/// The converted volume has the size, mode and pixel values the arguments ask
/// for, and the error paths report rather than writing a file.
#[test]
fn conversions_have_the_requested_geometry_and_error_paths_report() {
    let dir = scratch();
    make_inputs(&dir);

    for (name, argv) in cases() {
        let output = dir.join(format!("{name}.mrc"));
        let _ = std::fs::remove_file(&output);
        let result = common::imod_cmd("raw2mrc")
            .current_dir(&dir)
            .args(&argv)
            .arg(&output)
            .output()
            .unwrap();

        // Three outcomes, each pinned to what the source does.
        //
        // * `missing-input`/`short-input` report and exit non-zero, on
        //   **stdout** (`exitError` routes through `PipSetError`).
        // * `no-size` reaches `PipReadOrParseOptions`'s usage path: it prints
        //   the banner and option list and exits **0**, writing nothing.
        // * `bad-type` succeeds: `setintype` returns -1 (`raw2mrc.c:539`) and
        //   `main` never tests it, so `raw2mrc.c:181-182` falls back to the
        //   byte default and the conversion runs.
        //
        // Native does all three the same way; the native comparison below is
        // what establishes that.
        let failing = matches!(name, "missing-input" | "short-input");
        let usage_only = name == "no-size";
        if usage_only {
            assert!(result.status.success(), "{name}: {result:?}");
            assert!(
                !output.exists(),
                "{name} should not have written an output file"
            );
            continue;
        }
        if failing {
            assert!(
                !result.status.success(),
                "{name} should have failed: {result:?}"
            );
            continue;
        }
        assert!(result.status.success(), "{name}: {result:?}");

        let written = std::fs::read(&output).unwrap();
        assert!(written.len() > 1024, "{name} wrote no data");
        let nx = i32::from_le_bytes(written[0..4].try_into().unwrap());
        let ny = i32::from_le_bytes(written[4..8].try_into().unwrap());
        let nz = i32::from_le_bytes(written[8..12].try_into().unwrap());
        assert_eq!((nx, ny), (64, 48), "{name} size");
        assert_eq!(nz, if name == "one-section" { 1 } else { 3 }, "{name} nz");
        // One label, written by the program itself.
        assert_eq!(
            i32::from_le_bytes(written[220..224].try_into().unwrap()),
            1,
            "{name} nlabl"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `-flip` reverses the line order within each section, so the first output
/// line is the input's last.
#[test]
fn flip_reverses_the_line_order_within_each_section() {
    let dir = scratch();
    make_inputs(&dir);
    let plain = dir.join("plain.mrc");
    let flipped = dir.join("flipped.mrc");
    for (target, extra) in [(&plain, None), (&flipped, Some("-f"))] {
        let mut command = common::imod_cmd("raw2mrc");
        command.current_dir(&dir);
        command.args(["-x", "64", "-y", "48", "-z", "3", "-t", "byte"]);
        if let Some(flag) = extra {
            command.arg(flag);
        }
        let result = command.arg("b.raw").arg(target).output().unwrap();
        assert!(result.status.success(), "{result:?}");
    }
    let a = std::fs::read(&plain).unwrap();
    let b = std::fs::read(&flipped).unwrap();
    let (nx, ny) = (64usize, 48usize);
    for line in 0..ny {
        let from = 1024 + line * nx;
        let to = 1024 + (ny - 1 - line) * nx;
        assert_eq!(&a[from..from + nx], &b[to..to + nx], "line {line}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// With `IMOD_NATIVE_RAW2MRC` set, every case runs on both sides and the
/// written bytes, stdout, stderr and exit status must agree.
#[test]
fn native_comparison_matches_every_case() {
    let Ok(native) = std::env::var("IMOD_NATIVE_RAW2MRC") else {
        return;
    };
    let dir = scratch();
    make_inputs(&dir);

    for (name, argv) in cases() {
        let mut outputs = Vec::new();
        for side in ["nat", "rs"] {
            let output = dir.join(format!("{side}-{name}.mrc"));
            let _ = std::fs::remove_file(&output);
            let result = if side == "nat" {
                Command::new(&native)
                    .current_dir(&dir)
                    .args(&argv)
                    .arg(&output)
                    .output()
                    .unwrap()
            } else {
                common::imod_cmd("raw2mrc")
                    .current_dir(&dir)
                    .args(&argv)
                    .arg(&output)
                    .output()
                    .unwrap()
            };
            outputs.push((result, output));
        }
        let (native_run, native_file) = &outputs[0];
        let (rust_run, rust_file) = &outputs[1];
        assert_eq!(
            native_run.status.code(),
            rust_run.status.code(),
            "{name} exit status"
        );
        // `-help` and the usage paths print the version banner, which carries
        // the compiler's `__DATE__`/`__TIME__`; only its shape is comparable.
        assert_eq!(
            without_build_stamp(&native_run.stdout),
            without_build_stamp(&rust_run.stdout),
            "{name} stdout"
        );
        assert_eq!(
            without_build_stamp(&native_run.stderr),
            without_build_stamp(&rust_run.stderr),
            "{name} stderr"
        );
        assert_eq!(
            native_file.exists(),
            rust_file.exists(),
            "{name} output presence"
        );
        if native_file.exists() {
            let a = mask(&std::fs::read(native_file).unwrap());
            let b = mask(&std::fs::read(rust_file).unwrap());
            assert_eq!(a.len(), b.len(), "{name} output size");
            if a != b {
                // The only known exception is a NaN mean: byte-swapped float
                // input decodes to NaNs, and which payload survives the
                // running sum is a code-generation choice (CLAUDE.md).
                let differing: Vec<usize> =
                    (0..a.len()).filter(|&index| a[index] != b[index]).collect();
                let mean_only = differing.iter().all(|&index| (84..88).contains(&index));
                let native_mean = f32::from_le_bytes(a[84..88].try_into().unwrap());
                let rust_mean = f32::from_le_bytes(b[84..88].try_into().unwrap());
                assert!(
                    mean_only && native_mean.is_nan() && rust_mean.is_nan(),
                    "{name}: {} bytes differ at {:?}",
                    differing.len(),
                    &differing[..differing.len().min(8)]
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
