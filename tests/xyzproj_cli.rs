//! Native-golden coverage for `xyzproj` (`IMOD/flib/image/xyzproj.f90`).
//!
//! Every row of `fixtures/xyzproj/cases.tsv` was run through the native
//! reference program by `fixtures/make-xyzproj-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created.
//! Inputs: seeded float, short and byte volumes and a tilt series written by
//! the native raw2mrc, and tilt-angle files (`make-inputs.sh`).
//! Exit status, standard output and every output file must match byte for
//! byte, except the `hh:mm:ss` time and the date of a `dd-Mmm-yy  hh:mm:ss`
//! stamp in an MRC label, which differ between runs.
//!
//! The cases listed in `fixtures/xyzproj/make-fixed-cases.txt` reach upstream
//! defects the translation fixes (`BUGS.md`, "`xyzproj`"); their goldens are
//! imod-rs output, and the `fixed_*` tests below assert the defined behaviour
//! directly.
//!
//! Pruned 2026-09-26: 46 of 57 rows kept (dropped value-only repeats -- a zero-width angle range, `-mode 1`/`6`, a wider `-width`, `-constant` on X, a subarea around X, reversed Z around Y -- the short and second byte input, the two-letter axis, and `-adjust` without a tilt file); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "xyzproj";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(PROGRAM)
}

/// The fixture inputs: the regular files of the fixture directory other than
/// the case table and the generator scripts.
fn inputs() -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !path.is_file()
            || name == "cases.tsv"
            || name == "golden.manifest"
            || name.starts_with("make-")
        {
            continue;
        }
        map.insert(name, std::fs::read(&path).unwrap());
    }
    map
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let inputs = inputs();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
        let dir =
            std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (file, bytes) in &inputs {
            std::fs::write(dir.join(file), bytes).unwrap();
        }
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let stdin = if stdin == "-" {
            String::new()
        } else {
            stdin.replace("\\n", "\n")
        };
        let mut child = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(stdin.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let prefix = format!("{name}.out.");
        let mut expected = BTreeMap::new();
        for file in common::golden::list(&golden) {
            if let Some(out) = file.strip_prefix(&prefix) {
                expected.insert(out.to_owned(), common::golden::expect(&golden.join(&file)));
            }
        }
        let mut written = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).unwrap();
            if inputs.get(&file) != Some(&bytes) {
                written.insert(file, bytes);
            }
        }
        if expected.keys().ne(written.keys()) {
            failures.push(format!(
                "{name}: output files {:?}, native {:?}",
                written.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            ));
        }
        for (file, native) in &expected {
            if let Some(ours) = written.get(file) {
                if let Err(why) = native.compare(ours, common::mask_stamps, false) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 20, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `xyzproj` with `args` in a fresh directory holding the fixture inputs
/// plus `extra` files, and returns the output volume `o.mrc` as
/// `[z][y][x]` floats.
fn run_float_volume(tag: &str, args: &[&str], extra: &[(&str, &str)]) -> Vec<Vec<Vec<f32>>> {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-{PROGRAM}-fixed-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in &inputs() {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    for (file, text) in extra {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-input"])
        .args(args)
        .args(["-output", "o.mrc"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{tag}: {output:?}");
    let bytes = std::fs::read(dir.join("o.mrc")).unwrap();
    let word = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let (nx, ny, nz, mode, next) = (
        word(0) as usize,
        word(1) as usize,
        word(2) as usize,
        word(3),
        word(23) as usize,
    );
    assert_eq!(mode, 2, "{tag}: float output expected");
    let data = &bytes[1024 + next..];
    let _ = std::fs::remove_dir_all(&dir);
    (0..nz)
        .map(|z| {
            (0..ny)
                .map(|y| {
                    (0..nx)
                        .map(|x| {
                            let i = 4 * ((z * ny + y) * nx + x);
                            f32::from_le_bytes(data[i..i + 4].try_into().unwrap())
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// `xyzproj.f90:403` foreshortens the section just loaded with
/// `tiltAngles(load0 + 1)` after `load0` has moved past it, i.e. with the next
/// view's angle.  Defined behaviour: each section uses its own angle.  So
/// changing only the last section's angle (keeping the largest |angle|, which
/// sizes the projection boxes) changes only the last output row of every
/// view; natively it changes the row before it instead.
#[test]
fn fixed_series_view_uses_its_own_tilt_angle() {
    let tlt = std::fs::read_to_string(fixture_dir().join("ts.tlt")).unwrap();
    let mut lines: Vec<&str> = tlt.lines().collect();
    let last = lines.len() - 1;
    lines[last] = "   0.00";
    let changed = lines.join("\n") + "\n";
    let args = [
        "ts.mrc",
        "-series",
        "-tiltfile",
        "t.tlt",
        "-angles",
        "0,90,45",
    ];
    let a = run_float_volume("own-angle-a", &args, &[("t.tlt", &tlt)]);
    let b = run_float_volume("own-angle-b", &args, &[("t.tlt", &changed)]);
    assert_eq!(a.len(), 3);
    for (va, vb) in a.iter().zip(&b) {
        let ny = va.len();
        assert_eq!(ny, 9);
        for y in 0..ny - 1 {
            assert_eq!(va[y], vb[y], "row {y} must not depend on the last angle");
        }
        assert_ne!(va[ny - 1], vb[ny - 1], "last row must use the last angle");
    }
}

/// `commonLineRays` (`xyzproj.f90:854`) clears ray counts from the last ray it
/// set, so natively the last column of every tilt-series projection row is the
/// fill value.  Defined behaviour: every ray of the box is kept.  With `-full`
/// and one view the box spans the whole output, so no pixel is fill.
#[test]
fn fixed_series_keeps_last_ray() {
    let args = [
        "ts.mrc",
        "-series",
        "-tiltfile",
        "ts.tlt",
        "-full",
        "-angles",
        "10,10,0",
        "-fill",
        "12345",
    ];
    let v = run_float_volume("last-ray", &args, &[]);
    for row in &v[0] {
        assert_eq!(row.len(), 20);
        assert!(row.iter().all(|&p| p != 12345.), "fill value in {row:?}");
    }
}

/// `useIntersectionAreas` (`xyzproj.f90:600`) limits each slice line with
/// `min(nxOut, ixEnd)`.  Natively an output wider than the slice (22 pixels)
/// reads past each slice line into the next one, so the right-hand padding of
/// `-width 30` carries data, and a narrower output (`-width 12`) drops the
/// pixels past column 12 of the slice.  Defined behaviour: the limit is the
/// slice width.
#[test]
fn fixed_ray_limits_slice_line_to_slice_width() {
    let base = ["v.mrc", "-axis", "Z", "-angles", "0,0,1", "-ray", "-width"];
    let wide = run_float_volume("ray30", &[&base[..], &["30"]].concat(), &[]);
    let full = run_float_volume("ray22", &[&base[..], &["22"]].concat(), &[]);
    let narrow = run_float_volume("ray12", &[&base[..], &["12"]].concat(), &[]);
    for y in 0..wide[0].len() {
        let (w, f, n) = (&wide[0][y], &full[0][y], &narrow[0][y]);
        // right-hand padding is the fill value, as the left-hand padding is
        for x in 26..30 {
            assert_eq!(w[x], w[29]);
            assert!((w[x] - w[0]).abs() < 0.01, "row {y} col {x}: {}", w[x]);
        }
        // the data columns are the full-width projection
        for x in 0..22 {
            assert!((w[x + 4] - f[x]).abs() < 1e-3, "row {y} col {x}");
        }
        // the narrow output is the centre of the full-width one
        for x in 0..12 {
            assert!((n[x] - f[x + 5]).abs() < 0.05, "row {y} col {x}");
        }
    }
}
