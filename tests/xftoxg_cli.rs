//! Native-golden coverage for `xftoxg` (`IMOD/flib/image/xftoxg.f90`).
//!
//! Every `xftoxg` row of `fixtures/xftransforms/cases.tsv` was run through the
//! native reference program by `fixtures/make-xftransforms-goldens.sh`, with
//! standard output captured through a pipe.  `golden/xftoxg-<case>.rc` is the
//! exit status, `.stdout` the standard output (the transform counts, fit
//! coefficients and file-open lines are all byte-identical, so the whole
//! stream is compared) and `.out` the transform or warping file native
//! wrote, when it wrote one.  Inputs: the seeded linear, grid-warping and
//! control-point files in `fixtures/xftransforms`, plus three `.xf` files
//! from the vendored `IMOD/Etomo/uitestData`.
//! Pruned 2026-09-26: 44 of 78 xftoxg rows kept (dropped nfit/order/ref/mixed/robust value permutations, extra small-count and warp-fit variants, the large `medium` input and one interactive form); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "xftoxg";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/xftransforms")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "xf") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData");
    for name in [
        "BB/BBa.xf",
        "midzone2/midzone2a.xf",
        "mediumscansb2/mediumscansb2_midas.xf",
    ] {
        let source = vendored.join(name);
        std::fs::copy(&source, dir.join(source.file_name().unwrap())).unwrap();
    }
    dir
}

/// The `printf` escapes the golden script feeds the interactive cases.
fn unescape(text: &str) -> String {
    if text == "-" {
        return String::new();
    }
    text.replace("\\n", "\n")
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (program, name, args, stdin) = (fields[0], fields[1], fields[2], fields[3]);
        if program != PROGRAM {
            continue;
        }
        let stem = format!("{program}-{name}");
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
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
            .write_all(unescape(stdin).as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{stem}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{stem}.stdout")));
        if !stdout.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        // A case whose output a BUGS.md fix changes compares against the
        // defined output in `defined/` instead of native's golden (kept in
        // `golden/` as the record).  `xftoxg-cpmix`: a section with 3 or
        // fewer control points directly after the first one used a zero
        // grid whose start and interval native left at the last grid read
        // in setup, so its cumulative product was stored in that layout and
        // read as the common one; the zero grid now carries the common
        // layout (differences in the third decimal of the output grid).
        let defined = fixture_dir().join("defined").join(format!("{stem}.out"));
        let expected = common::golden::load(&defined)
            .or_else(|| common::golden::load(&golden.join(format!("{stem}.out"))));
        let written = ["o.xg", "BBa.xg", "p.xf"]
            .iter()
            .find_map(|o| std::fs::read(dir.join(o)).ok());
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches(&w) => {}
            (g, w) => failures.push(format!(
                "{name}: output differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 44, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `xftoxg` in `dir` under a 60 s timeout (a regression of the fixed
/// hangs must fail the test, not stall the suite).
fn run_timed(dir: &Path, args: &[&str]) -> std::process::Output {
    let mut command = std::process::Command::new("timeout");
    command
        .arg("60")
        .arg(env!("CARGO_BIN_EXE_imod"))
        .arg(PROGRAM)
        .args(args)
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .stdin(Stdio::null());
    command.output().unwrap()
}

/// BUGS.md `xftoxg` / `xfproduct`, fixed in translation: a warping file
/// whose sections have different grid layouts (`w7var.xf`) made native store
/// each cumulative product in the section's own layout and read it back as
/// the common one -- positions it never wrote, output varying run to run.
/// Defined: products live in the common layout, so the result is the same on
/// every run and holds no NaN.
#[test]
fn varying_grid_layouts_give_a_stable_finite_warping() {
    let dir = scratch("defined_w7var");
    let first = run_timed(&dir, &["-in", "w7var.xf", "-g", "a.xg", "-nfit", "0"]);
    assert_eq!(first.status.code(), Some(0), "{first:?}");
    let second = run_timed(&dir, &["-in", "w7var.xf", "-g", "b.xg", "-nfit", "0"]);
    assert_eq!(second.status.code(), Some(0));
    let a = std::fs::read_to_string(dir.join("a.xg")).unwrap();
    let b = std::fs::read_to_string(dir.join("b.xg")).unwrap();
    assert_eq!(a, b);
    assert!(!a.to_ascii_lowercase().contains("nan"), "{a}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// BUGS.md `xftoxg` / `xfproduct`, fixed in translation: a control-point
/// file where no section has 4 or more points (`cplin.xf`) defines no grid;
/// native derives a zero-extent grid, NaN angles, and loops forever.
/// Defined: an error exit.
#[test]
fn control_points_without_a_grid_stop_with_an_error() {
    let dir = scratch("defined_cplin");
    let output = run_timed(&dir, &["-in", "cplin.xf", "-g", "o.xg"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("enough control points"),
        "{output:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// BUGS.md `xftoxg` / `xfproduct`, fixed in translation: `groupRotations`
/// with an angle that fits no range (here a negative `-range`) never reduces
/// its count of ungrouped angles; native loops forever.  Defined: an error.
#[test]
fn ungroupable_rotation_angles_stop_with_an_error() {
    let dir = scratch("defined_negrange");
    let output = run_timed(&dir, &["-in", "groups.xf", "-g", "o.xg", "-range", "-1"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("cannot be grouped"),
        "{output:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
