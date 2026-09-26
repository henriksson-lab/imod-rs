//! Native-golden coverage for `ccderaser` (`IMOD/flib/model/ccderaser.f90`).
//!
//! Every row of `fixtures/ccderaser/cases.tsv` was run through the native
//! reference program by `fixtures/make-ccderaser-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output, and `.o.mrc`, `.p.mod` and `.inplace` the
//! output image, the `-points` model and an input image modified in place,
//! when native wrote one.  Images are compared byte for byte with the
//! label date/time stamps masked (`common::mask_stamps`); point models
//! byte for byte; in the two regions native leaves uninitialised (`Imod.name`
//! past its terminator, and the `MINX` chunk's `oscale`/`orot`, which
//! `putimageref` never sets; `BUGS.md` §2) ours must hold the defined value
//! (`common::reconcile_uninitialised`).
//!
//! Upstream defects are fixed in the translation (`BUGS.md`, 2026-09-26), so
//! a case whose output a fix changes has its expectation in `defined/`
//! instead (same names, written from the fixed translation by
//! `fixtures/make-ccderaser-goldens.sh defined`); `golden/` keeps the native
//! record for every case.  The fixes are asserted directly by the tests
//! after the golden loop.
//! Pruned 2026-09-26: 56 of 68 rows kept (dropped: find_points_trial, find_xyscan_edge, model_multi, model_lines_b, model_trial, noskip (paths other rows reach), and the near-duplicate defined-output rows find_b, find_verbose, find_half_s, find_iter6, model_merge, interactive2 -- the border-ring fix they carry is pinned by the rest); the pruned rows stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "ccderaser";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/ccderaser")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|n| n != "cases.tsv" && n != "golden.manifest")
        {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
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
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
        let defined = fixture_dir().join("defined");
        let golden = if common::golden::exists(&defined.join(format!("{name}.rc"))) {
            &defined
        } else {
            &golden
        };
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
            .env("OMP_NUM_THREADS", "1")
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
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches_masked(&output.stdout, common::mask_stamps) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let mut inplace = None;
        for input in ["f.mrc", "s.mrc", "b.mrc"] {
            let now = std::fs::read(dir.join(input)).unwrap();
            if now != std::fs::read(fixture_dir().join(input)).unwrap() {
                inplace = Some(now);
            }
        }
        let checks: [(&str, Option<Vec<u8>>, fn(&[u8]) -> Vec<u8>); 3] = [
            (
                "o.mrc",
                std::fs::read(dir.join("o.mrc")).ok(),
                common::mask_stamps,
            ),
            (
                "p.mod",
                std::fs::read(dir.join("p.mod")).ok(),
                common::mask_stamps,
            ),
            ("inplace", inplace, common::mask_stamps),
        ];
        for (suffix, written, mask) in checks {
            let expected = common::golden::load(&golden.join(format!("{name}.{suffix}")));
            match (expected, written) {
                (None, None) => {}
                (Some(g), Some(w)) if g.matches_reconciled(&w, mask) => {}
                (g, w) => failures.push(format!(
                    "{name}: {suffix} differs (native {:?}, ours {:?} bytes)",
                    g.map(|b| b.describe()),
                    w.map(|b| b.len())
                )),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 56, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Run the translation in a fresh copy of the fixtures, feeding `stdin`.
fn run(name: &str, args: &[&str], stdin: &str) -> (Option<i32>, String, PathBuf) {
    let dir = scratch(name);
    let mut child = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .args(args)
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
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        dir,
    )
}

/// BUGS.md, ccderaser: `write(*,107) 'is not in one Z-plane'` has no items for
/// the format's integers, so native stops with a gfortran runtime error
/// (status 2).  The translation prints the intended message and exits 1.
#[test]
fn contour_not_in_one_z_plane_is_the_intended_error() {
    let (rc, stdout, dir) = run(
        "fix_zplane",
        &["-input", "s.mrc", "-output", "o.mrc", "-model", "mz.mod"],
        "",
    );
    assert_eq!(rc, Some(1));
    assert!(
        stdout.contains("ERROR: CCDERASER - object   1, contour     1 is not in one Z-plane"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// BUGS.md, ccderaser: `-better` with one value no longer prints the leftover
/// `[CCE1]` debug line.
#[test]
fn single_better_radius_prints_no_debug_line() {
    let (rc, stdout, dir) = run(
        "fix_cce1",
        &[
            "-input", "s.mrc", "-output", "o.mrc", "-model", "mb.mod", "-circle", "/", "-better",
            "3.5",
        ],
        "",
    );
    assert_eq!(rc, Some(0));
    assert!(!stdout.contains("[CCE1]"), "{stdout}");
    let _ = std::fs::remove_dir_all(dir);
}

/// BUGS.md, ccderaser: interactive input leaves the circle and boundary lists
/// empty (as PIP input does when they are not entered) instead of "every
/// object", so an interactive run erases instead of stopping with "Object 1
/// is included in more than one list".
#[test]
fn interactive_input_erases() {
    let (rc, stdout, dir) = run(
        "fix_interactive",
        &[],
        "s.mrc\no.mrc\nm1.mod\n6\n2\n/\n/\n/\n",
    );
    assert_eq!(rc, Some(0), "{stdout}");
    assert!(!stdout.contains("more than one list"), "{stdout}");
    assert!(stdout.contains("Section    1 - fixing points"), "{stdout}");
    let input = std::fs::read(dir.join("s.mrc")).unwrap();
    let output = std::fs::read(dir.join("o.mrc")).unwrap();
    assert_eq!(input.len(), output.len());
    assert_ne!(input[1024..], output[1024..], "nothing was erased");
    let _ = std::fs::remove_dir_all(dir);
}

/// BUGS.md, ccderaser: `ExcludeAdjacent` is a boolean whose value counts.
/// Natively any entry, `ExcludeAdjacent 0` included, excludes; here 0 is the
/// same as not entering it and 1 is the same as `-exclude`.
#[test]
fn exclude_adjacent_zero_does_not_exclude() {
    let base = "InputFile s.mrc\nOutputFile o.mrc\nFindPeaks\nBorderSize 3\n";
    let image = |name: &str, extra: &str| {
        let (rc, stdout, dir) = run(name, &["-StandardInput"], &format!("{base}{extra}"));
        assert_eq!(rc, Some(0), "{stdout}");
        let bytes = common::mask_stamps(&std::fs::read(dir.join("o.mrc")).unwrap());
        let _ = std::fs::remove_dir_all(dir);
        bytes
    };
    let absent = image("fix_excl_absent", "");
    let zero = image("fix_excl_zero", "ExcludeAdjacent 0\n");
    let one = image("fix_excl_one", "ExcludeAdjacent 1\n");
    assert_eq!(zero, absent);
    assert_ne!(one, absent);
}
