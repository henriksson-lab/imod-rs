//! Native-golden coverage for `findwarp` (`IMOD/flib/model/findwarp.f90`).
//!
//! Every row of `fixtures/findwarp/cases.tsv` was run through the native
//! reference program by `fixtures/make-findwarp-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created
//! or changed.
//! Inputs: synthetic corrsearch3d-style patch files written by
//! `fixtures/findwarp/make-patches.py` (affine and smoothly warped
//! displacement fields with noise, outliers, missing patches, one- and
//! two-layer grids, extra-value columns), region models written by the native
//! point2model, an initial 3D transform, and warp files the native findwarp
//! wrote (new style, old-style header, incomplete).  The cases cover every
//! option of `findwarp.adoc`, the interactive dialogue and the error paths.
//! Exit status, standard output and every output file must match byte for
//! byte.
//! Pruned 2026-09-26: 47 of 63 rows kept (dropped: value variants of -rowcol/-slabs/-extra/-legacy/-desired/-region and interactive slabs/single dialogues, and f_twolayer, whose p4.patch is absent so it only repeated f_nofile); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "findwarp";

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
    assert!(count >= 47, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs the Rust program on copies of the fixture inputs for a case that
/// reaches an upstream bug fixed in translation (BUGS.md), with a deadline
/// (a regression could hang).  Returns the exit status, stdout and the
/// scratch directory.
fn run_fixed(name: &str, args: &[&str]) -> (Option<i32>, String, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-{PROGRAM}-fixed-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in &inputs() {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    let mut child = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > std::time::Duration::from_secs(60) {
            let _ = child.kill();
            panic!("{name}: still running after 60 s");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        dir,
    )
}

/// `findwarp.f90:295-305` edits the INTEGER patch spacing with `f7.1`, so
/// native stops with a gfortran runtime error (status 2).  Fixed in
/// translation: the intended message, status 1.
#[test]
fn spacing_mismatch_reports_the_intended_error() {
    let (rc, stdout, dir) = run_fixed(
        "irreg",
        &[
            "-patch",
            "p12.patch",
            "-output",
            "w.xf",
            "-volume",
            "512,128,512",
            "-rowcol",
            "4,4",
            "-initial",
            "w12.xf",
        ],
    );
    assert_eq!(rc, Some(1), "{stdout}");
    assert!(
        stdout.contains(
            "ERROR: FINDWARP - The transform spacing in X in the the read-in file (   55.5) \
         does not match the patch spacing (   56.0)"
        ),
        "{stdout}"
    );
    assert!(!dir.join("w.xf").exists());
    let (rc, stdout, _) = run_fixed(
        "other",
        &[
            "-patch",
            "p1.patch",
            "-output",
            "w.xf",
            "-volume",
            "512,128,512",
            "-rowcol",
            "4,4",
            "-initial",
            "w2.xf",
        ],
    );
    assert_eq!(rc, Some(1), "{stdout}");
    assert!(
        stdout.contains(
            "The transform spacing in Y in the the read-in file (   55.0) \
         does not match the patch spacing (   48.0)"
        ),
        "{stdout}"
    );
}

/// An illegal `-rowcol` in a PIP run loops on "Illegal entry, try again"
/// natively (`findwarp.f90:640-644`).  Fixed in translation: status 1.
#[test]
fn illegal_rowcol_in_pip_run_exits() {
    for (name, rowcol) in [("rowcol14", "1,4"), ("rowcol94", "9,4")] {
        let (rc, stdout, _) = run_fixed(
            name,
            &[
                "-patch",
                "p2.patch",
                "-output",
                "w.xf",
                "-volume",
                "512,512,100",
                "-rowcol",
                rowcol,
            ],
        );
        assert_eq!(rc, Some(1), "{name}: {stdout}");
        assert!(
            stdout.contains("ERROR: FINDWARP - Improper number to include in fit"),
            "{stdout}"
        );
        assert!(!stdout.contains("Illegal entry, try again"), "{stdout}");
    }
}

/// `get_region_contours.f90:249` tests `icolSelect(i) > maxExtra` for a
/// negative entry, which can never be true, so native indexes past the extra
/// columns.  Fixed in translation: a column past them is rejected.
#[test]
fn negative_extra_column_past_the_extras_is_rejected() {
    let (rc, stdout, _) = run_fixed(
        "extraneg",
        &[
            "-patch",
            "p1.patch",
            "-output",
            "w.xf",
            "-volume",
            "512,128,512",
            "-target",
            "0.4",
            "-extra",
            "-9,1",
            "-select",
            "0.3",
        ],
    );
    assert_eq!(rc, Some(1), "{stdout}");
    assert!(
        stdout.contains(
            "There is no extra value column corresponding to the column number entered with -extra"
        ),
        "{stdout}"
    );
}
