//! Native-golden coverage for `patch2imod` (`IMOD/imodutil/patch2imod.c`).
//!
//! Every row of `fixtures/patch2imod/cases.tsv` was run through the native
//! reference program by `fixtures/make-patch2imod-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.mod` the output model, when native
//! wrote one.  Models are compared byte for byte; `Imod.name` past its
//! terminator is heap residue in native and must be our defined zeros there
//! (`BUGS.md` §2, `common::reconcile_uninitialised`); standard output with the `imodVersion` banner's compile date
//! and time masked (compile metadata).

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "patch2imod";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/patch2imod")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() && path.file_name().is_some_and(|n| n != "cases.tsv") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    dir
}

/// Drops the date and time after `Version 5.2.17` in the usage banner.
fn mask_banner(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    text.lines()
        .map(|line| match line.find(" Version ") {
            Some(k) => line[..k + 9].to_string(),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args) = (fields[0], fields[1]);
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let output = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .args(&args)
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
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
        let stdout = std::fs::read(golden.join(format!("{name}.stdout"))).unwrap();
        if mask_banner(&output.stdout) != mask_banner(&stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected = std::fs::read(golden.join(format!("{name}.mod"))).ok();
        let written = std::fs::read(dir.join("o.mod")).ok();
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if common::reconcile_uninitialised(&w, &g) == w => {}
            (g, w) => failures.push(format!(
                "{name}: o.mod differs (native {:?} bytes, ours {:?} bytes)",
                g.map(|b| b.len()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 34, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs our program on `args` in a fresh fixture copy and returns the exit
/// status, stdout and the model written.
fn run_ours(name: &str, args: &[&str]) -> (Option<i32>, Vec<u8>, Option<Vec<u8>>) {
    let dir = scratch(name);
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .args(args)
        .output()
        .unwrap();
    let model = std::fs::read(dir.join("o.mod")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    (output.status.code(), output.stdout, model)
}

/// BUGS.md: with `-l` native read `valTypeMap`/`orderedIDs` uninitialised.
/// Defined behaviour: the maps are built as for a file whose first line has
/// zero value IDs, so `-l` on `lvals.out` gives the model native writes for
/// the same lines under the count line `12 0 0` (goldens `lines_hdr`,
/// `lines_hdr_col`).
#[test]
fn count_lines_maps_value_columns_like_a_first_line() {
    let golden = fixture_dir().join("golden");
    for (name, args) in [
        ("lines_hdr", vec!["-l", "lvals.out", "o.mod"]),
        (
            "lines_hdr_col",
            vec!["-l", "-v", "-2", "lvals.out", "o.mod"],
        ),
    ] {
        let (rc, _, model) = run_ours(name, &args);
        assert_eq!(rc, Some(0), "{name}");
        let native = std::fs::read(golden.join(format!("{name}.mod"))).unwrap();
        let model = model.unwrap();
        assert!(
            common::reconcile_uninitialised(&model, &native) == model,
            "{name}: model differs"
        );
    }
}

/// BUGS.md: native rejected a patch file whose complete last line lacks a
/// newline.  Defined behaviour: that line is read, giving the model native
/// writes for the same file with the newline (golden `nolastnl_nl`).
#[test]
fn last_line_without_newline_is_read() {
    let (rc, _, model) = run_ours("nolastnl", &["nolastnl.out", "o.mod"]);
    assert_eq!(rc, Some(0));
    let native = std::fs::read(fixture_dir().join("golden/nolastnl_nl.mod")).unwrap();
    let model = model.unwrap();
    assert!(common::reconcile_uninitialised(&model, &native) == model);
}

/// BUGS.md: an option needing a value given last dereferenced `argv[argc]`
/// natively.  Defined behaviour: the run ends in the program's own
/// "Wrong # of arguments" exit.
#[test]
fn option_value_missing_is_a_clean_error() {
    for opt in ["-s", "-n", "-c"] {
        let (rc, stdout, model) = run_ours("missing", &["-f", opt]);
        assert_eq!(rc, Some(1), "{opt}");
        assert!(
            String::from_utf8_lossy(&stdout).contains("Wrong # of arguments"),
            "{opt}"
        );
        assert!(model.is_none());
    }
}
