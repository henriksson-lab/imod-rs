//! Native-golden coverage for `point2model` (`IMOD/imodutil/point2model.c`).
//!
//! Every row of `fixtures/point2model/cases.tsv` was run through the native
//! reference program by `fixtures/make-point2model-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.mod` the output model, when native
//! wrote one.  Models are compared byte for byte; `Imod.name` past its
//! terminator is heap residue in native (`imodNew` mallocs) and must be our
//! defined zeros there (`BUGS.md` §2, `common::reconcile_uninitialised`);
//! standard output with the usage banner's compile date and time masked.
//!
//! Representative rows only: each option once, each error exit once; the
//! `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md) add the
//! plain object/contour file and two more error exits.

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "point2model";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/point2model")
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
    for line in common::golden::case_rows(&table) {
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .output()
            .unwrap();
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
        if !stdout.matches_masked(&output.stdout, mask_banner) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected = common::golden::load(&golden.join(format!("{name}.mod")));
        let written = std::fs::read(dir.join("o.mod")).ok();
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches_reconciled(&w, common::golden::identity) => {}
            (g, w) => failures.push(format!(
                "{name}: o.mod differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 25, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
