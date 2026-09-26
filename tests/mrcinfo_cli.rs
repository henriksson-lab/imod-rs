//! Native-golden coverage for `mrcinfo` (`IMOD/mrc/mrcinfo.c`).
//!
//! Every case in `fixtures/mrcinfo/cases.tsv` was run through a native `mrcinfo` compiled from the vendored source (upstream does
//! not build it) by
//! `fixtures/make-mrcinfo-goldens.sh`, in a directory holding every input from
//! `fixtures/mrcx/` and an empty directory `sub`, stdout captured through a
//! pipe.  Exit status (`golden/<case>.rc`), stdout (`golden/<case>.out`) and
//! every file native created or changed (`golden/<case>/`) are compared byte
//! for byte.  Diagnostics are compared by line count only -- whether a
//! message happens is behaviour, its wording is not (`CLAUDE.md`); the
//! `argv[0]` and `errno` text in them differ by construction.
//! Pruned 2026-09-26: 16 of 35 rows kept (dropped: the big-endian twin of every layout but short and old-style, the trailing-byte/truncated/odd-extended-header files (mrcinfo never checks the body), and the directory/two-file cases that repeat an existing path); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/mrcinfo")
}

fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/mrcx")
}

fn is_input(path: &Path) -> bool {
    path.is_file()
        && path.file_name().unwrap() != "golden.manifest"
        && !path.extension().is_some_and(|e| e == "py" || e == "tsv")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-mrcinfo-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    for entry in std::fs::read_dir(input_dir()).unwrap() {
        let path = entry.unwrap().path();
        if is_input(&path) {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let mut fields = line.split('\t');
        let name = fields.next().unwrap();
        let args = fields.next().unwrap_or(".");
        let args: Vec<&str> = if args == "." {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        let expected_err = common::golden::expect(&golden.join(format!("{name}.err")));
        let dir = scratch(name);
        let output = common::imod_cmd("mrcinfo")
            .current_dir(&dir)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        if !expected_out.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let lines = |b: &[u8]| b.iter().filter(|&&c| c == b'\n').count();
        if !expected_err.matches_masked(&output.stderr, |b| lines(b).to_string().into_bytes()) {
            failures.push(format!(
                "{name}: stderr differs\n--- native\n{}\n--- ours\n{}",
                expected_err.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .filter(|f| {
                let input = input_dir().join(f);
                !(input.is_file()
                    && std::fs::read(&input).unwrap() == std::fs::read(dir.join(f)).unwrap())
            })
            .collect();
        produced.sort();
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, native {expected_files:?}"
            ));
        }
        for file in &expected_files {
            let Ok(ours) = std::fs::read(dir.join(file)) else {
                continue;
            };
            let theirs = common::golden::expect(&golden.join(name).join(file));
            if let Err(why) = theirs.compare(&ours, common::golden::identity, false) {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 16, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
