//! Native-golden coverage for `modifymdoc` (`IMOD/mrc/modifymdoc.cpp`).
//!
//! Every case in `fixtures/modifymdoc/cases.tsv` was run through the native
//! reference `modifymdoc` by `fixtures/make-modifymdoc-goldens.sh`, stdout
//! captured through a pipe and `TZ=UTC` (the dose ordering goes through
//! `mktime`).  The exit status is `golden/<case>.rc`, stdout is
//! `golden/<case>.out`, and every file native created or changed is in
//! `golden/<case>/`.  The inputs are hand-authored SerialEM-style `.mdoc`
//! files; each case's directory also holds an empty directory `sub`.
//! Pruned 2026-09-26: 43 of 70 rows kept (dropped: value permutations of -order/-binning/-pixel that reach the same branch, a second case per exitError message (doseneg, bin0/052/neg, pixneg, nooutputopt, dirinput, frameset, noimagefile), the reversed-order twins of the NaN/scrambled/tie sorts, and the usage-only plain/twoopt); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/modifymdoc")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-modifymdoc-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|e| e == "mdoc" || e == "param")
        {
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
        let dir = scratch(name);
        let output = common::imod_cmd("modifymdoc")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("TZ", "UTC")
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
        // The usage listing starts with a version banner carrying the build
        // date (`imodVersion`); only the lines after it are compared.
        let skip_banner = |b: &[u8]| -> Vec<u8> {
            if b.starts_with(b"modifymdoc Version") {
                b.iter()
                    .position(|&c| c == b'\n')
                    .map_or(Vec::new(), |p| b[p + 1..].to_vec())
            } else {
                b.to_vec()
            }
        };
        let ours = skip_banner(&output.stdout);
        if !expected_out.matches_masked(&output.stdout, skip_banner) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&ours)
            ));
        }
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .filter(|f| {
                let fixture = fixture_dir().join(f);
                !(fixture.is_file()
                    && std::fs::read(&fixture).unwrap() == std::fs::read(dir.join(f)).unwrap())
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
    assert!(count >= 43, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
