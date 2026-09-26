//! Native-golden coverage for `findcontrast` (`IMOD/flib/image/findcontrast.f90`).
//!
//! Every case in `fixtures/findcontrast/cases.tsv` was run through the native
//! reference `findcontrast` by `fixtures/make-densmatch-findcontrast-goldens.sh`,
//! stdout captured through a pipe.  The exit status is `golden/<case>.rc`,
//! stdout is `golden/<case>.out`, and every file native created or changed
//! is in `golden/<case>/`.  A third column in the table is fed on stdin.
//! The inputs are seeded synthetic volumes written by the native `raw2mrc`:
//! `f.mrc`, `s.mrc` and `b.mrc` from `fixtures/densmatch`, which uses the same
//! three (they are not stored twice).
//!
//! Pruned 2026-09-26: 15 of 19 rows kept (dropped: second -slices and -xminmax/-yminmax errors (identical output), -truncate 0,0, and a second interactive run); the rest stay in cases.tsv as `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/findcontrast")
}

/// Where the input volumes live: `fixtures/densmatch`'s `f.mrc`, `s.mrc`, `b.mrc`.
fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/densmatch")
}

const INPUTS: [&str; 3] = ["f.mrc", "s.mrc", "b.mrc"];

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-findcontrast-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for name in INPUTS {
        std::fs::copy(input_dir().join(name), dir.join(name)).unwrap();
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
        let stdin = fields.next().unwrap_or("").replace("\\n", "\n");
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
        let mut child = common::imod_cmd("findcontrast")
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
        let _ = child.stdin.take().unwrap().write_all(stdin.as_bytes());
        let output = child.wait_with_output().unwrap();
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
            if b.starts_with(b"findcontrast Version") {
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
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| {
                let fixture = input_dir().join(f);
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
            if let Err(why) = theirs.compare(&ours, mask, true) {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count > 10, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
