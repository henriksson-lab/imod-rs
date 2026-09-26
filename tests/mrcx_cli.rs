//! Native-golden coverage for `mrcx` (`IMOD/mrc/mrcx.c`).
//!
//! Every case in `fixtures/mrcx/cases.tsv` was run through the native reference `mrcx` by
//! `fixtures/make-mrcx-goldens.sh`, in a directory holding every input from
//! `fixtures/mrcx/` and an empty directory `sub`, stdout captured through a
//! pipe.  Exit status (`golden/<case>.rc`), stdout (`golden/<case>.out`) and
//! every file native created or changed (`golden/<case>/`) are compared byte
//! for byte.  Diagnostics are compared by line count only -- whether a
//! message happens is behaviour, its wording is not (`CLAUDE.md`); the
//! `argv[0]` and `errno` text in them differ by construction.
//!
//! The inputs (`fixtures/mrcx/make-inputs.py`) cover modes 0, 1, 2, 3, 4 and
//! 16 in both byte orders -- mode 12, which `mrcx` refuses, too -- new- and
//! old-style headers, an odd-sized extended header, trailing bytes within
//! and beyond the 511-byte allowance, and truncated data; each is converted
//! both to a new file and in place.

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/mrcx")
}

fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/mrcx")
}

fn is_input(path: &Path) -> bool {
    path.is_file() && !path.extension().is_some_and(|e| e == "py" || e == "tsv")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-mrcx-{}-{}", std::process::id(), name));
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
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut fields = line.split('\t');
        let name = fields.next().unwrap();
        let args = fields.next().unwrap_or(".");
        let args: Vec<&str> = if args == "." {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let expected_out = std::fs::read(golden.join(format!("{name}.out"))).unwrap();
        let expected_err = std::fs::read(golden.join(format!("{name}.err"))).unwrap();
        let dir = scratch(name);
        let output = common::imod_cmd("mrcx")
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
        if output.stdout != expected_out {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_out),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let lines = |b: &[u8]| b.iter().filter(|&&c| c == b'\n').count();
        if lines(&output.stderr) != lines(&expected_err) {
            failures.push(format!(
                "{name}: stderr differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_err),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let mut expected_files: Vec<String> = std::fs::read_dir(golden.join(name))
            .map(|entries| {
                entries
                    .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default();
        expected_files.sort();
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
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if ours != theirs {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 60, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
