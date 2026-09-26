//! Native-golden coverage for `findcontrast` (`IMOD/flib/image/findcontrast.f90`).
//!
//! Every case in `fixtures/findcontrast/cases.tsv` was run through the native
//! reference `findcontrast` by `fixtures/make-densmatch-findcontrast-goldens.sh`,
//! stdout captured through a pipe.  The exit status is `golden/<case>.rc`,
//! stdout is `golden/<case>.out`, and every file native created or changed
//! is in `golden/<case>/`.  A third column in the table is fed on stdin.
//! The inputs are seeded synthetic volumes written by the native `raw2mrc`.

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/findcontrast")
}

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
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "mrc") {
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
        let stdin = fields.next().unwrap_or("").replace("\\n", "\n");
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
        let (ours, theirs) = (skip_banner(&output.stdout), skip_banner(&expected_out));
        if ours != theirs {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&theirs),
                String::from_utf8_lossy(&ours)
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
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
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
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if mask(&ours) != mask(&common::reconcile_uninitialised(&ours, &theirs)) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count > 10, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
