//! Native-golden coverage for `tilt` (`IMOD/flib/tilt/tilt.cpp`).
//!
//! Every case in `fixtures/tilt/cases.tsv` was run through the native
//! reference `tilt` by `fixtures/make-tilt-goldens.sh`, stdout captured
//! through a pipe.  The exit status is `golden/<case>.rc`, stdout is
//! `golden/<case>.out`, and every file native left behind is in
//! `golden/<case>/`.  The inputs are a seeded synthetic aligned tilt series
//! written by the native `raw2mrc`, matching tilt/X-tilt/Z-factor/local
//! alignment files, a reconstruction of it made by the native `tilt`, and a
//! scattered-point model made by the native `point2model`.  An argument column
//! `STDIN:<file>` runs `tilt -StandardInput` with the file on standard input,
//! the way `tilt.com` drives it.
//!
//! Pruned 2026-09-26: 22 of 23 rows kept (dropped `basic`, the base argument set every other run extends); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tilt")
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-tilt-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.starts_with("t.") || file == "pm.mod" || file == "std.com" {
                std::fs::copy(&path, dir.join(&file)).unwrap();
            }
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
        let (name, args) = line.split_once('\t').unwrap();
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        let dir = scratch(name);
        let inputs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        let mut command = common::imod_cmd("tilt");
        command
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .env("IMOD_NO_IMAGE_BACKUP", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let stdin_file = args.strip_prefix("STDIN:");
        if stdin_file.is_some() {
            command.arg("-StandardInput").stdin(Stdio::piped());
        } else {
            command.args(args.split_whitespace()).stdin(Stdio::null());
        }
        let mut child = command.spawn().unwrap();
        if let Some(file) = stdin_file {
            let text = std::fs::read(dir.join(file)).unwrap();
            child.stdin.take().unwrap().write_all(&text).unwrap();
        }
        let output = child.wait_with_output().unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // The header listings carry the labels' time stamps.
        let ours = common::mask_stamps(&output.stdout);
        if !expected_out.matches_masked(&output.stdout, common::mask_stamps) {
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
            .filter(|f| !inputs.contains(f))
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
    assert!(count >= 22, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
