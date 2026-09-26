//! Native-golden coverage for `ctfphaseflip` (`IMOD/mrc/ctfphaseflip.cpp`).
//!
//! Every case in `fixtures/ctfphaseflip/cases.tsv` was run through the native
//! reference `ctfphaseflip` by `fixtures/make-ctfphaseflip-goldens.sh`, stdout
//! captured through a pipe.  A case may have several steps separated by
//! ` ;; ` (a parallel-writing setup run and its chunks) run in one directory;
//! each step's stdout is followed by `rc=<status>`.  The last step's exit
//! status is `golden/<case>.rc`, stdout is `golden/<case>.out`, and every file
//! native left behind is in `golden/<case>/`.  The inputs are seeded synthetic
//! stacks written by the native `raw2mrc` and text files in every defocus
//! format `ctfutils.cpp` reads (`fixtures/make-ctfphaseflip-mtffilter-inputs.py`).
//!
//! Pruned 2026-09-26: 19 of 20 rows kept (dropped `strips`, the large short-mode s160 run; integer input and strip handling stay covered by `byte` and the float cases); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "ctfphaseflip";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(PROGRAM)
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

/// Drops the usage banner line, which carries the build date.
fn without_banner(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    text.lines()
        .filter(|line| !line.starts_with(&format!("{PROGRAM} Version")))
        .map(|line| format!("{line}\n"))
        .collect::<String>()
        .into_bytes()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path.file_name().unwrap() != "cases.tsv"
            && path.file_name().unwrap() != "golden.manifest"
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
        let mut stdout = Vec::new();
        let mut status = None;
        for step in args.split(" ;; ") {
            let output = common::imod_cmd(PROGRAM)
                .current_dir(&dir)
                .env(
                    "AUTODOC_DIR",
                    concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
                )
                .env("OMP_NUM_THREADS", "1")
                .env_remove("IMOD_USE_GPU")
                .env_remove("IMOD_OUTPUT_FORMAT")
                .args(step.split_whitespace())
                .output()
                .unwrap();
            stdout.extend_from_slice(&output.stdout);
            stdout.extend_from_slice(
                format!("rc={}\n", output.status.code().unwrap_or(-1)).as_bytes(),
            );
            status = output.status.code();
        }
        count += 1;
        if status != Some(rc) {
            failures.push(format!("{name}: exit {status:?}, native {rc}"));
        }
        let ours = without_banner(&stdout);
        if !expected_out.matches_masked(&stdout, without_banner) {
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
                // New files, and inputs the case rewrote in place.
                !inputs.contains(f)
                    || std::fs::read(dir.join(f)).ok() != std::fs::read(fixture_dir().join(f)).ok()
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
    assert!(count >= 19, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
