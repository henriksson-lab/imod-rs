//! Native-golden coverage for `mtffilter` (`IMOD/mrc/mtffilter.cpp`).
//!
//! Every case in `fixtures/mtffilter/cases.tsv` was run through the native
//! reference `mtffilter` by `fixtures/make-mtffilter-goldens.sh` (a link to
//! `make-ctfphaseflip-goldens.sh`), stdout captured through a pipe and each
//! step's stdout followed by `rc=<status>`.  The last step's exit status is
//! `golden/<case>.rc`, stdout is `golden/<case>.out`, and every file native
//! left behind is in `golden/<case>/` -- for the in-place case that is the
//! rewritten input itself, compared below by name.  The inputs are seeded
//! synthetic stacks written by the native `raw2mrc`, a 3-D FFT from the
//! native `clip fft -3d`, an MTF curve, dose files of each plain type and an
//! mdoc (`fixtures/make-ctfphaseflip-mtffilter-inputs.py`).

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "mtffilter";

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
        if path.is_file() && path.file_name().unwrap() != "cases.tsv" {
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
        let (name, args) = line.split_once('\t').unwrap();
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let expected_out = std::fs::read(golden.join(format!("{name}.out"))).unwrap();
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
        let (ours, theirs) = (without_banner(&stdout), without_banner(&expected_out));
        if ours != theirs {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&theirs),
                String::from_utf8_lossy(&ours)
            ));
        }
        let mut expected_files: Vec<String> = std::fs::read_dir(golden.join(name))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        expected_files.sort();
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
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if mask(&ours) != mask(&common::reconcile_uninitialised(&ours, &theirs)) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 26, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// BUGS.md: native's fallback option table is one string holding all 42
/// options, so without `mtffilter.adoc` it read past the array and
/// segfaulted.  Defined behaviour: the fallback table has the 42 options as
/// entries, and a run without the autodoc writes the same output as the
/// native golden with it (`lowpass`).
#[test]
fn runs_without_autodoc_from_fallback_table() {
    let dir = scratch("noadoc");
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env_remove("AUTODOC_DIR")
        .env_remove("IMOD_DIR")
        .args([
            "f48.mrc",
            "out.mrc",
            "-lowpass",
            "0.2,0.05",
            "-highpass",
            "0.02",
        ])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let ours = std::fs::read(dir.join("out.mrc")).unwrap();
    let native = std::fs::read(fixture_dir().join("golden/lowpass/out.mrc")).unwrap();
    assert!(mask(&ours) == mask(&common::reconcile_uninitialised(&ours, &native)));
    let _ = std::fs::remove_dir_all(&dir);
}
