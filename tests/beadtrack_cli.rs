//! Native-golden coverage for `beadtrack` (`IMOD/flib/beadtrack/beadtrack.cpp`).
//!
//! Every case in `fixtures/beadtrack/cases/<case>.in` is a PIP input that was
//! run as `beadtrack -StandardInput` through the native reference build by
//! `fixtures/make-beadtrack-goldens.sh`, stdout captured through a pipe.  The
//! exit status is `golden/<case>.rc`, stdout is `golden/<case>.out`, and every
//! file native left behind (the output model, snapshot models, elongation and
//! XYZ files, box stacks and their piece lists) is in `golden/<case>/`.  The
//! input is a seeded synthetic tilt series with gold beads; see the script for
//! how it was made.  The cases cover Sobel centering (fixed and scalable
//! sigma), local areas, multiple objects, skipped views, three rounds with
//! trial positions, indexed parameters with near-zero shifts, snapshots and
//! save-all objects, box output, the trace output, and the error paths
//! (including the degenerate-fit NaN position that makes native fail reading
//! the image, reached by `base` and `elong`).
//!
//! **Defined, not native, goldens** (`BUGS.md` "`beadtrack` `beadtrack.cpp`",
//! fixed in translation 2026-09-26): `base` and `elong` (a NaN position is in
//! no piece, so the run completes where native fails reading the image, exit
//! 1 -> 0), `err-noseed` (the seed-model message names the file), and every
//! Sobel-centering case -- `boxout`, `indexed`, `local`, `objs`, `rounds3`,
//! `skip`, `snap`, `sobel`, `sobelelong`, `trace` -- where the Sobel peaks are
//! scaled and then offset (`peak * scale + offset`) instead of multiplied by
//! `scale + offset`, and the Sobel/centroid residual report prints both
//! values.  They are listed in `fixtures/beadtrack/defined.list` and
//! `fixtures/regen-golden.sh beadtrack` records them from our own build
//! (`make-beadtrack-goldens.sh defined`); a build with only the Sobel fix
//! reverted reproduces native on every Sobel case.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/beadtrack")
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-beadtrack-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.starts_with("t1.") {
                std::fs::copy(&path, dir.join(&file)).unwrap();
            }
        }
    }
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let golden = fixture_dir().join("golden");
    let mut cases: Vec<String> = std::fs::read_dir(fixture_dir().join("cases"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
        .filter(|f| f.ends_with(".in"))
        .map(|f| f.trim_end_matches(".in").to_string())
        .collect();
    cases.sort();
    assert!(cases.len() >= 18, "missing beadtrack cases");
    let mut failures = Vec::new();
    for name in &cases {
        let input = std::fs::read(fixture_dir().join("cases").join(format!("{name}.in"))).unwrap();
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
        let mut child = common::imod_cmd("beadtrack")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .arg("-StandardInput")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
        }
        let output = child.wait_with_output().unwrap();
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
    assert!(
        failures.is_empty(),
        "{} of {} beadtrack cases differ from native:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
