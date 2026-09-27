//! Native-golden coverage for `pickbestseed` (`IMOD/imodutil/pickbestseed.cpp`).
//!
//! Every case in `fixtures/pickbestseed/cases.tsv` was run through the native
//! reference `pickbestseed` by `fixtures/make-pickbestseed-goldens.sh`, with
//! its input on stdin (`base.in` with the row's lines dropped and added) and
//! stdout captured through a pipe.  The exit status is `golden/<case>.rc`,
//! stdout `golden/<case>.out`, and every file the run wrote is in
//! `golden/<case>/`.  The inputs are the tracked models, elongation and
//! surface files of a native `autofidseed` run on the TS_01 prealigned stack.
//!
//! Upstream defects are fixed in the translation (`BUGS.md`, pickbestseed,
//! 2026-09-27), so a case whose output a fix changes has its expectation in
//! `defined/` instead (same layout, written from the fixed translation by
//! `fixtures/make-pickbestseed-goldens.sh defined`, and checked when written
//! against the reference source compiled with the same fixes); `golden/`
//! keeps the native record for every case.  Representative rows only; the
//! `#full` rows add the default weights and two more input-error exits.

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/pickbestseed")
}

/// A fresh copy of the inputs, as the make script lays them out: seedin.mod
/// as seed.mod (read by `-AppendToSeedModel`), and badco.txt, e0.txt with a
/// line naming contour 999.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-pickbestseed-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_str().unwrap().to_string();
        if path.is_file()
            && (file.ends_with(".mod") || file.ends_with(".txt"))
            && file != "seedin.mod"
        {
            std::fs::copy(&path, dir.join(&file)).unwrap();
        }
    }
    std::fs::copy(fixture_dir().join("seedin.mod"), dir.join("seed.mod")).unwrap();
    let mut bad = std::fs::read(fixture_dir().join("e0.txt")).unwrap();
    bad.extend_from_slice(
        b"  1    999     0.5000    0.5000    0.5000     0.2000     0.3000     0.3000     0.3000       0.00\n",
    );
    std::fs::write(dir.join("badco.txt"), bad).unwrap();
    dir
}

/// The case's standard input: `base.in` without the lines starting with any
/// comma-separated prefix in `drop`, then the `;`-separated lines of `add`.
fn case_input(drop: &str, add: &str) -> String {
    let base = std::fs::read_to_string(fixture_dir().join("base.in")).unwrap();
    let drops: Vec<&str> = if drop == "-" {
        Vec::new()
    } else {
        drop.split(',').collect()
    };
    let mut text = String::new();
    for line in base.lines() {
        if drops.iter().any(|d| line.starts_with(d)) {
            continue;
        }
        text.push_str(line);
        text.push('\n');
    }
    if add != "-" {
        for line in add.split(';') {
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

fn run_ours(dir: &Path, input: &str) -> std::process::Output {
    let mut child = common::imod_cmd("pickbestseed")
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .arg("-StandardInput")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, drop, add) = (fields[0], fields[1], fields[2]);
        let defined = fixture_dir().join("defined");
        let golden = if common::golden::exists(&defined.join(format!("{name}.rc"))) {
            defined
        } else {
            fixture_dir().join("golden")
        };
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
        let output = run_ours(&dir, &case_input(drop, add));
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
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| !inputs.contains(f))
            .collect();
        // seed.mod is an input the run replaces, renaming it to seed.mod~.
        if produced.iter().any(|f| f == "seed.mod~") {
            produced.push("seed.mod".to_string());
        }
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
            if let Err(why) = theirs.compare(&ours, common::golden::identity, true) {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 22, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The direct-call API autofidseed uses: the `Final:` counts are recorded as
/// printed (checked against the `Final:` line of the `base` expectation).
/// Run in process on this test's thread: `pickbestseed` ends in `exit`, which
/// the runner turns back into a status.
#[test]
fn final_counts_are_recorded() {
    use imod_rs::imod::imodutil::pickbestseed::{PickbestseedResult, pickbestseed_recording};
    use std::sync::{Arc, Mutex};
    let dir = scratch("recording");
    let input = case_input("-", "-");
    let sink = Arc::new(Mutex::new(PickbestseedResult::default()));
    let recorder = Arc::clone(&sink);
    let cwd = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let result = imod_rs::imod::commands::call_in_process(
        &["pickbestseed", "-StandardInput"],
        Some(input.as_bytes()),
        true,
        move || pickbestseed_recording(recorder),
    );
    std::env::set_current_dir(cwd).unwrap();
    let (status, _, _) = result.unwrap();
    assert_eq!(status, 0);
    let defined = fixture_dir().join("defined/base.out");
    let golden = if common::golden::exists(&defined) {
        defined
    } else {
        fixture_dir().join("golden/base.out")
    };
    let text = common::golden::read_to_string(&golden);
    let last = text.lines().find(|l| l.starts_with("Final:")).unwrap();
    let recorded = sink.lock().unwrap().clone();
    let total = recorded.final_total.unwrap();
    let (bottom, top) = recorded.on_bottom_top.unwrap();
    assert_eq!(
        last,
        format!(
            "Final:   total points accepted = {total}  -  on bottom = {bottom} , on top = {top}   [PBS2]"
        )
    );
    let _ = std::fs::remove_dir_all(&dir);
}
