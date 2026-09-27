//! Native-golden coverage for `clipmodel` (`IMOD/flib/model/clipmodel.f90`).
//!
//! Every row of `fixtures/clipmodel/cases.tsv` was run through the native
//! reference program by `fixtures/make-clipmodel-goldens.sh`, with standard
//! output captured through a pipe and standard input empty.
//! `golden/<case>.rc` is the exit status, `.stdout` the standard output,
//! `.mod` the output model and `.txt` the point list, when native wrote one.
//! Models are compared byte for byte, with `Imod.name` past its terminator
//! reconciled as our defined zeros (`BUGS.md` §2); standard output with the
//! usage banner's compile date masked.  The interactive (no-option) form was
//! checked against native by hand; only its end-of-input exit is a row.
//!
//! Pruned: 23 of 25 rows kept (dropped: the nm area range and a second
//! -xminmax count mismatch); the rest stay in cases.tsv as `#full` rows.

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "clipmodel";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(PROGRAM)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|n| n != "cases.tsv" && n != "golden.manifest")
        {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    dir
}

/// Drops the date and time after `Version 5.2.17` in a usage banner.
fn mask_banner(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    text.lines()
        .map(|line| match line.find(" Version ") {
            Some(k) => line[..k + 9].to_string(),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args) = (fields[0], fields[1]);
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let output = common::imod_cmd(PROGRAM)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .current_dir(&dir)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches_masked(&output.stdout, mask_banner) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        for ext in ["mod", "txt"] {
            let expected = common::golden::load(&golden.join(format!("{name}.{ext}")));
            let written = std::fs::read(dir.join(format!("o.{ext}"))).ok();
            match (expected, written) {
                (None, None) => {}
                (Some(g), Some(w)) if g.matches_reconciled(&w, common::golden::identity) => {}
                (g, w) => failures.push(format!(
                    "{name}: o.{ext} differs (native {:?}, ours {:?} bytes)",
                    g.map(|b| b.describe()),
                    w.map(|b| b.len())
                )),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 23, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The direct-call API autofidseed uses: `clipmodel_recording` reports the
/// "Number of points reduced from ... to ..." values it prints.
#[test]
fn recording_reports_points_kept() {
    use imod_rs::imod::flib::model::clipmodel::{ClipmodelResult, clipmodel_recording};
    let dir = scratch("recording");
    let input = dir.join("sc.mod");
    let output = dir.join("o.mod");
    let sink = std::sync::Arc::new(std::sync::Mutex::new(ClipmodelResult::default()));
    let recorder = std::sync::Arc::clone(&sink);
    let words = [
        "clipmodel".to_string(),
        "-keep".to_string(),
        "-zmin".to_string(),
        "2,2".to_string(),
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
    ];
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    let (status, _, _) = imod_rs::imod::commands::call_in_process(&words, None, true, move || {
        clipmodel_recording(recorder)
    })
    .unwrap();
    assert_eq!(status, 0);
    let result = sink.lock().unwrap().clone();
    assert_eq!(result.points_reduced, vec![(40, 8)]);
    assert_eq!(result.contours_reduced, vec![(1, 1)]);
    let _ = std::fs::remove_dir_all(&dir);
}
