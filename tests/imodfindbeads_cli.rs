//! Native-golden coverage for `imodfindbeads` (`IMOD/imodutil/imodfindbeads.cpp`).
//!
//! Every case in `fixtures/imodfindbeads/cases.tsv` was run through the native
//! reference `imodfindbeads` by `fixtures/make-imodfindbeads-goldens.sh`,
//! stdout captured through a pipe.  The exit status is `golden/<case>.rc`,
//! stdout `golden/<case>.out`, and every file native left behind is in
//! `golden/<case>/`.  The input is a seeded synthetic 160x160x3 byte stack of
//! dark gold-like beads written by the native `raw2mrc`, with boundary,
//! reference and add-to models, a prealignment file and tilt angles.
//!
//! Upstream defects are fixed in the translation (`BUGS.md`, imodfindbeads),
//! so a case whose output a fix changes has its expectation in `defined/`
//! (written from the fixed translation by `make-imodfindbeads-goldens.sh
//! defined`).  One fix reaches every two-pass case: `extractDiameter` now fits
//! all the rings around the steepest gradient (the source never advanced
//! `nfit` and always fitted two points), so the printed "Diameter of average
//! bead at zero-crossing" differs from native wherever it is printed.  That
//! value is masked when comparing with a native golden; it only changes other
//! output with `-adjust`, and the `adjust` case holds it in `defined/`.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/imodfindbeads")
}

fn is_input(file: &str) -> bool {
    [".mrc", ".mod", ".prexg", ".tlt"]
        .iter()
        .any(|ext| file.ends_with(ext))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-imodfindbeads-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_str().unwrap().to_string();
        if path.is_file() && is_input(&file) {
            std::fs::copy(&path, dir.join(&file)).unwrap();
        }
    }
    dir
}

/// Blanks the number after `Diameter of average bead at zero-crossing = `
/// (see the module comment) and the usage banner's build date.
fn mask_diameter(bytes: &[u8]) -> Vec<u8> {
    const KEY: &str = "Diameter of average bead at zero-crossing = ";
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::new();
    for line in text.split_inclusive('\n') {
        if let Some(k) = line.find(KEY) {
            out.push_str(&line[..k + KEY.len()]);
            out.push_str("#\n");
        } else if line.starts_with("imodfindbeads Version") {
            out.push_str("imodfindbeads Version\n");
        } else {
            out.push_str(line);
        }
    }
    out.into_bytes()
}

fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    common::imod_cmd("imodfindbeads")
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let (name, args) = line.split_once('\t').unwrap();
        let defined_dir = fixture_dir().join("defined");
        let defined = common::golden::exists(&defined_dir.join(format!("{name}.rc")));
        let golden = if defined {
            defined_dir
        } else {
            fixture_dir().join("golden")
        };
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let output = run(&dir, &args);
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, expected {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout_ok = if defined {
            expected_out.matches_masked(&output.stdout, common::golden::identity)
        } else {
            expected_out.matches_masked(&output.stdout, mask_diameter)
        };
        if !stdout_ok {
            failures.push(format!(
                "{name}: stdout differs\n--- expected\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| !is_input(f) || f == "o.mod" || f == "f.mrc")
            .collect();
        produced.sort();
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, expected {expected_files:?}"
            ));
        }
        for file in &expected_files {
            let Ok(ours) = std::fs::read(dir.join(file)) else {
                continue;
            };
            let theirs = common::golden::expect(&golden.join(name).join(file));
            if let Err(why) = theirs.compare(&ours, mask, true) {
                failures.push(format!("{name}: {file} differs: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 25, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The direct-call API autofidseed uses: the reports recorded by
/// `imodfindbeads_recording` are the values of the last lines printed.
#[test]
fn recording_reports_what_autofidseed_parses() {
    use imod_rs::imod::imodutil::imodfindbeads::{
        FindbeadsReport, FindbeadsResult, imodfindbeads_recording,
    };
    use std::sync::{Arc, Mutex};
    let dir = scratch("recording");
    let input = dir.join("fb.mrc").to_str().unwrap().to_string();
    let model = dir.join("o.mod").to_str().unwrap().to_string();
    let area = dir.join("area.mod").to_str().unwrap().to_string();
    // SAFETY: set before the in-process runs below start their threads; the
    // test harness runs this suite with --test-threads=1.
    unsafe {
        std::env::set_var(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        );
    }
    let native = run(
        &dir,
        &[
            "-input", &input, "-output", &model, "-size", "8", "-store", "-1", "-guess", "20",
        ],
    );
    let text = String::from_utf8_lossy(&native.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    let last = lines[lines.len() - 1];
    let total: i32 = last.split_whitespace().next().unwrap().parse().unwrap();

    let sink = Arc::new(Mutex::new(FindbeadsResult::default()));
    let recorder = Arc::clone(&sink);
    let words = [
        "imodfindbeads",
        "-input",
        input.as_str(),
        "-output",
        model.as_str(),
        "-size",
        "8",
        "-store",
        "-1",
        "-guess",
        "20",
    ];
    let (status, _, _) = imod_rs::imod::commands::call_in_process(&words, None, true, move || {
        imodfindbeads_recording(recorder)
    })
    .unwrap();
    assert_eq!(status, 0);
    let reports = sink.lock().unwrap().reports.clone();
    let n = reports.len();
    assert!(n >= 2, "{reports:?}");
    assert!(
        matches!(reports[n - 2], FindbeadsReport::PeaksAboveDip { .. }),
        "{reports:?}"
    );
    match &reports[n - 1] {
        FindbeadsReport::TotalPeaksStored { num, text } => {
            assert_eq!(*num, total);
            assert!(text.starts_with("total peaks being stored"), "{text}");
        }
        other => panic!("last report {other:?}"),
    }

    let sink = Arc::new(Mutex::new(FindbeadsResult::default()));
    let recorder = Arc::clone(&sink);
    let words = [
        "imodfindbeads",
        "-input",
        input.as_str(),
        "-area",
        area.as_str(),
        "-query",
        "1",
    ];
    let (status, _, _) = imod_rs::imod::commands::call_in_process(&words, None, true, move || {
        imodfindbeads_recording(recorder)
    })
    .unwrap();
    assert_eq!(status, 0);
    let reports = sink.lock().unwrap().reports.clone();
    assert!(
        matches!(reports.as_slice(), [FindbeadsReport::Area(a)] if (*a - 0.0064).abs() < 1e-9),
        "{reports:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
