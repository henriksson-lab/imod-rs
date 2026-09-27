//! Native-golden coverage for `imodmop` (`IMOD/imodutil/imodmop.c`).
//!
//! Every row of `fixtures/imodmop/cases.tsv` was run through the native
//! reference program by `fixtures/make-imodmop-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.mrc` the output image, when native
//! wrote one.  Images are compared byte for byte with the label time stamps
//! masked and unused label slots reconciled (`BUGS.md` §2); standard output
//! with version-banner dates and label stamps masked.  The `-project` row
//! runs xyzproj: natively through `system()`, here in process.
//!
//! Not a row: `-noise`, whose `gaussianDeviate(-1)` seeds from the clock, so
//! native differs from itself run to run (checked: 273 bytes over two native
//! runs).  Pruned: 21 of 23 rows kept (dropped: -mask, a second limit error).

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "imodmop";

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

/// Banner dates, `dd-Mmm-yy  HH:MM:SS` stamps, and the process id in the
/// `-project` temporary file names (`imodmop.rec0.<pid>`).
fn mask_stdout(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(&common::mask_stamps(&mask_banner(bytes))).into_owned();
    let mut out = String::new();
    let mut rest = text.as_str();
    while let Some(k) = rest.find("imodmop.") {
        let (head, tail) = rest.split_at(k + 8);
        out.push_str(head);
        let tail = if tail.starts_with("rec0.") || tail.starts_with("xyz0.") {
            out.push_str(&tail[..5]);
            out.push_str("PID");
            tail[5..].trim_start_matches(|c: char| c.is_ascii_digit())
        } else {
            tail
        };
        rest = tail;
    }
    out.push_str(rest);
    out.into_bytes()
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
        if !stdout.matches_masked(&output.stdout, mask_stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        for ext in ["mrc"] {
            let expected = common::golden::load(&golden.join(format!("{name}.{ext}")));
            let written = std::fs::read(dir.join(format!("o.{ext}"))).ok();
            match (expected, written) {
                (None, None) => {}
                (Some(g), Some(w)) if g.matches_reconciled(&w, common::mask_stamps) => {}
                (g, w) => failures.push(format!(
                    "{name}: o.{ext} differs (native {:?}, ours {:?} bytes)",
                    g.map(|b| b.describe()),
                    w.map(|b| b.len())
                )),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 21, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
