//! Native-golden coverage for `sortbeadsurfs`
//! (`IMOD/flib/model/sortbeadsurfs.f90`).
//!
//! Every row of `fixtures/sortbeadsurfs/cases.tsv` was run through the
//! native reference program by `fixtures/make-sortbeadsurfs-goldens.sh`, with
//! standard output captured through a pipe.  `golden/<case>.rc` is the exit
//! status, `.stdout` the standard output, `.mod` the output model and `.txt`
//! the surface text file, when native wrote one.  Models are compared byte
//! for byte with `Imod.name` past its terminator reconciled (`BUGS.md` §2);
//! standard output with the usage banner's compile date masked.  The
//! `-StandardInput` form autofidseed uses was checked against native by hand.
//!
//! Pruned: 14 of 15 rows kept (dropped: plain sorting without options, which
//! `majority` and `setparam` cover); the rest stay as `#full` rows.

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "sortbeadsurfs";

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
    assert!(count >= 14, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
