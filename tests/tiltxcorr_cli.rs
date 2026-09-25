//! Native-golden coverage for `tiltxcorr` (`IMOD/imodutil/tiltxcorr.cpp`).
//!
//! Every case in `fixtures/tiltxcorr/cases.tsv` was run through the native
//! reference `tiltxcorr` at `OMP_NUM_THREADS=1` by
//! `fixtures/make-tiltxcorr-goldens.sh`, stdout captured through a pipe.  The
//! exit status is `golden/<case>.rc`, stdout is `golden/<case>.out`, and every
//! file native left behind is in `golden/<case>/`.  The input is a seeded
//! synthetic tilt series written by the native `raw2mrc`; the cases cover plain
//! and cumulative correlation, filters, binning and antialiasing, skipping and
//! breaking, the mag search and rotation scan, a reference file, boundary
//! models, test output, patch tracking (grid, seed model, prealignment, patch
//! expansion, local domains) and both warp modes, plus error exits.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiltxcorr")
}

/// Blank what native cannot reproduce.  A model's `MINX` chunk carries the
/// `IrefImage` that `putimageref` `malloc`s without setting `oscale` or
/// `orot` (`imodel_fwrap.c:2013-2027`), so those 24 bytes are heap residue in
/// native; an MRC file's labels carry a date/time stamp, and the label slots
/// past `nlabl` are stack residue (`BUGS.md` §2).
fn mask(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.starts_with(b"IMOD") {
        if let Some(pos) = masked.windows(4).position(|w| w == b"MINX") {
            masked[pos + 8..pos + 20].fill(0);
            masked[pos + 32..pos + 44].fill(0);
        }
    } else if masked.len() > 1024 && &masked[208..212] == b"MAP " {
        let nlabl = i32::from_le_bytes(masked[220..224].try_into().unwrap()).max(0) as usize;
        for lab in 0..10 {
            let start = 224 + 80 * lab;
            if lab >= nlabl {
                masked[start..start + 80].fill(0);
            } else {
                masked[start + 55..start + 80].fill(0);
            }
        }
    }
    masked
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-tiltxcorr-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.starts_with("tx.") || file.ends_with(".mod") {
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
        let output = common::imod_cmd("tiltxcorr")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .args(args.split_whitespace())
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // The usage listing starts with a version banner carrying the build
        // date, and the header listing of the input carries raw2mrc's label
        // with its time stamp; the verbose timing line is wall-clock time.
        let clean = |b: &[u8]| -> Vec<u8> {
            String::from_utf8_lossy(b)
                .lines()
                .filter(|l| !l.starts_with("tiltxcorr Version"))
                .filter(|l| !l.starts_with("interp "))
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
        };
        let (ours, theirs) = (clean(&output.stdout), clean(&expected_out));
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
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if mask(&ours) != mask(&theirs) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 30, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
