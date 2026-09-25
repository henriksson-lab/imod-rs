//! Native-golden coverage for `blendmont` (`IMOD/flib/blend/blendmont.f90`).
//!
//! Every case in `fixtures/blendmont/cases.tsv` was run through the native
//! reference `blendmont` at `OMP_NUM_THREADS=1` by
//! `fixtures/make-blendmont-goldens.sh`, stdout captured through a pipe.  The
//! exit status is `golden/<case>.rc`, stdout is `golden/<case>.out`, and every
//! file native left behind is in `golden/<case>/`.  The input
//! (`make-blendmont-inputs.py`) is a seeded synthetic 3 x 3 montage of two
//! sections written by the native `raw2mrc`; the `old.*` edge files the reuse
//! cases read were made by the native program too.  The cases cover plain
//! blending, sloppy and piece-shifting modes (edges, correlations, robust
//! fitting, alternative peaks), output modes, floating, binning, windows and
//! multiple output frames, section lists, aligned coordinates, test mode, edge
//! functions only, intensity scaling with and without gradients (including
//! `-sum`, whose `clip plane` runs in process), mag gradients, g transforms,
//! an exclusion model, multiple negatives, old edge functions, read-in and
//! expected correlations, parallel setup and header writing, and error exits.
//! The montage's overlaps are wide enough that no edge grid has a single row
//! or column, which keeps every case clear of the source's reads of
//! uninitialised memory (`BUGS.md`, blendmont).

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/blendmont")
}

/// Blank what native cannot reproduce: an MRC file's label date/time stamps,
/// and the label slots past `nlabl`, which are stack residue (`BUGS.md` §2).
fn mask(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.len() >= 1024 && &masked[208..212] == b"MAP " {
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

fn scratch(name: &str) -> (PathBuf, Vec<String>) {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-blendmont-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut inputs = Vec::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file == "cases.tsv" || file == "excl.txt" {
                continue;
            }
            std::fs::copy(&path, dir.join(&file)).unwrap();
            inputs.push(file);
        }
    }
    (dir, inputs)
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
        let (dir, inputs) = scratch(name);
        let output = common::imod_cmd("blendmont")
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
        // Compared as a multiset of lines: the lines `dopen` and the image
        // open print go to a stream of their own here, so their order against
        // the program's C-stdio lines can differ from native's (messages need
        // not interleave the same way; the lines themselves must all be there).
        let lines = |b: &[u8]| -> Vec<String> {
            let mut v: Vec<String> = String::from_utf8_lossy(b)
                .lines()
                .map(str::to_owned)
                .collect();
            v.sort();
            v
        };
        let (ours, theirs) = (lines(&output.stdout), lines(&expected_out));
        if ours != theirs {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_out),
                String::from_utf8_lossy(&output.stdout)
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
    assert!(count >= 38, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
