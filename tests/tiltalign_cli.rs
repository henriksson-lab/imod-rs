//! Native-golden coverage for `tiltalign` (`IMOD/flib/tiltalign/tiltalign.cpp`).
//!
//! Every row of `fixtures/tiltalign/cases.tsv` was run through the native
//! reference program by `fixtures/make-tiltalign-goldens.sh` in a fresh copy of
//! the inputs, with standard output captured through a pipe.
//! `golden/<case>.rc` is the exit status, `.stdout` the standard output (the
//! variable mappings, minimisation trace, solution and residual tables are all
//! data), and `golden/<case>.<file>` every file the run created.
//!
//! # Why numbers are compared with a tolerance
//!
//! The initial X/Y/Z solution goes through LAPACK `dspsv`
//! (`solve_xyzd.cpp:455`), which this crate replaces with `faer`
//! (`flib/subrs/lapack/dspsv.rs`); every later number is derived from that
//! starting point through `metroSearch`.  `CLAUDE.md` accepts small
//! differences there, so the comparison is: exit status exact; the text split
//! into tokens with the token count and every non-numeric token exact; a
//! numeric token either identical or within `1e-5` relative, or one unit in its
//! last printed decimal (a relative difference far below the print precision
//! can still flip a rounding digit).  Binary model files are compared as
//! 4-byte words: identical, or two finite floats within `1e-5` relative.
//!
//! Measured (2026-09-25, `/big/henriksson/realbench/wave6-tiltalign`): inside
//! real runs the `faer` solutions differ from the reference `dspsv_` by at most
//! `5.0e-14` of the largest component, yet every golden here is byte-identical
//! (stdout apart from the usage banner's build date, and every file), as are 78
//! of the 81 cases of the differential they were taken from (the other three:
//! the banner, and two runs whose native output depends on a heap over-read,
//! `BUGS.md`) and 24 runs at `MaximumCycles` 1 through 1000.  Perturbing the
//! reference solution by `1e-11` relative is still invisible in every native
//! output; `1e-9` is the first level that shows.  So the tolerance is a margin,
//! not a mask: any difference that reaches it is a defect to investigate.

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "tiltalign";
const REL_TOL: f64 = 1e-5;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiltalign")
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
                .is_some_and(|n| n != "cases.tsv" && n != "gen6.py")
        {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    dir
}

/// Decimal places printed in a numeric token, or `None` for a non-number.
fn decimals(token: &str) -> Option<(f64, i32)> {
    let value: f64 = token.parse().ok()?;
    if token.contains(['e', 'E']) || token.contains("inf") || token.contains("nan") {
        return Some((value, 0));
    }
    let places = token
        .find('.')
        .map_or(0, |dot| (token.len() - dot - 1) as i32);
    Some((value, places))
}

/// Compares two text streams under the tolerance in the module doc; returns a
/// description of the first mismatch.
fn compare_text(expected: &[u8], actual: &[u8]) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    let split = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .replace(',', " , ")
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let (want, got) = (split(expected), split(actual));
    if want.len() != got.len() {
        return Err(format!("token count {} vs {}", want.len(), got.len()));
    }
    for (index, (w, g)) in want.iter().zip(&got).enumerate() {
        if w == g {
            continue;
        }
        match (decimals(w), decimals(g)) {
            (Some((a, pa)), Some((b, pb))) => {
                let scale = a.abs().max(b.abs());
                let unit = 10f64.powi(-pa.max(pb)) * 1.000001;
                let diff = (a - b).abs();
                if !(diff <= REL_TOL * scale || diff <= unit) {
                    return Err(format!("token {index}: {w} vs {g}"));
                }
            }
            _ => return Err(format!("token {index}: {w:?} vs {g:?}")),
        }
    }
    Ok(())
}

/// Compares two binary model files word by word under the same tolerance.
fn compare_binary(expected: &[u8], actual: &[u8]) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    if expected.len() != actual.len() {
        return Err(format!("size {} vs {}", expected.len(), actual.len()));
    }
    for (index, (w, g)) in expected.chunks(4).zip(actual.chunks(4)).enumerate() {
        if w == g {
            continue;
        }
        if w.len() != 4 {
            return Err(format!("trailing bytes differ at word {index}"));
        }
        let a = f32::from_be_bytes([w[0], w[1], w[2], w[3]]) as f64;
        let b = f32::from_be_bytes([g[0], g[1], g[2], g[3]]) as f64;
        if !(a.is_finite() && b.is_finite() && (a - b).abs() <= REL_TOL * a.abs().max(b.abs())) {
            return Err(format!("word {index} at byte {}: {a} vs {b}", index * 4));
        }
    }
    Ok(())
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let golden_names: Vec<String> = std::fs::read_dir(&golden)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args) = (fields[0], fields[1]);
        let dir = scratch(name);
        let inputs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let output = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .args(&args)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // The usage banner carries the build date and time (`CLAUDE.md`,
        // "Compile metadata"); only its shape is comparable.
        let unbannered = |bytes: &[u8]| -> Vec<u8> {
            if bytes.starts_with(b"tiltalign Version ") {
                let end = bytes
                    .iter()
                    .position(|&b| b == b'\n')
                    .unwrap_or(bytes.len());
                bytes[end..].to_vec()
            } else {
                bytes.to_vec()
            }
        };
        let want = unbannered(&std::fs::read(golden.join(format!("{name}.stdout"))).unwrap());
        if let Err(why) = compare_text(&want, &unbannered(&output.stdout)) {
            failures.push(format!("{name}: stdout {why}"));
        }
        // Every file native created must exist and match; nothing else may appear.
        let prefix = format!("{name}.");
        let expected_files: Vec<&str> = golden_names
            .iter()
            .filter_map(|g| g.strip_prefix(&prefix))
            .filter(|f| *f != "rc" && *f != "stdout")
            .collect();
        for file in &expected_files {
            let want = std::fs::read(golden.join(format!("{name}.{file}"))).unwrap();
            match std::fs::read(dir.join(file)) {
                Err(_) => failures.push(format!("{name}: {file} not written")),
                Ok(got) => {
                    let result = if file.ends_with(".3dmod") || file.ends_with(".fid") {
                        compare_binary(&want, &got)
                    } else {
                        compare_text(&want, &got)
                    };
                    if let Err(why) = result {
                        failures.push(format!("{name}: {file} {why}"));
                    }
                }
            }
        }
        for entry in std::fs::read_dir(&dir).unwrap() {
            let file = entry.unwrap().file_name().to_string_lossy().into_owned();
            if !inputs.contains(&file) && !expected_files.contains(&file.as_str()) {
                failures.push(format!("{name}: {file} written, native did not"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 16, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
