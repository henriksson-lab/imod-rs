//! Native-golden coverage for `findsection` (`IMOD/imodutil/findsection.cpp`).
//!
//! Every case in `fixtures/findsection/cases.tsv` was run through the native
//! reference `findsection` by `fixtures/make-findsection-goldens.sh`, stdout
//! captured through a pipe.  The exit status is `golden/<case>.rc`, stdout is
//! `golden/<case>.out`, and every file native left behind is in
//! `golden/<case>/`.  The inputs are seeded synthetic slabs written by the
//! native `raw2mrc`, a bead model made by the native `findsection` and
//! `imodtrans -i`, and `fixtures/imodtrans/multi.mod` as a model with no
//! `IrefImage`.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/findsection")
}

/// Blank what native cannot reproduce.  A model's 128-byte name field
/// (`Imod.name`, bytes 8..136) is `malloc` residue past its terminator in
/// native (`imodNew`), and an MRC file's labels carry a date/time stamp.
fn mask(name: &str, bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.starts_with(b"IMOD") && masked.len() > 136 {
        let end = masked[8..136].iter().position(|&b| b == 0).unwrap_or(128);
        masked[8 + end..136].fill(0);
    } else if (name.ends_with(".means") || name.ends_with(".SDs") || name.ends_with(".colmed"))
        && masked.len() > 1024
    {
        masked[224..1024].fill(0);
    }
    masked
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-findsection-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.ends_with(".mrc") || file.ends_with(".mod") {
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
        let output = common::imod_cmd("findsection")
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
        // date (`imodVersion`); only the lines after it are compared.
        let (ours, theirs) = if expected_out.starts_with(b"findsection Version") {
            let skip = |b: &[u8]| -> Vec<u8> {
                b.iter()
                    .position(|&c| c == b'\n')
                    .map_or(Vec::new(), |p| b[p + 1..].to_vec())
            };
            (skip(&output.stdout), skip(&expected_out))
        } else {
            (output.stdout.clone(), expected_out)
        };
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
            if mask(file, &ours) != mask(file, &theirs) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 18, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
