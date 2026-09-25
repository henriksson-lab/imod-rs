//! Native-golden coverage for `tilt` (`IMOD/flib/tilt/tilt.cpp`).
//!
//! Every case in `fixtures/tilt/cases.tsv` was run through the native
//! reference `tilt` by `fixtures/make-tilt-goldens.sh`, stdout captured
//! through a pipe.  The exit status is `golden/<case>.rc`, stdout is
//! `golden/<case>.out`, and every file native left behind is in
//! `golden/<case>/`.  The inputs are a seeded synthetic aligned tilt series
//! written by the native `raw2mrc`, matching tilt/X-tilt/Z-factor/local
//! alignment files, a reconstruction of it made by the native `tilt`, and a
//! scattered-point model made by the native `point2model`.  An argument column
//! `STDIN:<file>` runs `tilt -StandardInput` with the file on standard input,
//! the way `tilt.com` drives it.

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tilt")
}

/// Replace every `HH:MM:SS` in `bytes` with `XX:XX:XX`.
fn mask_times(bytes: &mut [u8]) {
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let w = &bytes[i..i + 8];
        let digit = |k: usize| w[k].is_ascii_digit();
        if digit(0)
            && digit(1)
            && w[2] == b':'
            && digit(3)
            && digit(4)
            && w[5] == b':'
            && digit(6)
            && digit(7)
        {
            bytes[i..i + 8].copy_from_slice(b"XX:XX:XX");
            i += 8;
        } else {
            i += 1;
        }
    }
}

/// Blank what native cannot reproduce: a model's 128-byte name field
/// (`Imod.name`, bytes 8..136) is `malloc` residue past its terminator in
/// native (`imodNew`), and an MRC file's labels carry a date/time stamp; only
/// the time is masked.
fn mask(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.starts_with(b"IMOD") && masked.len() > 136 {
        let end = masked[8..136].iter().position(|&b| b == 0).unwrap_or(128);
        masked[8 + end..136].fill(0);
    } else if masked.len() >= 1024 {
        mask_times(&mut masked[224..1024]);
    }
    masked
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-tilt-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.starts_with("t.") || file == "pm.mod" || file == "std.com" {
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
        let mut expected_out = std::fs::read(golden.join(format!("{name}.out"))).unwrap();
        let dir = scratch(name);
        let inputs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        let mut command = common::imod_cmd("tilt");
        command
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .env("IMOD_NO_IMAGE_BACKUP", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let stdin_file = args.strip_prefix("STDIN:");
        if stdin_file.is_some() {
            command.arg("-StandardInput").stdin(Stdio::piped());
        } else {
            command.args(args.split_whitespace()).stdin(Stdio::null());
        }
        let mut child = command.spawn().unwrap();
        if let Some(file) = stdin_file {
            let text = std::fs::read(dir.join(file)).unwrap();
            child.stdin.take().unwrap().write_all(&text).unwrap();
        }
        let output = child.wait_with_output().unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // The header listings carry the labels' time stamps.
        let mut ours = output.stdout.clone();
        mask_times(&mut ours);
        mask_times(&mut expected_out);
        if ours != expected_out {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_out),
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
    assert!(count >= 22, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
