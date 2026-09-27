//! Golden coverage for `autofidseed` (`IMOD/pysrc/autofidseed`, translated in
//! `src/imod/pysrc/autofidseed.rs`).
//!
//! Every row of `fixtures/autofidseed/cases.tsv` was run through the native
//! Python script by `fixtures/make-autofidseed-goldens.sh`, with this crate's
//! programs on `PATH` for every program the script runs (see that script
//! for why: the programs have their own native suites, carry fixed upstream
//! bugs, and native beadtrack is not deterministic).  Here `imod
//! autofidseed` runs on the same inputs, calling those programs in process,
//! and must leave the same files with the same bytes, exit with the same
//! status and print the same lines.
//!
//! Masked: the process ID in temporary file names and messages
//! (`afs<pid>.`), the `PID`, `TrackTime`, `ImageTime` and `BoundTime` lines of the info
//! file, MRC label stamps (`common::mask_stamps`), and the order of standard
//! output lines, which differs because the Python script's own prints are
//! block-buffered into the pipe while the programs it runs write directly
//! (`CLAUDE.md`: message interleaving is not an acceptance criterion).

mod common;

use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/autofidseed")
}

/// Splits `text` on blanks, keeping a `'...'` group as one word.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut any = false;
    for c in text.chars() {
        match c {
            '\'' => {
                quoted = !quoted;
                any = true;
            }
            ' ' if !quoted => {
                if any {
                    out.push(std::mem::take(&mut current));
                    any = false;
                }
            }
            _ => {
                current.push(c);
                any = true;
            }
        }
    }
    if any {
        out.push(current);
    }
    out
}

/// `afs<digits>.` -> `afsPID.`
fn mask_pid(text: &str) -> String {
    let re = regex::Regex::new(r"afs[0-9]+\.").unwrap();
    re.replace_all(text, "afsPID.").into_owned()
}

fn mask_file(name: &str, bytes: &[u8]) -> Vec<u8> {
    if name.ends_with(".info") {
        let text = String::from_utf8_lossy(bytes);
        return text
            .lines()
            .map(|line| {
                if line.starts_with("PID ")
                    || line.starts_with("TrackTime ")
                    || line.starts_with("BoundTime ")
                {
                    line.split(' ').next().unwrap().to_owned()
                } else if line.starts_with("ImageTime ") {
                    "ImageTime".to_owned()
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
    }
    common::mask_stamps(bytes)
}

fn sorted_lines(bytes: &[u8]) -> Vec<String> {
    let mut lines: Vec<String> = mask_pid(&String::from_utf8_lossy(bytes))
        .lines()
        .map(str::to_owned)
        .collect();
    lines.sort();
    lines
}

fn walk(dir: &Path, base: &Path, files: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            walk(&path, base, files);
        } else {
            files.push(
                path.strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
}

#[test]
fn every_case_matches_golden() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let golden = fixture().join("golden");
    let table = std::fs::read_to_string(fixture().join("cases.tsv")).unwrap();
    let inputs = ["track.com", "small.st", "small.rawtlt", "bound.mod"];
    let mut failures = Vec::new();
    let mut count = 0;
    for row in common::golden::case_rows(&table) {
        let fields: Vec<&str> = row.split('\t').collect();
        let (name, first, args) = (fields[0], fields[1], fields[2]);
        count += 1;
        let work =
            std::env::temp_dir().join(format!("imod-rs-autofidseed-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        for input in inputs {
            std::fs::copy(fixture().join(input), work.join(input)).unwrap();
        }
        let run = |args: &str| {
            common::imod_cmd("autofidseed")
                .current_dir(&work)
                .env("IMOD_DIR", root.join("IMOD"))
                .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
                .env_remove("IMOD_OUTPUT_FORMAT")
                .env_remove("PARALLEL_BOUNDARY_SIZE")
                .env_remove("RUNCMD_VERBOSE")
                .args(words(args))
                .output()
                .unwrap()
        };
        if first != "-" {
            run(first);
        }
        let output = run(args);
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, golden {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected_out = common::golden::read(&golden.join(format!("{name}.out")));
        if sorted_lines(&expected_out) != sorted_lines(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- ours\n{}\n--- golden\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&expected_out)
            ));
        }
        let mut left = Vec::new();
        walk(&work, &work, &mut left);
        let mut masked: Vec<(String, String)> =
            left.iter().map(|f| (mask_pid(f), f.clone())).collect();
        masked.sort();
        let names: Vec<&str> = masked.iter().map(|(m, _)| m.as_str()).collect();
        let expected_files = common::golden::read_to_string(&golden.join(format!("{name}.files")));
        let expected_files: Vec<&str> = expected_files.lines().collect();
        if names != expected_files {
            failures.push(format!(
                "{name}: files left {names:?}, golden {expected_files:?}"
            ));
        }
        for (masked_name, file) in &masked {
            let ours = std::fs::read(work.join(file)).unwrap();
            match common::golden::load(&golden.join(name).join(masked_name)) {
                Some(expected) => {
                    let mask = |bytes: &[u8]| mask_file(masked_name, bytes);
                    if let Err(why) = expected.compare(&ours, mask, false) {
                        failures.push(format!("{name}: {masked_name}: {why}"));
                    }
                }
                None => {
                    if !inputs.contains(&file.as_str())
                        || std::fs::read(fixture().join(file)).unwrap() != ours
                    {
                        failures.push(format!("{name}: {masked_name} has no golden"));
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&work);
    }
    assert!(count >= 11, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
