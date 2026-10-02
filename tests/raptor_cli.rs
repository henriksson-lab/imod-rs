//! Native-golden coverage for `RAPTOR` (`IMOD/raptor/main.cpp` and the units
//! it links, translated in `src/imod/raptor`) and the `MarkersCorrespond`
//! program it runs.
//!
//! Every row of `fixtures/raptor/cases.tsv` was run through the native
//! reference RAPTOR by `fixtures/make-raptor-goldens.sh` on the synthetic
//! tilt series `fixtures/raptor/syn1.mrc`, in a fresh directory, with
//! standard output captured through a pipe: `golden/<case>.rc` is the exit
//! status, `.out` the standard output and `golden/<case>/` every file RAPTOR
//! left under its output directory.  The cases in `defined.list` carry the
//! defined behaviour of upstream bugs fixed in translation (`BUGS.md`,
//! RAPTOR) -- chiefly the `.cfg` marker count, which changes the model of
//! nearly every run -- and were checked against a native RAPTOR with those
//! fixes patched in (`make-raptor-goldens.sh patched`, 2026-09-27: every
//! file identical but `edgepeak`, whose native run reads past an image).
//! Masked on both sides: the `strftime("%c")` stamps and the argv line of the
//! log (`argv[0]` is the path the program was started under) and the integer
//! seconds of the timing lines, which depend on the clock.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("raptor")
}

/// Every file under `dir`, as paths relative to it.
fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(
                    path.strip_prefix(dir)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    out.sort();
    out
}

/// Masks what differs between two runs of the same program: `%c` date
/// stamps, the line after "RAPTOR called with the following command:", and
/// "took N sec(ond)s".
fn mask(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    let date = regex::Regex::new(r"[A-Z][a-z]{2} [A-Z][a-z]{2} [ 0-9][0-9] [0-9:]{8} [0-9]{4}")
        .unwrap();
    let took = regex::Regex::new(r"took -?[0-9]+ sec").unwrap();
    let mut out = String::new();
    let mut skip = false;
    for line in text.split_inclusive('\n') {
        if skip {
            out.push_str("ARGV\n");
            skip = false;
            continue;
        }
        if line.starts_with("RAPTOR called with the following command:") {
            skip = true;
        }
        let line = date.replace_all(line, "DATE");
        out.push_str(&took.replace_all(&line, "took N sec"));
    }
    out.into_bytes()
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, opts, drop) = (fields[0], fields[1], fields[2]);
        let dir = std::env::temp_dir().join(format!(
            "imod-rs-raptor-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for input in ["syn1.mrc", "syn1.rawtlt"] {
            if input != drop {
                std::fs::copy(fixture_dir().join(input), dir.join(input)).unwrap();
            }
        }
        let output = common::imod_cmd("RAPTOR")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(["-exec", ".", "-path", ".", "-inp", "syn1.mrc", "-out", "r"])
            .args(opts.split_whitespace())
            .output()
            .unwrap();
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.out")));
        if let Err(e) = stdout.compare(&output.stdout, identity_mask, false) {
            failures.push(format!("{name}: stdout: {e}"));
        }
        // RAPTOR writes two levels deep (`IMOD/`, `align/`, `debug/`, `temp/`).
        let mut expected: Vec<String> = Vec::new();
        for top in common::golden::list(&golden.join(name)) {
            for file in common::golden::list(&golden.join(name).join(&top)) {
                expected.push(format!("{top}/{file}"));
            }
        }
        let ours = files_under(&dir.join("r"));
        let mut exp_sorted = expected.clone();
        exp_sorted.sort();
        if exp_sorted != ours {
            failures.push(format!(
                "{name}: files differ\n  native: {exp_sorted:?}\n  ours:   {ours:?}"
            ));
        }
        for file in &expected {
            let Ok(bytes) = std::fs::read(dir.join("r").join(file)) else {
                continue;
            };
            let g = common::golden::expect(&golden.join(name).join(file));
            if let Err(e) = g.compare(&bytes, mask, false) {
                failures.push(format!("{name}: {file}: {e}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn identity_mask(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}
