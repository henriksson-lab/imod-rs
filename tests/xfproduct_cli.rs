//! Native-golden coverage for `xfproduct` (`IMOD/flib/image/xfproduct.f`).
//!
//! Every `xfproduct` row of `fixtures/xftransforms/cases.tsv` was run through the
//! native reference program by `fixtures/make-xftransforms-goldens.sh`, with
//! standard output captured through a pipe.  `golden/xfproduct-<case>.rc` is the
//! exit status, `.stdout` the standard output (the transform counts, fit
//! coefficients and file-open lines are all byte-identical, so the whole
//! stream is compared) and `.out` the transform or warping file native
//! wrote, when it wrote one.  Inputs: the seeded linear, grid-warping and
//! control-point files in `fixtures/xftransforms`, plus three `.xf` files
//! from the vendored `IMOD/Etomo/uitestData`.

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "xfproduct";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/xftransforms")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "xf") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData");
    for name in [
        "BB/BBa.xf",
        "midzone2/midzone2a.xf",
        "mediumscansb2/mediumscansb2_midas.xf",
    ] {
        let source = vendored.join(name);
        std::fs::copy(&source, dir.join(source.file_name().unwrap())).unwrap();
    }
    dir
}

/// The `printf` escapes the golden script feeds the interactive cases.
fn unescape(text: &str) -> String {
    if text == "-" {
        return String::new();
    }
    text.replace("\\n", "\n")
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
        let fields: Vec<&str> = line.split('\t').collect();
        let (program, name, args, stdin) = (fields[0], fields[1], fields[2], fields[3]);
        if program != PROGRAM {
            continue;
        }
        let stem = format!("{program}-{name}");
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let mut child = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(unescape(stdin).as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        count += 1;
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{stem}.rc")))
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
        let stdout = std::fs::read(golden.join(format!("{stem}.stdout"))).unwrap();
        if output.stdout != stdout {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected = std::fs::read(golden.join(format!("{stem}.out"))).ok();
        let written = ["o.xg", "BBa.xg", "p.xf"]
            .iter()
            .find_map(|o| std::fs::read(dir.join(o)).ok());
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g == w => {}
            (g, w) => failures.push(format!(
                "{name}: output differs (native {:?} bytes, ours {:?} bytes)",
                g.map(|b| b.len()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 40, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
