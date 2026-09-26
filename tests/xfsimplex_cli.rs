//! Native-golden coverage for `xfsimplex` (`IMOD/flib/image/xfsimplex.f90`,
//! with `simplexdiff.c`).
//!
//! Every row of `fixtures/xfsimplex/cases.tsv` was run through the native
//! reference program by `fixtures/make-xfsimplex-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output (the search trace and final values) and
//! `.out.<file>` each file the run created.  Inputs (`make-inputs.sh`): a
//! seeded smooth image pair, B rotated, stretched and shifted from A, as
//! float and short MRC written by the native raw2mrc, and transform files.
//! Exit status, standard output and every output file must match byte for
//! byte.

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "xfsimplex";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(PROGRAM)
}

/// The fixture inputs: the regular files of the fixture directory other than
/// the case table and the generator scripts.
fn inputs() -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !path.is_file() || name == "cases.tsv" || name.starts_with("make-") {
            continue;
        }
        map.insert(name, std::fs::read(&path).unwrap());
    }
    map
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let inputs = inputs();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
        let dir =
            std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (file, bytes) in &inputs {
            std::fs::write(dir.join(file), bytes).unwrap();
        }
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let stdin = if stdin == "-" {
            String::new()
        } else {
            stdin.replace("\\n", "\n")
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
            .write_all(stdin.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
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
        let stdout = std::fs::read(golden.join(format!("{name}.stdout"))).unwrap();
        if output.stdout != stdout {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let prefix = format!("{name}.out.");
        let mut expected = BTreeMap::new();
        for entry in std::fs::read_dir(&golden).unwrap() {
            let file = entry.unwrap().file_name().to_string_lossy().into_owned();
            if let Some(out) = file.strip_prefix(&prefix) {
                expected.insert(out.to_owned(), std::fs::read(golden.join(&file)).unwrap());
            }
        }
        let mut written = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).unwrap();
            if inputs.get(&file) != Some(&bytes) {
                written.insert(file, bytes);
            }
        }
        if expected.keys().ne(written.keys()) {
            failures.push(format!(
                "{name}: output files {:?}, native {:?}",
                written.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            ));
        }
        for (file, bytes) in &expected {
            if let Some(ours) = written.get(file) {
                if common::mask_stamps(ours) != common::mask_stamps(bytes) {
                    failures.push(format!("{name}: {file} differs from native"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 20, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// BUGS.md, fixed in translation: `xfsimplex.f90:383` compares A's Y size
/// with itself, so native never checks B's height and reads a shorter B
/// with A's size.  The translation refuses a B of different height.
#[test]
fn different_y_size_is_refused() {
    let dir = std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-ysize", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in &inputs() {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    let raw: Vec<u8> = (0..96 * 70 * 2)
        .flat_map(|index| ((index % 97) as f32).to_le_bytes())
        .collect();
    std::fs::write(dir.join("by.raw"), &raw).unwrap();
    assert!(
        common::imod_cmd("raw2mrc")
            .current_dir(&dir)
            .args([
                "-x", "96", "-y", "70", "-z", "2", "-t", "float", "by.raw", "by.mrc"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-aimage", "a.mrc", "-bimage", "by.mrc", "-output", "o.xf"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(text.contains("must be the same size in X and Y"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
