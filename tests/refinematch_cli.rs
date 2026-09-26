//! Native-golden coverage for `refinematch` (`IMOD/flib/model/refinematch.f90`).
//!
//! Every row of `fixtures/refinematch/cases.tsv` was run through the native
//! reference program by `fixtures/make-refinematch-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created
//! or changed.
//! Inputs: synthetic corrsearch3d-style patch files written by
//! `fixtures/findwarp/make-patches.py`, region models written by the native
//! point2model and an initial 3D transform.  The cases cover every option of
//! `refinematch.adoc`, the one-layer (fixed-column simplex) fit, the
//! interactive dialogue and the error paths.
//! Exit status, standard output and every output file must match byte for
//! byte.
//! Pruned 2026-09-26: 19 of 24 rows kept (dropped: r_prod1, r_onelayer_prod and r_select_pass (option combinations other rows reach), r_prodonly (same error as r_initonly), r_nooutput (same output as r_nonopt)); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "refinematch";

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
        if !path.is_file()
            || name == "cases.tsv"
            || name == "golden.manifest"
            || name.starts_with("make-")
        {
            continue;
        }
        map.insert(name, std::fs::read(&path).unwrap());
    }
    for name in SHARED {
        let path = fixture_dir().join("../findwarp").join(name);
        map.insert(name.to_owned(), std::fs::read(path).unwrap());
    }
    map
}

/// Inputs `fixtures/findwarp` holds for both suites (not stored twice).
const SHARED: [&str; 9] = [
    "badinit2.xf",
    "init.xf",
    "p1.patch",
    "p2.patch",
    "p3.patch",
    "p5.patch",
    "p9.patch",
    "reg1.mod",
    "reg2.mod",
];

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let inputs = inputs();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
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
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let prefix = format!("{name}.out.");
        let mut expected = BTreeMap::new();
        for file in common::golden::list(&golden) {
            if let Some(out) = file.strip_prefix(&prefix) {
                expected.insert(out.to_owned(), common::golden::expect(&golden.join(&file)));
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
        for (file, native) in &expected {
            if let Some(ours) = written.get(file) {
                if let Err(why) = native.compare(ours, common::mask_stamps, false) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 19, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
