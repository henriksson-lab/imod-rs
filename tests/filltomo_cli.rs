//! Native-golden coverage for `filltomo` (`IMOD/flib/model/filltomo.f90`).
//!
//! Every row of `fixtures/filltomo/cases.tsv` was run through the native
//! reference program by `fixtures/make-filltomo-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created
//! or changed -- here the fill tomogram, which filltomo rewrites in place.
//! Inputs: seeded volumes from the native raw2mrc, the volume to fill and its
//! inverse transform from the native matchvol, and boundary models from the
//! native point2model (`make-inputs.sh`).
//! Exit status, standard output and every output file must match byte for
//! byte, except the `hh:mm:ss` time and the date of a `dd-Mmm-yy  hh:mm:ss`
//! stamp in an MRC label, which differ between runs.
//!
//! Pruned 2026-09-26: 21 of 23 rows kept (dropped: interactive numeric-size entry and the -sxform/-mxform run with few transforms (f_bothx and f_fewxf kept)); the rest stay in cases.tsv as `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "filltomo";

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
    map
}

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
    assert!(count >= 20, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs the Rust program on copies of the fixture inputs for a case that
/// reaches an upstream bug fixed in translation (BUGS.md), with a deadline
/// (a regression could hang).  Returns the exit status, stdout and the
/// scratch directory.
fn run_fixed(name: &str, args: &[&str]) -> (Option<i32>, String, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-{PROGRAM}-fixed-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in &inputs() {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    let mut child = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > std::time::Duration::from_secs(60) {
            let _ = child.kill();
            panic!("{name}: still running after 60 s");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        dir,
    )
}

/// A transform file with no transforms: native averages the uninitialised
/// first slot of `flist` (`filltomo.f90:340-343`).  Fixed in translation:
/// the program stops with a message.
#[test]
fn empty_stack_transform_file_is_an_error() {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-emptyxf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("empty.xf"), b"").unwrap();
    let (rc, stdout, run) = run_fixed(
        "emptyxf",
        &[
            "-matched",
            "mat.mrc",
            "-fill",
            "fill.mrc",
            "-source",
            "src.mrc",
            "-inverse",
            "inv.xf",
            "-sraw",
            "22,12,16",
            "-sxform",
            dir.join("empty.xf").to_str().unwrap(),
        ],
    );
    assert_eq!(rc, Some(1), "{stdout}");
    assert!(
        stdout.contains("The Source stack transform file has no transforms"),
        "{stdout}"
    );
    // The file to fill is untouched.
    assert_eq!(
        std::fs::read(run.join("fill.mrc")).unwrap(),
        std::fs::read(fixture_dir().join("fill.mrc")).unwrap()
    );
}
