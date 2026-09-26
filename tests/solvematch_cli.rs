//! Native-golden coverage for `solvematch` (`IMOD/flib/model/solvematch.f90`).
//!
//! Every row of `fixtures/solvematch/cases.tsv` was run through the native
//! reference program by `fixtures/make-solvematch-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created
//! or changed.  Inputs (`make-inputs.sh`): synthetic dual-axis bead sets in
//! tiltalign point-file format (two surfaces, one surface, inverted, outliers,
//! no object/contour columns, relative and old-style pixel headers, a
//! distorted set for local fits and center shifts, contour numbers past the
//! arrays), 2D fiducial and matching models from the native point2model,
//! transfer-coordinate files, and tiny tomograms for the pixel size.  The
//! cases cover every option of `solvematch.adoc`, interactive entry, and the
//! error exits.  Exit status, standard output and every output file must
//! match byte for byte.

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "solvematch";

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
    assert!(count >= 110, "only {count} cases read");
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

/// `getDelta` reads `-atomogram`/`-btomogram` into `character*80`
/// (`solvematch.f90:1096`), so native opens a longer name truncated to 80
/// characters.  Fixed in translation: the name is used whole, and the run is
/// the `basic` case's.
#[test]
fn tomogram_name_longer_than_80_characters_is_used_whole() {
    let deep = std::env::temp_dir().join(format!(
        "imod-rs-{PROGRAM}-longname-{}/{}",
        std::process::id(),
        "a_directory_name_long_enough_to_push_the_path_past_eighty_characters_by_itself"
    ));
    std::fs::create_dir_all(&deep).unwrap();
    let tomo = deep.join("tiny1.mrc");
    std::fs::copy(fixture_dir().join("tiny1.mrc"), &tomo).unwrap();
    let tomo = tomo.to_str().unwrap().to_owned();
    assert!(tomo.len() > 80);
    let blist = "6,23,10,24,29,13,3,7,21,20,8,12,11,16,14,19,22,5,1,26,15,30,28,25,2,9,17,4,18,27";
    let (rc, stdout, dir) = run_fixed(
        "longname",
        &[
            "-afid", "twoA.xyz", "-bfid", "twoB.xyz", "-alist", "1-30", "-blist", blist, "-atom",
            &tomo, "-btom", &tomo, "-output", "out.xf",
        ],
    );
    let golden = fixture_dir().join("golden");
    assert_eq!(rc, Some(0), "{stdout}");
    assert_eq!(
        stdout,
        String::from_utf8(std::fs::read(golden.join("basic.stdout")).unwrap()).unwrap()
    );
    assert_eq!(
        std::fs::read(dir.join("out.xf")).unwrap(),
        std::fs::read(golden.join("basic.out.out.xf")).unwrap()
    );
}
