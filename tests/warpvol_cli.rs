//! Native-golden coverage for `warpvol` (`IMOD/flib/image/warpvol.f90`, with the shared
//! `rotmatwarpsubs.f90`).
//!
//! Every row of `fixtures/warpvol/cases.tsv` was run through the native
//! reference program by `fixtures/make-warpvol-goldens.sh`, in a fresh
//! directory holding copies of the fixture inputs, with standard output
//! captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.out.<file>` each file the run created
//! or changed.  Inputs: seeded float, short and byte volumes written by the
//! native raw2mrc, 3x4 transform files and findwarp-format warp files
//! (`fixtures/matchvol/make-inputs.sh warpvol`).  The byte volume `mb.mrc`
//! with `-memory` limits forces several cubes, scratch files and the
//! recomposition pass.  The three volumes are `fixtures/matchvol`'s, shared
//! rather than stored twice.
//! Exit status, standard output and every output file must match byte for
//! byte, except the `hh:mm:ss` time and the date of a `dd-Mmm-yy  hh:mm:ss`
//! stamp in an MRC label, the time-derived digits of scratch file names
//! (`<input>.mat<hhmmss>N`, `.wrp<hhmmss>N`) and the verbose timing lines,
//! which differ between runs.
//!
//! Pruned 2026-09-26: 21 of 23 rows kept (dropped: -filled with -patch together, and the interactive size entry); the rest stay in cases.tsv as `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "warpvol";

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
    for name in ["vf.mrc", "vs.mrc", "mb.mrc"] {
        let shared = fixture_dir().join("../matchvol").join(name);
        map.insert(name.to_owned(), std::fs::read(shared).unwrap());
    }
    map
}

/// Masks what differs between two runs of the same program in standard
/// output: the time digits of scratch file names and the timing lines.
fn mask_stdout(data: &[u8]) -> String {
    let text = String::from_utf8_lossy(data);
    let mut out = String::new();
    for line in text.split_inclusive('\n') {
        if line.contains("Wall time") || line.contains("wall time") || line.contains("CPU time") {
            continue;
        }
        let mut line = line.to_string();
        for tag in [".mat", ".wrp"] {
            if let Some(pos) = line.find(tag) {
                let start = pos + tag.len();
                let end = (start + 6).min(line.len());
                if line[start..end].bytes().all(|b| b.is_ascii_digit()) {
                    line.replace_range(start..end, "######");
                }
            }
        }
        out.push_str(&line);
    }
    out
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
        if !stdout.matches_masked(&output.stdout, |b| mask_stdout(b).into_bytes()) {
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

/// BUGS.md: native's 5% transform reserve (`warpvol.f90:278`) truncated
/// `-memory 1` to 0, set up negative cube counts, wrote a bare header and
/// exited 0.  Defined behaviour: a positive limit stays at least 1 MB, so
/// the small volume is warped in one cube and the output equals the native
/// golden of the same run at the default limit (`w_9`).
#[test]
fn memory_limit_of_one_megabyte_warps_the_volume() {
    let dir = std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-mem1", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in &inputs() {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "vf.mrc", "o.mrc", "-xforms", "w9.txt", "-same", "-memory", "1",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rotations done in  1 layers"), "{stdout}");
    let ours = std::fs::read(dir.join("o.mrc")).unwrap();
    let native = common::golden::expect(&fixture_dir().join("golden/w_9.out.o.mrc"));
    assert!(native.matches_masked(&ours, common::mask_stamps));
    let _ = std::fs::remove_dir_all(&dir);
}
