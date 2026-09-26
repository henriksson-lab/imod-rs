//! Native-golden coverage for `combinefft` (`IMOD/flib/image/combinefft.f90`).
//!
//! Every row of `fixtures/combinefft/cases.tsv` was run through the native
//! reference program by `fixtures/make-combinefft-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output, and `.out` the output file (`out.fft`,
//! `out.mrc`, or the second input `bb.fft` rewritten in place) when native
//! wrote one.  Output files are compared byte for byte with the label
//! date/time stamps masked (`common::mask_stamps`).  Standard output is
//! compared as a multiset of lines: the header dumps `irdhdr` writes through
//! libc stdout and the gfortran unit-6 lines interleave differently in the
//! two programs, and message order is not part of the acceptance target
//! (CLAUDE.md), while every line native prints must still be printed.
//!
//! Pruned 2026-09-26: 27 of 30 rows kept (dropped: -weight with a tilt file (the -weight/-both/-highest case is kept), interactive output into B, and the short inverse-xf read error (bad.xf kept)); the rest stay in cases.tsv as `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "combinefft";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinefft")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|n| n != "cases.tsv" && n != "golden.manifest")
        {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    std::fs::copy(fixture_dir().join("b.fft"), dir.join("bb.fft")).unwrap();
    dir
}

/// The `printf` escapes the golden script feeds the interactive cases.
fn unescape(text: &str) -> String {
    if text == "-" {
        return String::new();
    }
    text.replace("\\n", "\n")
}

/// Standard output with its lines sorted, stamps masked.
fn lines(bytes: &[u8]) -> Vec<u8> {
    let mut lines: Vec<Vec<u8>> = common::mask_stamps(bytes)
        .split(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    lines.sort();
    lines.join(&b'\n')
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
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
            .env("OMP_NUM_THREADS", "1")
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
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches_masked(&output.stdout, lines) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let mut written = None;
        for out in ["out.fft", "out.mrc"] {
            if let Ok(bytes) = std::fs::read(dir.join(out)) {
                written = Some(bytes);
            }
        }
        let bb = std::fs::read(dir.join("bb.fft")).unwrap();
        if bb != std::fs::read(fixture_dir().join("b.fft")).unwrap() {
            written = Some(bb);
        }
        let expected = common::golden::load(&golden.join(format!("{name}.out")));
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches_masked(&w, common::mask_stamps) => {}
            (g, w) => failures.push(format!(
                "{name}: output differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 27, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs the program in a fresh fixture copy with the given environment
/// choice (`autodoc` false clears `AUTODOC_DIR` and `IMOD_DIR`, forcing the
/// fallback option table), returning exit status and stdout.
fn run(name: &str, args: &[&str], autodoc: bool) -> (Option<i32>, String) {
    let dir = scratch(name);
    let mut cmd = common::imod_cmd(PROGRAM);
    cmd.current_dir(&dir).args(args).env("OMP_NUM_THREADS", "1");
    if autodoc {
        cmd.env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        );
    } else {
        cmd.env_remove("AUTODOC_DIR").env_remove("IMOD_DIR");
    }
    let output = cmd.stdin(Stdio::null()).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// BUGS.md: native's fallback table lacks `LockFileForHDF`, so without the
/// autodoc a chunked-HDF run stopped with "Illegal option".  Defined
/// behaviour: the fallback table has all 30 fields and the run behaves as it
/// does with the autodoc.
#[test]
fn fallback_table_reads_lock_file_option() {
    let args = [
        "-ainput",
        "a.mrc",
        "-binput",
        "b.mrc",
        "-output",
        "out.mrc",
        "-inverse",
        "inv.xf",
        "-atiltfile",
        "a.tlt",
        "-btiltfile",
        "b.tlt",
        "-xsave",
        "0,5",
        "-ysave",
        "0,5",
        "-zsave",
        "0,3",
        "-place",
        "0,0,0",
        "-lock",
        "l.lock",
    ];
    let with = run("lock_adoc", &args, true);
    let without = run("lock_noadoc", &args, false);
    assert!(!without.1.contains("Illegal option"), "{}", without.1);
    // Past PIP's missing-autodoc warning the two runs are the same.
    let marker = " Using fallback options in main program\n";
    let k = without.1.find(marker).expect("fallback warning") + marker.len();
    assert_eq!(with.0, without.0);
    assert_eq!(with.1, without.1[k..]);
}

/// BUGS.md: native computed the ring count before validating the width, so
/// a tiny positive width gave an INT_MIN count and carried on.  Defined
/// behaviour: the width is checked first and a count too large for the
/// arrays is reported as such.
#[test]
fn ring_width_validated_before_ring_count() {
    let base = [
        "-ainput",
        "a.fft",
        "-binput",
        "b.fft",
        "-output",
        "out.fft",
        "-inverse",
        "inv.xf",
        "-atiltfile",
        "a.tlt",
        "-btiltfile",
        "b.tlt",
        "-reduce",
        "0.5",
        "-ring",
    ];
    let mut args = base.to_vec();
    args.push("0");
    let (rc, out) = run("ring0", &args, true);
    assert_eq!(rc, Some(1));
    assert!(out.contains("Illegal entry for ring width"), "{out}");
    let mut args = base.to_vec();
    args.push("1e-30");
    let (rc, out) = run("ringtiny", &args, true);
    assert_eq!(rc, Some(1));
    assert!(out.contains("Too many rings"), "{out}");
    let mut args = base.to_vec();
    args.push("-1e-3");
    args.extend(["-radius", "1.0"]);
    let (rc, out) = run("ringneg", &args, true);
    assert_eq!(rc, Some(1));
    assert!(out.contains("Illegal entry for ring width"), "{out}");
}
